"""Workflow execution engine and context management.

This module provides the execution engine for workflows, coordinating step execution,
caching, callbacks, and state management.

Key Components:
    - WorkflowExecutor: Abstract base for execution engines
    - DefaultWorkflowExecutor: Default sequential execution implementation
    - WorkflowExecutionContext: Runtime context for workflow execution
    - WorkflowExecutionInfo: Metadata about workflow execution

Features:
    - Sequential step execution with dependency resolution
    - Cache integration for step results
    - Callback hooks for monitoring and logging
    - Execution state tracking
    - Git and environment metadata capture

Example:
    >>> from formed.workflow import WorkflowGraph, DefaultWorkflowExecutor
    >>> from formed.workflow.cache import FilesystemWorkflowCache
    >>>
    >>> # Load workflow and create executor
    >>> graph = WorkflowGraph.from_jsonnet("workflow.jsonnet")
    >>> executor = DefaultWorkflowExecutor()
    >>> cache = FilesystemWorkflowCache(".formed/cache")
    >>>
    >>> # Execute workflow
    >>> with executor:
    ...     context = executor(graph, cache=cache)
    >>> print(context.state.status)  # "completed"

"""

import contextvars
import dataclasses
import datetime
from collections.abc import Iterator, Mapping, Sequence
from enum import Enum
from importlib.metadata import version
from logging import getLogger
from types import TracebackType
from typing import Any, NewType, TypeVar

from colt import Registrable

from formed.common.attributeutils import xgetattr
from formed.common.git import GitInfo, get_git_info
from formed.common.pkgutils import PackageInfo, get_installed_packages

from .cache import EmptyWorkflowCache, WorkflowCache
from .callback import EmptyWorkflowCallback, WorkflowCallback
from .graph import WorkflowGraph
from .step import (
    WorkflowStep,
    WorkflowStepContext,
    WorkflowStepInfo,
    WorkflowStepRef,
    WorkflowStepState,
    WorkflowStepStatus,
)

logger = getLogger(__name__)

T = TypeVar("T")
WorkflowExecutorT = TypeVar("WorkflowExecutorT", bound="WorkflowExecutor")

_EXECUTION_CONTEXT = contextvars.ContextVar["WorkflowExecutionContext | None"]("_EXECUTION_CONTEXT", default=None)


WorkflowExecutionID = NewType("WorkflowExecutionID", str)


class WorkflowExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    FAILURE = "failure"
    CANCELED = "canceled"
    COMPLETED = "completed"


@dataclasses.dataclass
class WorkflowExecutionState:
    execution_id: WorkflowExecutionID | None = None
    status: WorkflowExecutionStatus = WorkflowExecutionStatus.PENDING
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None


@dataclasses.dataclass(frozen=True)
class WorkflowExecutionMetadata:
    version: str = version("formed")
    git: GitInfo | None = dataclasses.field(default_factory=get_git_info)
    environment: Mapping[str, str] = dataclasses.field(default_factory=dict)
    required_modules: Sequence[str] = dataclasses.field(default_factory=list)
    dependent_packages: Sequence[PackageInfo] = dataclasses.field(default_factory=get_installed_packages)


@dataclasses.dataclass
class WorkflowExecutionInfo:
    graph: WorkflowGraph

    id: WorkflowExecutionID | None = None
    metadata: WorkflowExecutionMetadata = dataclasses.field(default_factory=WorkflowExecutionMetadata)


@dataclasses.dataclass
class WorkflowExecutionContext:
    info: WorkflowExecutionInfo
    state: WorkflowExecutionState
    cache: WorkflowCache = dataclasses.field(default_factory=EmptyWorkflowCache)
    callback: WorkflowCallback = dataclasses.field(default_factory=EmptyWorkflowCallback)


class WorkflowExecutor(Registrable):
    """Abstract base class for workflow execution engines.

    WorkflowExecutor defines the interface for executing workflows. Subclasses
    implement different execution strategies (sequential, parallel, distributed, etc.).

    The executor can be used as a context manager for resource cleanup.

    Example:
        >>> executor = DefaultWorkflowExecutor()
        >>> with executor:
        ...     context = executor(graph, cache=cache)

    Note:
        Executors are registered and can be instantiated by name via colt.

    """

    def __call__(
        self,
        graph_or_execution: WorkflowGraph | WorkflowExecutionInfo,
        *,
        cache: WorkflowCache | None = None,
        callback: WorkflowCallback | None = None,
    ) -> WorkflowExecutionContext:
        """Execute a workflow.

        Args:
            graph_or_execution: Workflow graph or execution info to execute.
            cache: Optional cache for step results. Defaults to EmptyWorkflowCache.
            callback: Optional callback for monitoring execution. Defaults to EmptyWorkflowCallback.

        Returns:
            Execution context containing state and results.

        """
        raise NotImplementedError

    def __enter__(self: WorkflowExecutorT) -> WorkflowExecutorT:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass


@WorkflowExecutor.register("default")
class DefaultWorkflowExecutor(WorkflowExecutor):
    """Default sequential workflow executor.

    DefaultWorkflowExecutor executes workflow steps sequentially in topological order,
    respecting dependencies. It integrates caching and callbacks, handles errors,
    and tracks execution state.

    Features:
        - Sequential execution with dependency resolution
        - Automatic cache management
        - Callback integration for monitoring
        - Error handling with state tracking
        - Temporary cache for non-cacheable steps

    Example:
        >>> from formed.workflow import WorkflowGraph, DefaultWorkflowExecutor
        >>> from formed.workflow.cache import FilesystemWorkflowCache
        >>> from formed.workflow.callback import LoggingWorkflowCallback
        >>>
        >>> executor = DefaultWorkflowExecutor()
        >>> graph = WorkflowGraph.from_jsonnet("workflow.jsonnet")
        >>> cache = FilesystemWorkflowCache(".formed/cache")
        >>> callback = LoggingWorkflowCallback()
        >>>
        >>> with executor:
        ...     context = executor(graph, cache=cache, callback=callback)
        >>>
        >>> # Check execution status
        >>> print(context.state.status)  # "completed" or "failure"
        >>> print(context.state.execution_id)  # Unique execution ID

    Note:
        - Steps are executed in topological order
        - Cached results are used when fingerprints match
        - Non-deterministic steps always re-execute
        - Steps ending with "!" force re-execution (no temp cache)
        - Callbacks receive step_start/step_finish notifications

    """

    def __call__(
        self,
        graph_or_execution: WorkflowGraph | WorkflowExecutionInfo,
        *,
        cache: WorkflowCache | None = None,
        callback: WorkflowCallback | None = None,
    ) -> WorkflowExecutionContext:
        cache = cache if cache is not None else EmptyWorkflowCache()
        callback = callback if callback is not None else EmptyWorkflowCallback()
        execution_info = (
            graph_or_execution
            if isinstance(graph_or_execution, WorkflowExecutionInfo)
            else WorkflowExecutionInfo(graph_or_execution)
        )

        execution_state = WorkflowExecutionState(
            execution_id=execution_info.id,
            status=WorkflowExecutionStatus.RUNNING,
            started_at=datetime.datetime.now(),
        )
        execution_context = WorkflowExecutionContext(execution_info, execution_state, cache, callback)

        callback.on_execution_start(execution_context)

        # NOTE: execution info can be updated by the callback
        execution_state = dataclasses.replace(execution_state, execution_id=execution_info.id)
        execution_context = dataclasses.replace(execution_context, state=execution_state)

        temporary_cache: dict[WorkflowStepInfo, Any] = {}

        def _run_step(step_info: WorkflowStepInfo[WorkflowStep[T]]) -> T:
            assert cache is not None
            assert callback is not None

            step_state = WorkflowStepState(
                fingerprint=step_info.fingerprint,
                status=WorkflowStepStatus.RUNNING,
                started_at=datetime.datetime.now(),
            )

            step_context = WorkflowStepContext(step_info, step_state)

            result: T

            if step_info in cache:
                logger.info(f"Cached value found for step {step_info.name}")
                result = cache[step_info]
            elif step_info in temporary_cache:
                logger.info(f"Temporary cached value found for step {step_info.name}")
                result = temporary_cache[step_info]
            else:
                try:
                    callback.on_step_start(step_context, execution_context)
                    dependencies: Mapping[int | str | Sequence[int | str], Any] = {
                        path: _run_step(dep) for path, dep in step_info.dependencies
                    }
                    if set(dependencies.keys()) != set(path for path, _ in step_info.dependencies):
                        raise ValueError("Dependencies are not consistent with the graph")

                    step = step_info.step.construct(dependencies)
                    result = step(step_context)

                    if step_info.should_be_cached:
                        cache[step_info] = result
                        if isinstance(result, Iterator):
                            # NOTE: iterator should be restored since it is consumed by the cache
                            result = cache[step_info]
                    elif not step_info.name.endswith("!"):
                        # NOTE: You can force to run the step without caching by adding "!" at the end of the name
                        temporary_cache[step_info] = result
                except KeyboardInterrupt:
                    step_state = dataclasses.replace(step_state, status=WorkflowStepStatus.CANCELED)
                    step_context = dataclasses.replace(step_context, state=step_state)
                    raise
                except Exception as e:
                    step_state = dataclasses.replace(step_state, status=WorkflowStepStatus.FAILURE)
                    step_context = dataclasses.replace(step_context, state=step_state)
                    raise e
                else:
                    step_state = dataclasses.replace(step_state, status=WorkflowStepStatus.COMPLETED)
                    step_context = dataclasses.replace(step_context, state=step_state)
                finally:
                    step_state = dataclasses.replace(step_state, finished_at=datetime.datetime.now())
                    step_context = dataclasses.replace(step_context, state=step_state)
                    callback.on_step_end(step_context, execution_context)

            if isinstance(step_info, WorkflowStepRef) and step_info.fieldref is not None:
                result = xgetattr(result, step_info.fieldref)

            return result

        try:
            ctx = contextvars.copy_context()

            def execute() -> None:
                logger.info("Starting workflow execution %s...", execution_info.id)
                _EXECUTION_CONTEXT.set(execution_context)
                for step_info in execution_info.graph:
                    logger.info("Running step %s...", step_info.name)
                    _run_step(step_info)

            ctx.run(execute)

        except KeyboardInterrupt:
            execution_state = dataclasses.replace(execution_state, status=WorkflowExecutionStatus.CANCELED)
            execution_context = dataclasses.replace(execution_context, state=execution_state)
            raise
        except Exception as e:
            execution_state = dataclasses.replace(execution_state, status=WorkflowExecutionStatus.FAILURE)
            execution_context = dataclasses.replace(execution_context, state=execution_state)
            raise e
        else:
            execution_state = dataclasses.replace(execution_state, status=WorkflowExecutionStatus.COMPLETED)
            execution_context = dataclasses.replace(execution_context, state=execution_state)
        finally:
            execution_state = dataclasses.replace(execution_state, finished_at=datetime.datetime.now())
            callback.on_execution_end(execution_context)

        return dataclasses.replace(execution_context, state=execution_state)


def use_execution_context() -> WorkflowExecutionContext | None:
    return _EXECUTION_CONTEXT.get()
