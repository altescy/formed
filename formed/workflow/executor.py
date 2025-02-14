import contextvars
import dataclasses
import datetime
from collections.abc import Iterator, Mapping, Sequence
from enum import Enum
from importlib.metadata import version
from logging import getLogger
from types import TracebackType
from typing import Any, NewType, Optional, TypeVar, Union

from colt import Registrable

from formed.common.git import GitInfo, get_git_info
from formed.common.pkgutils import PackageInfo, get_installed_packages

from .cache import EmptyWorkflowCache, WorkflowCache
from .callback import EmptyWorkflowCallback, WorkflowCallback
from .graph import WorkflowGraph, WorkflowStepInfo
from .step import WorkflowStep, WorkflowStepContext, WorkflowStepState, WorkflowStepStatus

logger = getLogger(__name__)

T = TypeVar("T")
T_WorkflowExecutor = TypeVar("T_WorkflowExecutor", bound="WorkflowExecutor")

_EXECUTION_CONTEXT = contextvars.ContextVar[Optional["WorkflowExecutionContext"]]("_EXECUTION_CONTEXT", default=None)


WorkflowExecutionID = NewType("WorkflowExecutionID", str)


class WorkflowExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    FAILURE = "failure"
    CANCELED = "canceled"
    COMPLETED = "completed"


@dataclasses.dataclass
class WorkflowExecutionState:
    execution_id: Optional[WorkflowExecutionID] = None
    status: WorkflowExecutionStatus = WorkflowExecutionStatus.PENDING
    started_at: Optional[datetime.datetime] = None
    finished_at: Optional[datetime.datetime] = None


@dataclasses.dataclass(frozen=True)
class WorkflowExecutionMetadata:
    version: str = version("formed")
    git: Optional[GitInfo] = dataclasses.field(default_factory=get_git_info)
    environment: Mapping[str, str] = dataclasses.field(default_factory=dict)
    required_modules: Sequence[str] = dataclasses.field(default_factory=list)
    dependent_packages: Sequence[PackageInfo] = dataclasses.field(default_factory=get_installed_packages)


@dataclasses.dataclass
class WorkflowExecutionInfo:
    graph: WorkflowGraph

    id: Optional[WorkflowExecutionID] = None
    metadata: WorkflowExecutionMetadata = dataclasses.field(default_factory=WorkflowExecutionMetadata)


@dataclasses.dataclass
class WorkflowExecutionContext:
    info: WorkflowExecutionInfo
    state: WorkflowExecutionState
    cache: WorkflowCache = dataclasses.field(default_factory=EmptyWorkflowCache)
    callback: WorkflowCallback = dataclasses.field(default_factory=EmptyWorkflowCallback)


class WorkflowExecutor(Registrable):
    def __call__(
        self,
        graph_or_exection: Union[WorkflowGraph, WorkflowExecutionInfo],
        *,
        cache: Optional[WorkflowCache] = None,
        callback: Optional[WorkflowCallback] = None,
    ) -> WorkflowExecutionContext:
        raise NotImplementedError

    def __enter__(self: T_WorkflowExecutor) -> T_WorkflowExecutor:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        pass


@WorkflowExecutor.register("default")
class DefaultWorkflowExecutor(WorkflowExecutor):
    def __call__(
        self,
        graph_or_exection: Union[WorkflowGraph, WorkflowExecutionInfo],
        *,
        cache: Optional[WorkflowCache] = None,
        callback: Optional[WorkflowCallback] = None,
    ) -> WorkflowExecutionContext:
        cache = cache if cache is not None else EmptyWorkflowCache()
        callback = callback if callback is not None else EmptyWorkflowCallback()
        execution_info = (
            graph_or_exection
            if isinstance(graph_or_exection, WorkflowExecutionInfo)
            else WorkflowExecutionInfo(graph_or_exection)
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

        def _run_step(step_info: WorkflowStepInfo[WorkflowStep[T]]) -> T:
            assert cache is not None
            assert callback is not None

            step_state = WorkflowStepState(
                fingerprint=step_info.fingerprint,
                status=WorkflowStepStatus.RUNNING,
                started_at=datetime.datetime.now(),
            )

            step_context = WorkflowStepContext(step_info, step_state)

            if step_info in cache:
                logger.info(f"Cached value found for step {step_info.name}")
                result = cache[step_info]
            else:
                try:
                    callback.on_step_start(step_context, execution_context)
                    dependencies: Mapping[Union[int, str, Sequence[Union[int, str]]], Any] = {
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


def use_execution_context() -> Optional[WorkflowExecutionContext]:
    return _EXECUTION_CONTEXT.get()
