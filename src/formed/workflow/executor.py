"""Workflow execution engine and context management.

This module provides the execution engine for workflows, coordinating step execution,
caching, callbacks, and state management.

Key Components:
    - `WorkflowExecutor`: Abstract base for execution engines
    - `DefaultWorkflowExecutor`: Default sequential execution implementation
    - `WorkflowExecutionContext`: Runtime context for workflow execution
    - `WorkflowExecutionInfo`: Metadata about workflow execution

Features:
    - Sequential step execution with dependency resolution
    - Cache integration for step results
    - Callback hooks for monitoring and logging
    - Execution state tracking
    - Git and environment metadata capture

Examples:
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

import asyncio
import contextvars
import dataclasses
import datetime
import inspect
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from enum import Enum
from importlib.metadata import version
from logging import getLogger
from types import TracebackType
from typing import Any, NewType, Optional, TypeVar, Union, cast

from colt import Lazy, Registrable
from colt.constructed import Constructed
from typing_extensions import Self

from formed.common.attributeutils import xgetattr
from formed.common.git import GitInfo, get_git_info
from formed.common.pkgutils import PackageInfo, get_installed_packages
from formed.types import JsonValue

from .cache import EmptyWorkflowCache, WorkflowCache
from .callback import EmptyWorkflowCallback, WorkflowCallback
from .graph import WorkflowGraph, WorkflowGraphArchive
from .step import (
    WorkflowStep,
    WorkflowStepContext,
    WorkflowStepInfo,
    WorkflowStepState,
    WorkflowStepStatus,
)
from .utils import (
    BufferedAsyncIteratorState,
    BufferedSyncIteratorState,
    as_jsonvalue,
    buffer_async_iterator,
    from_jsonvalue,
)

logger = getLogger(__name__)

T = TypeVar("T")
WorkflowExecutorT = TypeVar("WorkflowExecutorT", bound="WorkflowExecutor")

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
    tags: Sequence[str] = dataclasses.field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Normalize tags to a tuple for hashability and consistency
        if not isinstance(self.tags, tuple):
            object.__setattr__(self, "tags", tuple(self.tags))


@dataclasses.dataclass
class WorkflowExecutionInfo:
    graph: WorkflowGraph

    id: Optional[WorkflowExecutionID] = None
    metadata: WorkflowExecutionMetadata = dataclasses.field(default_factory=WorkflowExecutionMetadata)

    def json(self) -> dict[str, JsonValue]:
        """Convert to JSON-serializable dict.

        Returns a dict that may contain types requiring `WorkflowJSONEncoder`
        for proper serialization (e.g., datetime, git info).

        Examples:
            >>> data = execution_info.to_json_dict()
            >>> json.dump(data, file, cls=WorkflowJSONEncoder)
        """
        return {
            "format_version": "2.0",
            "id": self.id or "",
            "graph": self.graph.to_archive().json(),
            "metadata": as_jsonvalue(self.metadata),
        }

    @classmethod
    def from_json(cls, data: dict[str, JsonValue], /) -> Self:
        """Reconstruct from JSON-serializable dict.

        Validates format version and reconstructs all fields.

        Examples:
            >>> data = json.load(file, cls=WorkflowJSONDecoder)
            >>> execution_info = WorkflowExecutionInfo.from_json_dict(data)
        """

        if data.get("format_version") != "2.0":
            raise ValueError(f"Unsupported format version '{data.get('format_version')}'. Expected '2.0'.")

        execution_id = WorkflowExecutionID(cast(str, data["id"])) if data["id"] else None
        graph = WorkflowGraph.from_archive(WorkflowGraphArchive.from_json(cast(dict[str, JsonValue], data["graph"])))
        metadata: WorkflowExecutionMetadata = from_jsonvalue(data["metadata"])

        assert isinstance(metadata, WorkflowExecutionMetadata)

        return cls(
            graph=graph,
            id=execution_id,
            metadata=metadata,
        )


@dataclasses.dataclass
class WorkflowExecutionContext:
    info: WorkflowExecutionInfo
    state: WorkflowExecutionState
    cache: WorkflowCache = dataclasses.field(default_factory=EmptyWorkflowCache)
    callback: WorkflowCallback = dataclasses.field(default_factory=EmptyWorkflowCallback)


class WorkflowExecutor(Registrable):
    def __call__(
        self,
        graph_or_execution: Union[WorkflowGraph, WorkflowExecutionInfo],
        *,
        cache: Optional[WorkflowCache] = None,
        callback: Optional[WorkflowCallback] = None,
    ) -> WorkflowExecutionContext:
        raise NotImplementedError

    def __enter__(self: WorkflowExecutorT) -> WorkflowExecutorT:
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
    """Sequential executor with transparent support for async steps.

    Steps run one at a time in dependency order. A step that returns an awaitable
    or async iterator is driven to completion via ``asyncio.run`` in its own
    fresh event loop for that step; async iterators are materialized eagerly.

    Because each async step runs on a separate short-lived event loop, async
    resources bound to one step's loop (e.g. an ``aiohttp`` session or an
    ``asyncio.Lock``) cannot be shared with another step, and no steps run
    concurrently. Use ``AsyncWorkflowExecutor`` when steps must share an event
    loop or run in parallel.
    """

    def __call__(
        self,
        graph_or_execution: Union[WorkflowGraph, WorkflowExecutionInfo],
        *,
        cache: Optional[WorkflowCache] = None,
        callback: Optional[WorkflowCallback] = None,
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

        def _resolve_awaitable(result: Any, step_name: str) -> Any:
            if not inspect.isawaitable(result):
                return result
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(cast(Any, result))
            raise RuntimeError(
                f"Step '{step_name}' returned an awaitable while an event loop is already running. "
                "Use AsyncWorkflowExecutor in async contexts."
            )

        def _materialize_async_iterator_sync(result: Any, step_name: str) -> Any:
            if not isinstance(result, AsyncIterator):
                return result
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(buffer_async_iterator(result))
            raise RuntimeError(
                f"Step '{step_name}' returned an async iterator while an event loop is already running. "
                "Use AsyncWorkflowExecutor in async contexts."
            )

        def _restore_stream(result: Any) -> Any:
            if isinstance(result, (BufferedAsyncIteratorState, BufferedSyncIteratorState)):
                return result.iterator()
            return result

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
                result = cast(T, _restore_stream(cache[step_info]))
            elif step_info in temporary_cache:
                logger.info(f"Temporary cached value found for step {step_info.name}")
                result = cast(T, _restore_stream(temporary_cache[step_info]))
            else:
                try:
                    callback.on_step_start(step_context, execution_context)
                    raw_dependencies: Mapping[Union[int, str, Sequence[Union[int, str]]], Any] = {
                        path: _run_step(dep) for path, dep in step_info.dependencies
                    }
                    if set(raw_dependencies.keys()) != set(path for path, _ in step_info.dependencies):
                        raise ValueError("Dependencies are not consistent with the graph")

                    # Wrap resolved ref values so colt does not try to re-build them as
                    # configuration objects (e.g., a dict containing a plain `type` key).
                    dependencies: Mapping[Union[int, str, Sequence[Union[int, str]]], Any] = {
                        path: Constructed(value) for path, value in raw_dependencies.items()
                    }

                    # Type narrowing: ensure we're working with a live step
                    if not isinstance(step_info.step, Lazy):
                        raise TypeError(
                            f"Cannot execute archived step '{step_info.name}'. "
                            f"Archived steps are immutable snapshots from past executions."
                        )

                    step = step_info.step.construct(dependencies)
                    result = cast(T, _resolve_awaitable(step(step_context), step_info.name))
                    result = cast(T, _materialize_async_iterator_sync(result, step_info.name))

                    if step_info.should_be_cached:
                        cache[step_info] = result
                        if isinstance(result, Iterator) or isinstance(result, BufferedAsyncIteratorState):
                            # NOTE: iterator should be restored since it is consumed by the cache
                            result = cast(T, _restore_stream(cache[step_info]))
                    elif not step_info.name.endswith("!"):
                        # NOTE: You can force to run the step without caching by adding "!" at the end of the name
                        if not isinstance(result, Iterator):
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

            if step_info.fieldref is not None:
                result = cast(T, xgetattr(result, step_info.fieldref))

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


@WorkflowExecutor.register("async")
class AsyncWorkflowExecutor(WorkflowExecutor):
    """Executor that runs independent steps concurrently on a single event loop.

    Steps whose dependencies are satisfied are scheduled concurrently, optionally
    bounded by ``max_concurrency``. Awaitable and non-awaitable steps may be mixed
    freely.

    Notes:
        - Callbacks are invoked concurrently from multiple step tasks and their
          ``on_step_start``/``on_step_end`` calls may interleave across steps.
          Callback implementations must therefore be concurrency-safe.
        - Streaming results (sync and async iterators) are fully buffered in
          memory so they can be replayed independently by every consumer.
          Steps returning unbounded/infinite iterators are not supported here;
          use ``DefaultWorkflowExecutor`` for lazy streaming.

    """

    def __init__(self, *, max_concurrency: int | None = None) -> None:
        if max_concurrency is not None and max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer")
        self._max_concurrency = max_concurrency

    def __call__(
        self,
        graph_or_execution: Union[WorkflowGraph, WorkflowExecutionInfo],
        *,
        cache: Optional[WorkflowCache] = None,
        callback: Optional[WorkflowCallback] = None,
    ) -> WorkflowExecutionContext:
        return asyncio.run(self._run_workflow(graph_or_execution, cache=cache, callback=callback))

    async def _run_workflow(
        self,
        graph_or_execution: WorkflowGraph | WorkflowExecutionInfo,
        *,
        cache: Optional[WorkflowCache] = None,
        callback: Optional[WorkflowCallback] = None,
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

        execution_state = dataclasses.replace(execution_state, execution_id=execution_info.id)
        execution_context = dataclasses.replace(execution_context, state=execution_state)

        semaphore = asyncio.Semaphore(self._max_concurrency) if self._max_concurrency is not None else None
        temporary_cache: dict[str, Any] = {}
        running_tasks: dict[str, asyncio.Task[Any]] = {}

        def _restore_stream(result: Any) -> Any:
            if isinstance(result, (BufferedAsyncIteratorState, BufferedSyncIteratorState)):
                return result.iterator()
            return result

        async def _run_step_raw(step_info: WorkflowStepInfo[WorkflowStep[Any]]) -> Any:
            assert cache is not None

            fp = step_info.fingerprint
            if step_info in cache:
                logger.info(f"Cached value found for step {step_info.name}")
                return _restore_stream(cache[step_info])
            if fp in temporary_cache:
                logger.info(f"Temporary cached value found for step {step_info.name}")
                return _restore_stream(temporary_cache[fp])

            if fp in running_tasks:
                return _restore_stream(await running_tasks[fp])

            async def _execute_step(
                dependencies: Mapping[Union[int, str, Sequence[Union[int, str]]], Any],
            ) -> Any:
                assert callback is not None

                step_state = WorkflowStepState(
                    fingerprint=step_info.fingerprint,
                    status=WorkflowStepStatus.RUNNING,
                    started_at=datetime.datetime.now(),
                )
                step_context = WorkflowStepContext(step_info, step_state)

                try:
                    callback.on_step_start(step_context, execution_context)

                    if not isinstance(step_info.step, Lazy):
                        raise TypeError(
                            f"Cannot execute archived step '{step_info.name}'. "
                            "Archived steps are immutable snapshots from past executions."
                        )

                    step = step_info.step.construct(dependencies)

                    async def _call_step() -> Any:
                        # Call the step and, while still holding the concurrency
                        # slot, fully materialize any streaming result. Buffering
                        # here (rather than after the semaphore is released) makes
                        # the real streaming work count against `max_concurrency`
                        # and produces a replayable snapshot that every consumer
                        # can iterate independently.
                        output = step(step_context)
                        if inspect.isawaitable(output):
                            output = await cast(Any, output)
                        if isinstance(output, AsyncIterator):
                            return await buffer_async_iterator(output)
                        if isinstance(output, Iterator):
                            return BufferedSyncIteratorState(output)
                        return output

                    if semaphore is None:
                        result = await _call_step()
                    else:
                        async with semaphore:
                            result = await _call_step()

                    if step_info.should_be_cached:
                        cache[step_info] = result
                    elif not step_info.name.endswith("!"):
                        temporary_cache[fp] = result
                except asyncio.CancelledError:
                    step_state = dataclasses.replace(step_state, status=WorkflowStepStatus.CANCELED)
                    step_context = dataclasses.replace(step_context, state=step_state)
                    raise
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

            async def _resolve_and_execute() -> Any:
                # Resolve dependencies before starting the actual execution.
                dependency_items = list(step_info.dependencies)
                dependency_values = await asyncio.gather(*(_run_step(dep) for _, dep in dependency_items))
                dependencies: Mapping[Union[int, str, Sequence[Union[int, str]]], Any] = cast(
                    Mapping[Union[int, str, Sequence[Union[int, str]]], Any],
                    {path: value for (path, _), value in zip(dependency_items, dependency_values)},
                )
                return await _execute_step(dependencies)

            task = asyncio.create_task(_resolve_and_execute())
            running_tasks[fp] = task
            try:
                return _restore_stream(await task)
            finally:
                current = running_tasks.get(fp)
                if current is task and task.done():
                    running_tasks.pop(fp, None)

        async def _run_step(step_info: WorkflowStepInfo[WorkflowStep[T]]) -> T:
            result = await _run_step_raw(step_info)
            if step_info.fieldref is not None:
                result = cast(T, xgetattr(result, step_info.fieldref))
            return cast(T, result)

        token = _EXECUTION_CONTEXT.set(execution_context)
        tasks = [asyncio.create_task(_run_step(step_info)) for step_info in execution_info.graph]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            first_exception: Optional[BaseException] = None
            for result in results:
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                    first_exception = result
                    break

            if first_exception is not None:
                execution_state = dataclasses.replace(execution_state, status=WorkflowExecutionStatus.FAILURE)
                execution_context = dataclasses.replace(execution_context, state=execution_state)
                raise first_exception
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
            _EXECUTION_CONTEXT.reset(token)

        return dataclasses.replace(execution_context, state=execution_state)
