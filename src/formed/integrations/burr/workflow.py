import dataclasses
import datetime
from collections.abc import Iterator, Mapping, Sequence
from logging import getLogger
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Optional, Union, cast

from burr.core import ApplicationBuilder, State
from burr.core.action import SingleStepAction
from burr.core.persistence import BaseStatePersister, PersistedStateData
from colt import Lazy

from formed.common.attributeutils import xgetattr
from formed.workflow import (
    WorkflowCache,
    WorkflowCallback,
    WorkflowExecutionContext,
    WorkflowExecutionInfo,
    WorkflowExecutionState,
    WorkflowExecutionStatus,
    WorkflowExecutor,
    WorkflowGraph,
    WorkflowStepContext,
    WorkflowStepInfo,
    WorkflowStepState,
    WorkflowStepStatus,
)
from formed.workflow.cache import EmptyWorkflowCache
from formed.workflow.callback import EmptyWorkflowCallback

logger = getLogger(__name__)

_BURR_DEFAULT_DIRECTORY = ".formed/burr"


class FormedCacheStatePersister(BaseStatePersister):
    """Bridges Burr's state persistence onto formed's :class:`WorkflowCache`.

    From formed's perspective this is the ordinary fingerprint-keyed cache; from
    Burr's perspective it is a persisted state store. Each Burr action corresponds
    to a single formed step, and the step result is written into the cache when the
    action completes successfully. On resume, an action whose result is already in
    the cache is short-circuited (see :class:`_StepAction`).
    """

    def __init__(
        self,
        cache: WorkflowCache,
        step_info_by_action: Mapping[str, WorkflowStepInfo],
    ) -> None:
        self._cache = cache
        self._step_info_by_action = step_info_by_action
        self._initialized = True

    def initialize(self) -> None:
        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    def list_app_ids(self, partition_key: str, **kwargs: Any) -> list[str]:
        return []

    def load(
        self,
        partition_key: str,
        app_id: Optional[str],
        sequence_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[PersistedStateData]:
        # Per-step resume is handled inside each action via the formed cache, so
        # there is no single application-level state to reload here.
        return None

    def save(
        self,
        partition_key: Optional[str],
        app_id: str,
        sequence_id: int,
        position: str,
        state: State,
        status: Literal["completed", "failed"],
        **kwargs: Any,
    ) -> None:
        if status != "completed":
            return
        step_info = self._step_info_by_action.get(position)
        if step_info is None or not step_info.should_be_cached:
            return
        if step_info in self._cache:
            return
        key = _state_key(step_info)
        if key not in state:
            return
        result = state[key]
        if isinstance(result, Iterator):
            return
        self._cache[step_info] = result


class _StepAction(SingleStepAction):
    """A Burr action that executes a single formed step.

    Dependency results are read from the accumulated Burr :class:`State` (keyed by
    each dependency's state key) rather than being recomputed, so Burr acts as the
    true orchestrator while the declared ``reads``/``writes`` expose the real
    dependency structure to the Burr tracking UI.

    The action also feeds the Burr UI:
    - **Code** tab via :meth:`get_source` (the step's wrapped function source).
    - **Data** tab via the returned result dict (step metadata + result summary).
    - **Insights** tab via ``__tracer.log_attributes`` (parameters, timing, cache).
    """

    def __init__(
        self,
        step_info: WorkflowStepInfo,
        execution_context: WorkflowExecutionContext,
        temporary_cache: dict[WorkflowStepInfo, Any],
    ) -> None:
        super().__init__()
        self._step_info = step_info
        self._execution_context = execution_context
        self._temporary_cache = temporary_cache
        self._reads = sorted({_state_key(dep) for _, dep in step_info.dependencies})
        self._write = _state_key(step_info)

    @property
    def reads(self) -> list[str]:
        return list(self._reads)

    @property
    def writes(self) -> list[str]:
        return [self._write]

    @property
    def inputs(self) -> tuple[list[str], list[str]]:
        # Burr injects a TracerFactory for the "__tracer" input; declare it optional
        # so the action still runs if tracking is disabled.
        return ([], ["__tracer"])

    def get_source(self) -> str:
        try:
            return self._step_info.step_class.get_source()
        except Exception:
            return f"# source unavailable for step '{self._step_info.name}'"

    def run_and_update(self, state: State, **run_kwargs: Any) -> tuple[dict, State]:
        tracer = run_kwargs.get("__tracer")

        result, metadata = _execute_step(
            self._step_info,
            state,
            self._execution_context,
            self._temporary_cache,
        )

        if tracer is not None:
            execution_info = self._execution_context.info
            attributes: dict[str, Any] = dict(metadata)
            attributes["execution_id"] = str(execution_info.id) if execution_info.id is not None else None
            attributes["formed_version"] = execution_info.metadata.version
            if execution_info.metadata.git is not None:
                attributes["git"] = _jsonify(dataclasses.asdict(execution_info.metadata.git))
            try:
                tracer.log_attributes(**attributes)
            except Exception:
                logger.debug("Failed to log attributes for step %s", self._step_info.name, exc_info=True)

        tracked_result = dict(metadata)
        tracked_result["result_repr"] = _jsonify(result)

        return tracked_result, state.update(**{self._write: result})


def _execute_step(
    step_info: WorkflowStepInfo,
    state: State,
    execution_context: WorkflowExecutionContext,
    temporary_cache: dict[WorkflowStepInfo, Any],
) -> tuple[Any, dict[str, Any]]:
    """Execute a single step, honoring formed's cache/callback semantics.

    Dependency values are resolved from the accumulated Burr ``state`` (each
    dependency has already executed earlier in the topological chain). This mirrors
    :class:`DefaultWorkflowExecutor`'s per-step logic (cache lookup, temporary cache,
    the ``!`` force-run suffix, ``fieldref`` extraction and callbacks) so behavior is
    identical regardless of which executor drives the graph.

    Returns the step result together with a JSON-friendly metadata dict describing
    how the step was executed (source of the value, timing, cache status). The
    metadata is surfaced in the Burr tracking UI.
    """
    cache = execution_context.cache
    callback = execution_context.callback

    step_state = WorkflowStepState(
        fingerprint=step_info.fingerprint,
        status=WorkflowStepStatus.RUNNING,
        started_at=datetime.datetime.now(),
    )
    step_context = WorkflowStepContext(step_info, step_state)

    result: Any
    source: str
    started_at = datetime.datetime.now()

    if step_info in cache:
        logger.info(f"Cached value found for step {step_info.name}")
        result = cache[step_info]
        source = "cache"
    elif step_info in temporary_cache:
        logger.info(f"Temporary cached value found for step {step_info.name}")
        result = temporary_cache[step_info]
        source = "temporary_cache"
    else:
        source = "executed"
        try:
            callback.on_step_start(step_context, execution_context)

            dependencies: dict[Union[int, str, Sequence[Union[int, str]]], Any] = {
                path: _resolve_dependency(dep, state) for path, dep in step_info.dependencies
            }
            if set(dependencies.keys()) != set(path for path, _ in step_info.dependencies):
                raise ValueError("Dependencies are not consistent with the graph")

            if not isinstance(step_info.step, Lazy):
                raise TypeError(
                    f"Cannot execute archived step '{step_info.name}'. "
                    f"Archived steps are immutable snapshots from past executions."
                )

            step = step_info.step.construct(dependencies)
            result = step(step_context)

            if step_info.should_be_cached:
                cache[step_info] = result
                if isinstance(result, Iterator):
                    result = cache[step_info]
            elif not step_info.name.endswith("!"):
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
        result = cast(Any, xgetattr(result, step_info.fieldref))

    elapsed = (datetime.datetime.now() - started_at).total_seconds()
    metadata = _step_metadata(step_info, source=source, elapsed_seconds=elapsed)

    return result, metadata


def _resolve_dependency(dep: WorkflowStepInfo, state: State) -> Any:
    key = _state_key(dep)
    if key not in state:
        raise ValueError(f"Dependency '{dep.name}' has not been executed before its dependent step")
    value = state[key]
    if dep.fieldref is not None:
        value = xgetattr(value, dep.fieldref)
    return value


def _jsonify(value: Any) -> Any:
    """Best-effort conversion of a value into something JSON-friendly for the UI."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonify(v) for v in value]
    return repr(value)


def _step_config(step_info: WorkflowStepInfo) -> dict[str, Any]:
    """Extract the step's declared parameters (its jsonnet/config values).

    Dependency placeholders (``ref``) are omitted since those are shown as edges in
    the graph; only literal parameters are surfaced.
    """
    if not isinstance(step_info.step, Lazy):
        return {}
    config = step_info.step.config
    if not isinstance(config, Mapping):
        return {}
    dependency_paths = {path[0] for path, _ in step_info.dependencies if path}
    params: dict[str, Any] = {}
    for key, value in config.items():
        if key in ("type",) or key in dependency_paths:
            continue
        params[str(key)] = _jsonify(value)
    return params


def _step_metadata(step_info: WorkflowStepInfo, *, source: str, elapsed_seconds: float) -> dict[str, Any]:
    step_type: str = step_info.step_class.__name__
    if isinstance(step_info.step, Lazy):
        config = step_info.step.config
        if isinstance(config, Mapping) and "type" in config:
            step_type = str(config["type"])
    return {
        "step_name": step_info.name,
        "step_type": step_type,
        "fingerprint": step_info.fingerprint,
        "version": step_info.version,
        "cacheable": step_info.cacheable,
        "deterministic": step_info.deterministic,
        "source": source,
        "cache_hit": source in ("cache", "temporary_cache"),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "parameters": _step_config(step_info),
    }


@WorkflowExecutor.register("burr")
class BurrWorkflowExecutor(WorkflowExecutor):
    """A :class:`WorkflowExecutor` that orchestrates steps through Apache Burr.

    The formed dependency graph is translated into a Burr ``Application`` in which
    each step becomes a Burr action. Steps are chained in topological order for
    execution, while each action declares its real dependencies via ``reads`` so the
    dependency structure is visible in the Burr tracking UI. formed's caching
    semantics are preserved for idempotent resume through
    :class:`FormedCacheStatePersister`.
    """

    def __init__(
        self,
        *,
        tracking: bool = True,
        project: Optional[str] = None,
        directory: Union[str, PathLike] = _BURR_DEFAULT_DIRECTORY,
    ) -> None:
        self._tracking = tracking
        self._project = project
        self._directory = Path(directory)

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

        execution_state = dataclasses.replace(execution_state, execution_id=execution_info.id)
        execution_context = dataclasses.replace(execution_context, state=execution_state)

        temporary_cache: dict[WorkflowStepInfo, Any] = {}

        step_infos = list(execution_info.graph)
        action_names = {step_info: _action_name(step_info) for step_info in step_infos}
        step_info_by_action = {name: step_info for step_info, name in action_names.items()}

        try:
            self._run_application(
                step_infos,
                action_names,
                step_info_by_action,
                execution_context,
                temporary_cache,
            )
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

    def _run_application(
        self,
        step_infos: Sequence[WorkflowStepInfo],
        action_names: Mapping[WorkflowStepInfo, str],
        step_info_by_action: Mapping[str, WorkflowStepInfo],
        execution_context: WorkflowExecutionContext,
        temporary_cache: dict[WorkflowStepInfo, Any],
    ) -> None:
        if not step_infos:
            logger.info("No steps to execute for %s", execution_context.info.id)
            return

        actions = {
            action_names[step_info]: _StepAction(step_info, execution_context, temporary_cache)
            for step_info in step_infos
        }

        transitions = [
            (action_names[step_infos[i]], action_names[step_infos[i + 1]]) for i in range(len(step_infos) - 1)
        ]

        entrypoint = action_names[step_infos[0]]
        terminal = action_names[step_infos[-1]]

        initial_state = {action_names[step_info]: None for step_info in step_infos}

        builder = ApplicationBuilder().with_actions(**actions)
        if transitions:
            builder = builder.with_transitions(*transitions)
        builder = builder.with_entrypoint(entrypoint).with_state(**initial_state)

        app_id = execution_context.info.id
        if app_id is not None:
            builder = builder.with_identifiers(app_id=str(app_id))

        persister = FormedCacheStatePersister(execution_context.cache, step_info_by_action)
        builder = builder.with_state_persister(persister)

        if self._tracking:
            project = self._project or (str(app_id) if app_id is not None else "formed")
            builder = builder.with_tracker("local", project=project)

        application = builder.build()

        logger.info("Starting Burr workflow execution %s...", execution_context.info.id)
        application.run(halt_after=[terminal])

    def __enter__(self) -> "BurrWorkflowExecutor":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        pass


def _action_name(step_info: WorkflowStepInfo) -> str:
    sanitized = "".join(c if (c.isalnum() or c == "_") else "_" for c in step_info.name)
    return f"{sanitized}__{step_info.fingerprint[:8]}"


def _state_key(step_info: WorkflowStepInfo) -> str:
    return _action_name(step_info)
