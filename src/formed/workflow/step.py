"""Workflow step definitions and decorators.

This module provides the core abstractions for defining workflow steps - units of
computation that can be cached, versioned, and organized into directed acyclic graphs.

Key Components:
    WorkflowStep: Base class for all workflow steps
    WorkflowStepInfo: Metadata about a step including its dependencies
    step: Decorator to convert functions into workflow steps
    WorkflowStepStatus: Execution status enumeration
    WorkflowStepContext: Runtime context available to executing steps

Features:
    - Content-based caching via fingerprinting
    - Automatic version management
    - Dependency tracking between steps
    - Custom serialization formats
    - Context managers for step-local resources

Example:
    >>> from formed import workflow
    >>>
    >>> @workflow.step
    ... def process_data(input_file: str) -> dict:
    ...     # Step function implementation
    ...     return {"result": "processed"}
    >>>
    >>> @workflow.step(version="1.0")
    ... def analyze(data: dict) -> float:
    ...     return data["result"]
    >>>
    >>> # Steps can reference each other in workflow configs
    >>> # and will be cached based on their fingerprint

"""

import contextvars
import dataclasses
import datetime
import inspect
import typing
from collections.abc import Callable, Mapping
from enum import Enum
from logging import Logger, getLogger
from pathlib import Path
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    TypeVar,
    cast,
    overload,
)

from colt import Lazy, Registrable

from formed.common.astutils import normalize_source

from .constants import WORKFLOW_WORKSPACE_DIRECTORY
from .format import AutoFormat, Format
from .types import StrictParamPath
from .utils import object_fingerprint

T = TypeVar("T")
OutputT = TypeVar("OutputT")
StepFunctionT = TypeVar("StepFunctionT", bound=Callable[..., Any])
WorkflowStepT = TypeVar("WorkflowStepT", bound="WorkflowStep")

_STEP_CONTEXT = contextvars.ContextVar["WorkflowStepContext | None"]("_STEP_CONTEXT", default=None)


class WorkflowStepArgFlag(str, Enum):
    IGNORE = "ignore"


class WorkflowStepResultFlag(str, Enum):
    METRICS = "metrics"

    @classmethod
    def get_flags(cls, step_or_annotation: Any) -> frozenset["WorkflowStepResultFlag"]:
        if isinstance(step_or_annotation, WorkflowStepInfo):
            step_or_annotation = step_or_annotation.step_class.get_output_type()
        if isinstance(step_or_annotation, WorkflowStep):
            step_or_annotation = step_or_annotation.get_output_type()
        origin = typing.get_origin(step_or_annotation)
        if origin is not Annotated:
            return frozenset()
        return frozenset(a for a in typing.get_args(step_or_annotation) if isinstance(a, WorkflowStepResultFlag))


class WorkflowStepStatus(str, Enum):
    """Execution status of a workflow step.

    Attributes:
        PENDING: Step has not started execution.
        RUNNING: Step is currently executing.
        FAILURE: Step execution failed with an error.
        CANCELED: Step execution was canceled.
        COMPLETED: Step successfully completed execution.

    """

    PENDING = "pending"
    RUNNING = "running"
    FAILURE = "failure"
    CANCELED = "canceled"
    COMPLETED = "completed"


@dataclasses.dataclass(frozen=True)
class WorkflowStepState:
    """Runtime state of a workflow step execution.

    Attributes:
        fingerprint: Content-based hash uniquely identifying this step configuration.
        status: Current execution status of the step.
        started_at: Timestamp when step execution began (None if not started).
        finished_at: Timestamp when step execution finished (None if not finished).

    """

    fingerprint: str
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None


@dataclasses.dataclass(frozen=True)
class WorkflowStepContext:
    """Context information available to executing workflow steps.

    This context is accessible via get_step_context() within a step function
    and provides metadata about the current step execution.

    Attributes:
        info: Metadata about the step (name, dependencies, etc.).
        state: Current runtime state of the step execution.

    Example:
        >>> from formed.workflow.step import get_step_context
        >>>
        >>> @workflow.step
        ... def my_step() -> str:
        ...     context = get_step_context()
        ...     print(f"Step name: {context.info.name}")
        ...     print(f"Status: {context.state.status}")
        ...     return "result"

    """

    info: "WorkflowStepInfo"
    state: WorkflowStepState


class WorkflowStep(Generic[OutputT], Registrable):
    """Base class for workflow steps representing units of computation.

    WorkflowStep wraps a function with metadata for caching, versioning, and
    dependency tracking. Steps are typically created via the @workflow.step
    decorator rather than instantiated directly.

    Type Parameters:
        OutputT: The return type of the step function.

    Class Attributes:
        VERSION: Optional version string for cache invalidation.
        DETERMINISTIC: Whether the step produces deterministic outputs.
        CACHEABLE: Whether step results should be cached (None = auto-detect).
        FORMAT: Serialization format for step outputs.
        FUNCTION: The underlying function implementing the step logic.

    Example:
        >>> # Steps are typically created via decorator
        >>> @workflow.step(version="1.0", deterministic=True)
        ... def process(data: str) -> dict:
        ...     return {"processed": data}
        >>>
        >>> # The decorator creates a WorkflowStep subclass
        >>> step_instance = process(data="input")
        >>> # Execute the step with optional context
        >>> result = step_instance(context=None)

    Note:
        - Fingerprint is computed from VERSION, source code, and arguments
        - Non-deterministic steps are always re-executed
        - CACHEABLE=False disables caching regardless of determinism

    """

    VERSION: ClassVar[str | None] = None
    DETERMINISTIC: ClassVar[bool] = True
    CACHEABLE: ClassVar[bool | None] = None
    FORMAT: Format[OutputT]
    FUNCTION: Callable[..., OutputT]

    def __init__(self, *args: Any, **kwargs: Any):
        self._args = args
        self._kwargs = kwargs

    def __call__(self, context: "WorkflowStepContext | None") -> OutputT:
        """Execute the step function with optional context.

        Args:
            context: Optional execution context providing step metadata and state.

        Returns:
            The result of executing the step function.

        Note:
            The context is set in a context variable accessible via get_step_context()
            within the step function.

        """
        cls = cast(WorkflowStep[OutputT], self.__class__)
        ctx = contextvars.copy_context()

        def run() -> OutputT:
            if context is not None:
                _STEP_CONTEXT.set(context)
            return cls.FUNCTION(*self._args, **self._kwargs)

        return ctx.run(run)

    @classmethod
    def get_output_type(cls, field: str | None = None) -> type[OutputT]:
        return_annotation = cls.FUNCTION.__annotations__["return"]
        if field is not None:
            return_annotation = typing.get_type_hints(return_annotation).get(field, Any)
        if getattr(return_annotation, "__parameters__", None):
            # This is a workaround for generic steps to skip the type checking.
            # We need to infer the output type from the configuration.
            return cast(type[OutputT], TypeVar("T"))
        return cast(type[OutputT], return_annotation)

    @classmethod
    def from_callable(
        cls,
        func: Callable[..., OutputT],
        *,
        version: str | None = None,
        deterministic: bool = True,
        cacheable: bool | None = None,
        format: str | Format[OutputT] | None = None,
    ) -> type["WorkflowStep[OutputT]"]:
        if isinstance(format, str):
            format = cast(type[Format[OutputT]], Format.by_name(format))()

        class WrapperStep(WorkflowStep):
            VERSION = version
            DETERMINISTIC = deterministic
            CACHEABLE = cacheable
            FUNCTION = func
            FORMAT = format or AutoFormat()

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)

        signature = inspect.signature(func)
        annotations = typing.get_type_hints(func)
        init_annotations = {k: v for k, v in annotations.items() if k != "return"}
        setattr(WrapperStep, "__name__", func.__name__)
        setattr(WrapperStep, "__qualname__", func.__qualname__)
        setattr(WrapperStep, "__doc__", func.__doc__)
        setattr(
            getattr(WrapperStep, "__init__"),
            "__annotations__",
            init_annotations,
        )
        setattr(
            getattr(WrapperStep, "__init__"),
            "__signature__",
            signature.replace(return_annotation=annotations.get("return", inspect.Signature.empty)),
        )
        return WrapperStep

    @classmethod
    def get_source(cls) -> str:
        return inspect.getsource(cls.FUNCTION)

    @classmethod
    def get_normalized_source(cls) -> str:
        return normalize_source(cls.get_source())

    @classmethod
    def get_ignore_args(cls) -> frozenset[str]:
        annotations = cls.FUNCTION.__annotations__
        return frozenset(k for k, v in annotations.items() if WorkflowStepArgFlag.IGNORE in typing.get_args(v))


@dataclasses.dataclass(frozen=True)
class WorkflowStepInfo(Generic[WorkflowStepT]):
    """Metadata and configuration for a workflow step.

    WorkflowStepInfo encapsulates all information needed to identify, execute,
    and cache a workflow step. It tracks dependencies, computes fingerprints,
    and provides access to step configuration.

    Type Parameters:
        WorkflowStepT: The concrete WorkflowStep subclass.

    Attributes:
        name: Unique identifier for this step in the workflow.
        step: Lazy-evaluated step instance with configuration.
        dependencies: Set of (parameter_path, step_info) tuples for dependencies.

    Properties:
        step_class: The WorkflowStep class (not instance).
        format: Serialization format for step outputs.
        version: Version string or source code fingerprint.
        deterministic: Whether the step is deterministic.
        cacheable: Whether caching is enabled for this step.
        should_be_cached: True if step should be cached (deterministic and cacheable).
        fingerprint: Content-based hash uniquely identifying this step configuration.

    Example:
        >>> from colt import Lazy
        >>>
        >>> # Step info is typically created by WorkflowGraph
        >>> step_info = WorkflowStepInfo(
        ...     name="preprocess",
        ...     step=Lazy({"input": "data.csv"}, cls=PreprocessStep),
        ...     dependencies=frozenset()
        ... )
        >>> print(step_info.fingerprint)  # Unique hash
        >>> print(step_info.should_be_cached)  # True if cacheable

    Note:
        - Fingerprint includes version, config, and dependency fingerprints
        - Two steps with identical fingerprints can share cached results
        - Dependencies use StrictParamPath to specify which parameter depends on which step

    """

    name: str
    step: Lazy[WorkflowStepT]
    dependencies: frozenset[tuple[StrictParamPath, "WorkflowStepInfo"]]

    @property
    def step_class(self) -> type[WorkflowStepT]:
        """Get the WorkflowStep class (not instance) for this step."""
        step_class = self.step.constructor
        if not isinstance(step_class, type) or not issubclass(step_class, WorkflowStep):
            raise ValueError(f"Step {self.name} is not a subclass of WorkflowStep")
        return cast(type[WorkflowStepT], step_class)

    @property
    def format(self) -> Format:
        """Get the serialization format for this step's outputs."""
        return self.step_class.FORMAT

    @property
    def version(self) -> str:
        """Get the version string or compute from source code."""
        return self.step_class.VERSION or object_fingerprint(self.step_class.get_normalized_source())

    @property
    def deterministic(self) -> bool:
        """Check if this step produces deterministic outputs."""
        return self.step_class.DETERMINISTIC

    @property
    def cacheable(self) -> bool | None:
        """Check if caching is explicitly enabled/disabled (None = auto)."""
        return self.step_class.CACHEABLE

    @property
    def should_be_cached(self) -> bool:
        """Determine if this step's results should be cached.

        Returns True if cacheable is explicitly True, or if cacheable is None
        and the step is deterministic.
        """
        return self.cacheable or (self.cacheable is None and self.deterministic)

    @property
    def fingerprint(self) -> str:
        """Compute a content-based fingerprint for this step.

        The fingerprint includes:
        - Step metadata (name, version, determinism, cacheability, format)
        - Configuration (excluding IGNORE-flagged arguments)
        - Dependency fingerprints (for transitive cache invalidation)

        Returns:
            A deterministic hash string uniquely identifying this configuration.

        """
        metadata = (
            self.name,
            self.version,
            self.deterministic,
            self.cacheable,
            self.format.identifier,
        )
        config = self.step.config
        ignore_args = self.step_class.get_ignore_args()
        if isinstance(config, Mapping):
            config = {k: v for k, v in config.items() if k not in ignore_args}
        dependencies = sorted(info.fingerprint for (key, *_), info in self.dependencies if key not in ignore_args)
        return object_fingerprint((metadata, config, dependencies))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, WorkflowStepInfo):
            return False
        return self.fingerprint == other.fingerprint

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    def __repr__(self) -> str:
        return f"WorkflowStepInfo[{self.name}:{self.fingerprint[:8]}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "format": self.format.identifier,
            "deterministic": self.deterministic,
            "cacheable": self.cacheable,
            "fingerprint": self.fingerprint,
            "config": self.step.config,
        }


@dataclasses.dataclass(frozen=True)
class WorkflowStepRef(Generic[WorkflowStepT], WorkflowStepInfo[WorkflowStepT]):
    fieldref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if self.fieldref is not None:
            data["fieldref"] = self.fieldref
        return data


@overload
def step(
    name: str,
    *,
    version: str | None = ...,
    deterministic: bool = ...,
    cacheable: bool | None = ...,
    exist_ok: bool = ...,
    format: str | Format | None = ...,
) -> Callable[[StepFunctionT], StepFunctionT]: ...


@overload
def step(
    name: StepFunctionT,
    *,
    version: str | None = ...,
    deterministic: bool = ...,
    cacheable: bool | None = ...,
    exist_ok: bool = ...,
    format: str | Format | None = ...,
) -> StepFunctionT: ...


@overload
def step(
    *,
    version: str | None = ...,
    deterministic: bool = ...,
    cacheable: bool | None = ...,
    exist_ok: bool = ...,
    format: str | Format | None = ...,
) -> Callable[[StepFunctionT], StepFunctionT]: ...


def step(
    name: str | StepFunctionT | None = None,
    *,
    version: str | None = None,
    deterministic: bool = True,
    cacheable: bool | None = None,
    exist_ok: bool = False,
    format: str | Format | None = None,
) -> StepFunctionT | Callable[[StepFunctionT], StepFunctionT]:
    """Decorator to convert a function into a workflow step.

    This is the primary API for defining workflow steps. It wraps a function
    with caching, versioning, and dependency tracking capabilities.

    Args:
        name: Optional step name. If not provided, uses function name.
              Can also be the function itself when used without parentheses.
        version: Optional version string for cache invalidation.
                 If None, computed from function source code.
        deterministic: Whether the function produces deterministic outputs.
                      Non-deterministic steps are always re-executed.
        cacheable: Explicit cache control. None means auto-detect from determinism.
                  False disables caching even for deterministic steps.
        exist_ok: If True, allow re-registering a step with the same name.
        format: Serialization format for step outputs. Can be a Format instance
               or a string identifier. If None, uses AutoFormat.

    Returns:
        The decorated function (when used with parentheses) or a decorator
        (when used without).

    Example:
        >>> from formed import workflow
        >>>
        >>> # Simple usage
        >>> @workflow.step
        ... def preprocess(data: str) -> dict:
        ...     return {"processed": data}
        >>>
        >>> # With explicit configuration
        >>> @workflow.step(version="1.0", deterministic=True, cacheable=True)
        ... def train_model(data: dict, epochs: int = 10) -> dict:
        ...     # Training logic
        ...     return {"model": "trained"}
        >>>
        >>> # Custom name
        >>> @workflow.step("custom_name")
        ... def my_function() -> str:
        ...     return "result"
        >>>
        >>> # Non-deterministic step (always re-executed)
        >>> @workflow.step(deterministic=False)
        ... def download_latest() -> dict:
        ...     # Downloads latest data from web
        ...     return {"data": "fresh"}
        >>>
        >>> # With IGNORE flag for arguments excluded from fingerprint
        >>> from typing import Annotated
        >>> from formed.workflow.step import WorkflowStepArgFlag
        >>>
        >>> @workflow.step
        ... def process(
        ...     data: str,
        ...     debug: Annotated[bool, WorkflowStepArgFlag.IGNORE] = False
        ... ) -> dict:
        ...     # debug changes don't invalidate cache
        ...     return {"result": data}

    Note:
        - Steps are registered in the WorkflowStep registry
        - Fingerprint includes version, source code, and arguments
        - Dependencies are auto-detected from argument types
        - Use Annotated[T, WorkflowStepArgFlag.IGNORE] to exclude args from fingerprint

    """

    def register(name: str, func: StepFunctionT) -> None:
        step_class = WorkflowStep[Any].from_callable(
            func,
            version=version,
            deterministic=deterministic,
            cacheable=cacheable,
            format=format,
        )
        WorkflowStep.register(name, exist_ok=exist_ok)(step_class)

    def decorator(func: StepFunctionT) -> StepFunctionT:
        nonlocal name
        name = name or func.__name__
        assert isinstance(name, str)
        register(name, func)
        return func

    if name is None:
        return decorator

    if not isinstance(name, str):
        func = name
        register(func.__name__, func)
        return func

    return decorator


def use_step_context() -> WorkflowStepContext | None:
    """Get the current step's execution context.

    This function accesses the context variable set during step execution,
    providing access to step metadata and state.

    Returns:
        The WorkflowStepContext if called within a step execution, None otherwise.

    Example:
        >>> from formed import workflow
        >>> from formed.workflow.step import use_step_context
        >>>
        >>> @workflow.step
        ... def my_step() -> str:
        ...     context = use_step_context()
        ...     if context:
        ...         print(f"Executing step: {context.info.name}")
        ...         print(f"Fingerprint: {context.state.fingerprint}")
        ...     return "result"

    Note:
        Returns None when called outside of step execution context.

    """
    return _STEP_CONTEXT.get()


@overload
def use_step_logger(default: str | Logger) -> Logger: ...


@overload
def use_step_logger(default: None = ...) -> Logger | None: ...


def use_step_logger(default: str | Logger | None = None) -> Logger | None:
    context = use_step_context()
    if context is not None:
        return get_step_logger_from_info(context.info)
    if default is None:
        return None
    if isinstance(default, str):
        return getLogger(default)
    return default


def use_step_workdir() -> Path:
    context = use_step_context()
    if context is None:
        raise RuntimeError("No step context found")
    workdir = WORKFLOW_WORKSPACE_DIRECTORY / context.info.fingerprint
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def get_step_logger_from_info(info: WorkflowStepInfo) -> Logger:
    return getLogger(f"worktop.workflow.step.{info.name}.{info.fingerprint[:8]}")
