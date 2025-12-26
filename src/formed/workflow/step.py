import contextvars
import dataclasses
import datetime
import inspect
import typing
from collections.abc import Callable, Mapping
from enum import Enum
from functools import cached_property
from logging import Logger, getLogger
from os import PathLike
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

import minato
from colt import Lazy, Registrable

from formed.common.astutils import normalize_source
from formed.types import JsonValue

from .archive import WorkflowStepArchive
from .constants import WORKFLOW_WORKSPACE_DIRECTORY
from .format import AutoFormat, Format
from .types import StrictParamPath
from .utils import object_fingerprint

if TYPE_CHECKING:
    pass

T = TypeVar("T")
OutputT = TypeVar("OutputT")
StepFunctionT = TypeVar("StepFunctionT", bound=Callable[..., Any])
WorkflowStepT = TypeVar("WorkflowStepT", bound="WorkflowStep")

_STEP_CONTEXT = contextvars.ContextVar[Optional["WorkflowStepContext"]]("_STEP_CONTEXT", default=None)


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
    PENDING = "pending"
    RUNNING = "running"
    FAILURE = "failure"
    CANCELED = "canceled"
    COMPLETED = "completed"


@dataclasses.dataclass(frozen=True)
class WorkflowStepState:
    fingerprint: str
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    started_at: Optional[datetime.datetime] = None
    finished_at: Optional[datetime.datetime] = None


@dataclasses.dataclass(frozen=True)
class WorkflowStepContext:
    info: "WorkflowStepInfo"
    state: WorkflowStepState


class WorkflowStep(Generic[OutputT], Registrable):
    VERSION: ClassVar[Optional[str]] = None
    DETERMINISTIC: ClassVar[bool] = True
    CACHEABLE: ClassVar[Optional[bool]] = None
    FORMAT: Format[OutputT]
    FUNCTION: Callable[..., OutputT]

    def __init__(self, *args: Any, **kwargs: Any):
        self._args = args
        self._kwargs = kwargs

    def __call__(self, context: Optional["WorkflowStepContext"]) -> OutputT:
        cls = cast(WorkflowStep[OutputT], self.__class__)
        ctx = contextvars.copy_context()

        def run() -> OutputT:
            if context is not None:
                _STEP_CONTEXT.set(context)
            return cls.FUNCTION(*self._args, **self._kwargs)

        return ctx.run(run)

    @classmethod
    def get_output_type(cls, field: Optional[str] = None) -> type[OutputT]:
        return_annotation = cls.FUNCTION.__annotations__.get("return", Any)
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
        version: Optional[str] = None,
        deterministic: bool = True,
        cacheable: Optional[bool] = None,
        format: Optional[Union[str, Format[OutputT]]] = None,
    ) -> type["WorkflowStep[OutputT]"]:
        if isinstance(format, str):
            format = cast(type[Format[OutputT]], Format.by_name(format))()
        if version is None:
            version = object_fingerprint(normalize_source(inspect.getsource(func)))

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
        setattr(getattr(WrapperStep, "__init__"), "__annotations__", init_annotations)
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
    """Unified step info that works for both live and archived steps.

    The `step` field determines the mode:

    - `Lazy[WorkflowStepT]`: Live mode - can be constructed and executed
    - `WorkflowStepArchive`: Archived mode - immutable snapshot from past execution

    Live mode (before execution):

    - step: `Lazy[WorkflowStepT]` that can be constructed
    - dependencies: references to other live `WorkflowStepInfo` objects
    - Properties computed from `step_class`

    Archived mode (after execution):

    - step: `WorkflowStepArchive` with pre-computed metadata
    - dependencies: references to other archived `WorkflowStepInfo` objects
    - Properties returned from archive
    """

    name: str
    step: Lazy[WorkflowStepT] | WorkflowStepArchive
    dependencies: frozenset[tuple[StrictParamPath, "WorkflowStepInfo"]]

    # For WorkflowStepRef compatibility
    fieldref: str | None = None

    def is_live(self) -> bool:
        """Check if this is a live step."""
        return isinstance(self.step, Lazy)

    def is_archived(self) -> bool:
        """Check if this is an archived step."""
        return isinstance(self.step, WorkflowStepArchive)

    @cached_property
    def step_class(self) -> type[WorkflowStepT]:
        """Get the step class. Only works in live mode."""
        if not isinstance(self.step, Lazy):
            raise TypeError(
                f"Step '{self.name}' is archived. "
                f"Cannot access step_class from archived steps. "
                f"Use the archive data directly or check is_live() first."
            )
        step_class = self.step.constructor
        if not isinstance(step_class, type) or not issubclass(step_class, WorkflowStep):
            raise ValueError(f"Step {self.name} is not a subclass of WorkflowStep")
        return cast(type[WorkflowStepT], step_class)

    @cached_property
    def archive(self) -> WorkflowStepArchive:
        """Get the archive. Only works in archived mode."""
        if not isinstance(self.step, WorkflowStepArchive):
            raise TypeError(
                f"Step '{self.name}' is live. "
                f"Cannot access archive from live steps. "
                f"Call to_archive() to create an archive."
            )
        return self.step

    @cached_property
    def format(self) -> Format:
        """Get the format. Works in both modes."""
        if isinstance(self.step, WorkflowStepArchive):
            return Format.by_name(self.step.format_identifier)()
        return self.step_class.FORMAT

    @cached_property
    def version(self) -> str:
        """Get the version. Works in both modes."""
        if isinstance(self.step, WorkflowStepArchive):
            return self.step.version
        return self.step_class.VERSION or object_fingerprint(self.step_class.get_normalized_source())

    @cached_property
    def deterministic(self) -> bool:
        """Get deterministic flag. Works in both modes."""
        if isinstance(self.step, WorkflowStepArchive):
            return self.step.deterministic
        return self.step_class.DETERMINISTIC

    @cached_property
    def cacheable(self) -> bool | None:
        """Get cacheable flag. Works in both modes."""
        if isinstance(self.step, WorkflowStepArchive):
            return self.step.cacheable
        return self.step_class.CACHEABLE

    @cached_property
    def should_be_cached(self) -> bool:
        """Check if step should be cached. Works in both modes."""
        if isinstance(self.step, WorkflowStepArchive):
            return self.step.should_be_cached
        return self.cacheable or (self.cacheable is None and self.deterministic)

    @cached_property
    def fingerprint(self) -> str:
        """Get fingerprint. Works in both modes."""
        if isinstance(self.step, WorkflowStepArchive):
            return self.step.fingerprint

        # Compute from current state (live mode)
        if not isinstance(self.step, Lazy):
            raise TypeError("Step is not in live mode")

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

    def to_archive(self) -> WorkflowStepArchive:
        """Convert to archive format for serialization.

        This is called at execution time to capture the current state.
        Only works in live mode.
        """
        if isinstance(self.step, WorkflowStepArchive):
            # Already archived, return as-is
            return self.step

        if not isinstance(self.step, Lazy):
            raise TypeError("Step must be either Lazy or dict")

        # Capture source code if available
        normalized_source: str | None = None
        try:
            normalized_source = self.step_class.get_normalized_source()
        except (OSError, TypeError):
            # Built-in functions or C extensions don't have source
            pass

        # Build dependency fingerprint map with fieldrefs
        dependency_fingerprints: dict[str, dict[str, Any]] = {}
        for path, dep_info in self.dependencies:
            param_path_str = ".".join(str(p) for p in (path if isinstance(path, tuple) else (path,)))
            dependency_fingerprints[param_path_str] = {
                "fingerprint": dep_info.fingerprint,
                "fieldref": dep_info.fieldref,
            }

        # Get step type name (the registered name in Colt)
        step_type: str
        if hasattr(self.step.constructor, "__registered_name__"):
            step_type = self.step.constructor.__registered_name__  # type: ignore
        else:
            # Fallback to class name
            step_type = self.step.constructor.__name__  # type: ignore

        # Build the archive using NamedTuple constructor
        return WorkflowStepArchive(
            name=self.name,
            step_type=step_type,
            fingerprint=self.fingerprint,
            format_identifier=self.format.identifier,
            version=self.version,
            source_hash=object_fingerprint(normalized_source) if normalized_source else "",
            config=dict(self.step.config) if isinstance(self.step.config, Mapping) else {},
            deterministic=self.deterministic,
            cacheable=self.cacheable,
            should_be_cached=self.should_be_cached,
            dependency_fingerprints=dependency_fingerprints,
            fieldref=self.fieldref,
        )

    @classmethod
    def from_archive(
        cls,
        archive: WorkflowStepArchive,
        dependency_map: Mapping[str, "WorkflowStepInfo"],
    ) -> "WorkflowStepInfo":
        """Reconstruct WorkflowStepInfo from an archive.

        Args:
            archive: The archived step metadata
            dependency_map: Map from fingerprint to `WorkflowStepInfo` for all dependencies

        Returns:
            `WorkflowStepInfo` in archived mode (step field is `WorkflowStepArchive`)
        """
        # Reconstruct dependencies using the dependency_map
        dependencies: set[tuple[StrictParamPath, WorkflowStepInfo]] = set()
        for param_path_str, dep_data in archive.dependency_fingerprints.items():
            dep_fingerprint = dep_data["fingerprint"]
            assert isinstance(dep_fingerprint, str)

            dep_fieldref = dep_data["fieldref"] if "fieldref" in dep_data else None
            assert dep_fieldref is None or isinstance(dep_fieldref, str)

            if dep_fingerprint not in dependency_map:
                raise ValueError(
                    f"Dependency with fingerprint {dep_fingerprint} not found in dependency_map for step {archive.name}"
                )
            path_parts = tuple(param_path_str.split("."))
            base_dep_info = dependency_map[dep_fingerprint]

            # If the dependency has a fieldref, create a new WorkflowStepInfo with fieldref set
            if dep_fieldref:
                dep_info = WorkflowStepInfo(
                    name=base_dep_info.name,
                    step=base_dep_info.step,
                    dependencies=base_dep_info.dependencies,
                    fieldref=dep_fieldref,
                )
            else:
                dep_info = base_dep_info

            dependencies.add((path_parts, dep_info))

        return cls(
            name=archive.name,
            step=archive,  # Store the archive directly, not a Lazy
            dependencies=frozenset(dependencies),
            fieldref=archive.fieldref,
        )

    def json(self) -> dict[str, JsonValue]:
        """Convert to dict for JSON serialization (legacy compatibility)."""
        if isinstance(self.step, WorkflowStepArchive):
            config = self.step.config
        elif isinstance(self.step, Lazy):
            config = self.step.config
        else:
            config = {}

        return {
            "name": self.name,
            "version": self.version,
            "format": self.format.identifier,
            "deterministic": self.deterministic,
            "cacheable": self.cacheable,
            "fingerprint": self.fingerprint,
            "config": config,
        }

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, WorkflowStepInfo):
            return False
        return self.fingerprint == other.fingerprint

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    def __repr__(self) -> str:
        mode = "live" if self.is_live() else "archived"
        return f"WorkflowStepInfo[{self.name}:{self.fingerprint[:8]}:{mode}]"


# WorkflowStepRef has been removed and merged into WorkflowStepInfo.
# Use WorkflowStepInfo with fieldref parameter instead.


@overload
def step(
    name: str,
    *,
    version: Optional[str] = ...,
    deterministic: bool = ...,
    cacheable: Optional[bool] = ...,
    exist_ok: bool = ...,
    format: Optional[Union[str, Format]] = ...,
) -> Callable[[StepFunctionT], StepFunctionT]: ...


@overload
def step(
    name: StepFunctionT,
    *,
    version: Optional[str] = ...,
    deterministic: bool = ...,
    cacheable: Optional[bool] = ...,
    exist_ok: bool = ...,
    format: Optional[Union[str, Format]] = ...,
) -> StepFunctionT: ...


@overload
def step(
    *,
    version: Optional[str] = ...,
    deterministic: bool = ...,
    cacheable: Optional[bool] = ...,
    exist_ok: bool = ...,
    format: Optional[Union[str, Format]] = ...,
) -> Callable[[StepFunctionT], StepFunctionT]: ...


def step(
    name: Optional[Union[str, StepFunctionT]] = None,
    *,
    version: Optional[str] = None,
    deterministic: bool = True,
    cacheable: Optional[bool] = None,
    exist_ok: bool = False,
    format: Optional[Union[str, Format]] = None,
) -> Union[StepFunctionT, Callable[[StepFunctionT], StepFunctionT]]:
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


def use_step_context() -> Optional[WorkflowStepContext]:
    return _STEP_CONTEXT.get()


@overload
def use_step_logger(default: Union[str, Logger]) -> Logger: ...


@overload
def use_step_logger(default: None = ...) -> Optional[Logger]: ...


def use_step_logger(default: Optional[Union[str, Logger]] = None) -> Optional[Logger]:
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
    return getLogger(f"formed.workflow.step.{info.name}.{info.fingerprint[:8]}")


##
## Built-in Steps
##


@step("formed::load_artifact", cacheable=False)
def load_artifact(path: str | PathLike, format: Format):
    path = minato.cached_path(path)
    return format.read(path)
