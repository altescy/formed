import contextvars
import dataclasses
import datetime
import inspect
import typing
from collections.abc import Callable, Mapping
from enum import Enum
from logging import Logger, getLogger
from pathlib import Path
from typing import Annotated, Any, ClassVar, Generic, Optional, TypeVar, Union, cast, overload

from colt import Lazy, Registrable

from formed.common.astutils import normalize_source

from .constants import WORKFLOW_WORKSPACE_DIRECTORY
from .format import AutoFormat, Format
from .types import StrictParamPath
from .utils import object_fingerprint

T = TypeVar("T")
T_Output = TypeVar("T_Output")
T_StepFunction = TypeVar("T_StepFunction", bound=Callable[..., Any])
T_WorkflowStep = TypeVar("T_WorkflowStep", bound="WorkflowStep")

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


class WorkflowStep(Generic[T_Output], Registrable):
    VERSION: ClassVar[Optional[str]] = None
    DETERMINISTIC: ClassVar[bool] = True
    CACHEABLE: ClassVar[Optional[bool]] = None
    FORMAT: Format[T_Output]
    FUNCTION: Callable[..., T_Output]

    def __init__(self, *args: Any, **kwargs: Any):
        self._args = args
        self._kwargs = kwargs

    def __call__(self, context: Optional["WorkflowStepContext"]) -> T_Output:
        cls = cast(WorkflowStep[T_Output], self.__class__)
        ctx = contextvars.copy_context()

        def run() -> T_Output:
            if context is not None:
                _STEP_CONTEXT.set(context)
            return cls.FUNCTION(*self._args, **self._kwargs)

        return ctx.run(run)

    @classmethod
    def get_output_type(cls) -> type[T_Output]:
        return_annotation = cls.FUNCTION.__annotations__["return"]
        if getattr(return_annotation, "__parameters__", None):
            # This is a workaround for generic steps to skip the type checking.
            # We need to infer the output type from the configuration.
            return cast(type[T_Output], TypeVar("T"))
        return cast(type[T_Output], return_annotation)

    @classmethod
    def from_callable(
        self,
        func: Callable[..., T_Output],
        *,
        version: Optional[str] = None,
        deterministic: bool = True,
        cacheable: Optional[bool] = None,
        format: Optional[Union[str, Format[T_Output]]] = None,
    ) -> type["WorkflowStep[T_Output]"]:
        if isinstance(format, str):
            format = cast(type[Format[T_Output]], Format.by_name(format))()

        class WrapperStep(WorkflowStep[T_Output]):
            VERSION = version
            DETERMINISTIC = deterministic
            CACHEABLE = cacheable
            FUNCTION = func
            FORMAT = cast(Format[T_Output], format or AutoFormat())

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)

        annotations = typing.get_type_hints(func)
        init_annotations = {k: v for k, v in annotations.items() if k != "return"}
        setattr(WrapperStep, "__name__", func.__name__)
        setattr(getattr(WrapperStep, "__init__"), "__annotations__", init_annotations)
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
class WorkflowStepInfo(Generic[T_WorkflowStep]):
    name: str
    step: Lazy[T_WorkflowStep]
    dependencies: frozenset[tuple[StrictParamPath, "WorkflowStepInfo"]]

    @property
    def step_class(self) -> type[T_WorkflowStep]:
        step_class = self.step.constructor
        if not isinstance(step_class, type) or not issubclass(step_class, WorkflowStep):
            raise ValueError(f"Step {self.name} is not a subclass of WorkflowStep")
        return cast(type[T_WorkflowStep], step_class)

    @property
    def format(self) -> Format:
        return self.step_class.FORMAT

    @property
    def version(self) -> str:
        return self.step_class.VERSION or object_fingerprint(self.step_class.get_normalized_source())

    @property
    def deterministic(self) -> bool:
        return self.step_class.DETERMINISTIC

    @property
    def cacheable(self) -> Optional[bool]:
        return self.step_class.CACHEABLE

    @property
    def should_be_cached(self) -> bool:
        return self.cacheable or (self.cacheable is None and self.deterministic)

    @property
    def fingerprint(self) -> str:
        metadata = (self.name, self.version, self.deterministic, self.cacheable, self.format.identifier)
        config = self.step.config
        ignore_args = self.step_class.get_ignore_args()
        if isinstance(config, Mapping):
            config = {k: v for k, v in config.items() if k not in ignore_args}
        dependencies = frozenset(info.fingerprint for (key, *_), info in self.dependencies if key not in ignore_args)
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


@overload
def step(
    name: str,
    *,
    version: Optional[str] = ...,
    deterministic: bool = ...,
    cacheable: Optional[bool] = ...,
    exist_ok: bool = ...,
    format: Optional[Union[str, Format]] = ...,
) -> Callable[[T_StepFunction], T_StepFunction]: ...


@overload
def step(
    name: T_StepFunction,
    *,
    version: Optional[str] = ...,
    deterministic: bool = ...,
    cacheable: Optional[bool] = ...,
    exist_ok: bool = ...,
    format: Optional[Union[str, Format]] = ...,
) -> T_StepFunction: ...


def step(
    name: Union[str, T_StepFunction],
    *,
    version: Optional[str] = None,
    deterministic: bool = True,
    cacheable: Optional[bool] = None,
    exist_ok: bool = False,
    format: Optional[Union[str, Format]] = None,
) -> Union[T_StepFunction, Callable[[T_StepFunction], T_StepFunction]]:
    def register(name: str, func: T_StepFunction) -> None:
        step_class = WorkflowStep[Any].from_callable(
            func,
            version=version,
            deterministic=deterministic,
            cacheable=cacheable,
            format=format,
        )
        WorkflowStep.register(name, exist_ok=exist_ok)(step_class)

    def decorator(func: T_StepFunction) -> T_StepFunction:
        assert isinstance(name, str)
        register(name, func)
        return func

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


def get_step_logger_from_info(info: WorkflowStepInfo) -> Logger:
    return getLogger(f"formed.workflow.step.{info.name}.{info.fingerprint[:8]}")


def use_step_workdir() -> Path:
    context = use_step_context()
    if context is None:
        raise RuntimeError("No step context found")
    workdir = WORKFLOW_WORKSPACE_DIRECTORY / context.info.fingerprint
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir
