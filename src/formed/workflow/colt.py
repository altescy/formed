import datetime
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any, Final, Generic, Optional, TypeVar, Union, cast

from colt import (
    ColtContext,
    ConfigurationError,
    Placeholder,
    SkipCallback,
)
from colt.builder import ColtBuilder
from colt.callback import ColtCallback, MultiCallback
from colt.types import ParamPath
from colt.utils import get_path_name, remove_optional

from formed.constants import COLT_ARGSKEY, COLT_TYPEKEY

from .constants import WORKFLOW_REFKEY, WORKFLOW_REFTYPE
from .step import WorkflowStep
from .types import StrictParamPath

_T = TypeVar("_T")


class WorkflowRef(Generic[_T], Placeholder[_T]):
    @staticmethod
    def is_ref(builder: "ColtBuilder", config: Any) -> bool:
        return (
            isinstance(config, Mapping)
            and set(config) == {builder.typekey, WORKFLOW_REFKEY}
            and config[builder.typekey] == WORKFLOW_REFTYPE
            and isinstance(config[WORKFLOW_REFKEY], str)
        )

    def __init__(
        self,
        annotation: _T,
        path: tuple[Union[int, str], ...],
        step_name: str,
        config: Any,
    ) -> None:
        super().__init__(annotation)
        self._path = path
        self._step_name = step_name
        self._config = config

    @property
    def path(self) -> tuple[Union[int, str], ...]:
        return self._path

    @property
    def step_name(self) -> str:
        return self._step_name

    @property
    def config(self) -> Any:
        return self._config


class RefCallback(ColtCallback):
    """
    Replace `ref` configs with `WorkflowRef` instances as placeholders
    """

    _STEP_NAME_TO_TYPE_KEY: Final[str] = "__formed_workflow_step__"

    def _get_step_name_to_type(self, graph_path: ParamPath, context: ColtContext) -> dict[str, type[WorkflowStep[Any]]]:
        registry: dict[ParamPath, dict[str, type[WorkflowStep[Any]]]] = context.state.setdefault(
            self._STEP_NAME_TO_TYPE_KEY, defaultdict(dict)
        )
        return registry[graph_path]

    def _register_step_types(
        self,
        builder: "ColtBuilder",
        graph_path: StrictParamPath,
        config: Mapping,
        context: ColtContext,
    ) -> None:
        steps = config.get("steps")
        step_path = graph_path + ("steps",)
        if steps is None:
            raise ConfigurationError(f"[{get_path_name(graph_path)}] Missing 'steps' key")
        if not isinstance(steps, Mapping):
            raise ConfigurationError(f"[{get_path_name(step_path)}] Expected a mapping, got {steps}")
        step_name_to_type = self._get_step_name_to_type(graph_path, context)
        for step_name, step_config in steps.items():
            if not isinstance(step_name, str):
                raise ConfigurationError(f"[{get_path_name(step_path)}] Expected a string key, got {step_name}")
            step_config_path = graph_path + ("steps", step_name)
            if not isinstance(step_config, Mapping):
                raise ConfigurationError(f"[{get_path_name(step_config_path)}] Expected a mapping")
            step_type = step_config.get(builder.typekey)
            step_type_path = step_config_path + (builder.typekey,)
            if step_type is None:
                raise ConfigurationError(f"[{get_path_name(step_config_path)}] Missing '{builder.typekey}' key")
            if not isinstance(step_type, str):
                raise ConfigurationError(f"[{get_path_name(step_type_path)}] Expected a string, got {step_type}")
            step = cast(type[WorkflowStep[Any]], WorkflowStep[Any].by_name(step_type))
            step_name_to_type[step_name] = step

    def _find_step_type(
        self, path: tuple[Union[int, str], ...], step_name: str, context: ColtContext
    ) -> type[WorkflowStep[Any]]:
        registry: dict[ParamPath, dict[str, type[WorkflowStep[Any]]]] = context.state[self._STEP_NAME_TO_TYPE_KEY]
        graph_path = path
        while graph_path:
            if graph_path in registry:
                break
            graph_path = graph_path[:-1]
        if graph_path not in registry:
            raise ConfigurationError(f"[get_path_name(path)] Step '{step_name}' not found")
        step_name_to_type = registry[graph_path]
        if step_name not in step_name_to_type:
            raise ConfigurationError(f"[get_path_name(path)] Step '{step_name}' not found")
        return step_name_to_type[step_name]

    def on_build(
        self,
        path: ParamPath,
        config: Any,
        builder: ColtBuilder,
        context: ColtContext,
        annotation: Optional[Union[type[_T], Callable[..., _T]]] = None,
    ) -> Any:
        from .graph import WorkflowGraph

        annotation = remove_optional(annotation)
        if config and isinstance(annotation, type) and issubclass(annotation, WorkflowGraph):
            if not isinstance(config, Mapping):
                raise ConfigurationError(f"[{get_path_name(path)}] Expected a mapping, got {config}")
            self._register_step_types(builder, path, config, context)
            return config

        if WorkflowRef.is_ref(builder, config):
            step_name = config[WORKFLOW_REFKEY]
            step_type = self._find_step_type(path, step_name, context)
            return WorkflowRef(step_type.get_output_type(), path, step_name, config)

        raise SkipCallback


class DatetimeCallback(ColtCallback):
    def on_build(
        self,
        path: ParamPath,
        config: Any,
        builder: ColtBuilder,
        context: ColtContext,
        annotation: Optional[Union[type[_T], Callable[..., _T]]] = None,
    ) -> Any:
        if isinstance(config, str) and annotation is datetime.datetime:
            return datetime.datetime.fromisoformat(config)
        raise SkipCallback


COLT_BUILDER: Final = ColtBuilder(
    typekey=COLT_TYPEKEY,
    argskey=COLT_ARGSKEY,
    callback=MultiCallback(DatetimeCallback(), RefCallback()),
)
