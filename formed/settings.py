import dataclasses
import os
from logging import getLogger
from os import PathLike
from typing import ClassVar, Mapping, Optional, Sequence, TypeVar, Union

import yaml
from colt import ColtBuilder, import_modules

from formed.constants import DEFAULT_FORMED_SETTINGS_PATH
from formed.workflow import WorkflowSettings

from .constants import COLT_ARGSKEY, COLT_TYPEKEY

logger = getLogger(__name__)

T_FormedSettings = TypeVar("T_FormedSettings", bound="FormedSettings")


@dataclasses.dataclass(frozen=True)
class FormedSettings:
    __COLT_BUILDER__: ClassVar[ColtBuilder] = ColtBuilder(typekey=COLT_TYPEKEY, argskey=COLT_ARGSKEY)

    workflow: WorkflowSettings = dataclasses.field(default_factory=WorkflowSettings)

    environment: Mapping[str, str] = dataclasses.field(default_factory=dict)
    required_modules: Sequence[str] = dataclasses.field(default_factory=list)

    @classmethod
    def from_file(cls: type[T_FormedSettings], path: Union[str, PathLike]) -> T_FormedSettings:
        with open(path, "r") as f:
            settings = yaml.safe_load(f)
        # load required modules
        required_modules = cls.__COLT_BUILDER__(settings.pop("required_modules", []), Sequence[str])
        import_modules(required_modules)
        print("imported:", required_modules)
        # load environment variables
        environment = cls.__COLT_BUILDER__(settings.pop("environment", {}), Mapping[str, str])
        os.environ.update(environment)
        return cls.__COLT_BUILDER__(settings, cls)


def load_formed_settings(path: Optional[Union[str, PathLike]] = None) -> FormedSettings:
    if path is not None or DEFAULT_FORMED_SETTINGS_PATH.exists():
        path = path or DEFAULT_FORMED_SETTINGS_PATH
        logger.info(f"Load formed settings from {path}")
        return FormedSettings.from_file(path)
    return FormedSettings()
