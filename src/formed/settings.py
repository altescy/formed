import dataclasses
import logging
import os
from os import PathLike
from typing import ClassVar, Literal, Mapping, Optional, Sequence, TypeVar, Union

import yaml
from colt import import_modules
from colt.builder import ColtBuilder

from formed.constants import DEFAULT_FORMED_SETTINGS_PATH
from formed.workflow import WorkflowSettings

from .constants import COLT_ARGSKEY, COLT_TYPEKEY

logger = logging.getLogger(__name__)

T_FormedSettings = TypeVar("T_FormedSettings", bound="FormedSettings")


@dataclasses.dataclass(frozen=True)
class LoggingSettings:
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    handlers: Sequence[logging.Handler] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class FormedSettings:
    __COLT_BUILDER__: ClassVar[ColtBuilder] = ColtBuilder(typekey=COLT_TYPEKEY, argskey=COLT_ARGSKEY)

    workflow: WorkflowSettings = dataclasses.field(default_factory=WorkflowSettings)

    environment: Mapping[str, str] = dataclasses.field(default_factory=dict)
    required_modules: Sequence[str] = dataclasses.field(default_factory=list)
    logging: Mapping[str, LoggingSettings] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_file(cls: type[T_FormedSettings], path: Union[str, PathLike]) -> T_FormedSettings:
        logger = logging.getLogger(__name__)

        with open(path, "r") as f:
            settings = yaml.safe_load(f)

        # load required modules
        required_modules = cls.__COLT_BUILDER__(settings.get("required_modules", []), Sequence[str])
        import_modules(required_modules)
        logger.info(f"Load required modules: {required_modules}")

        formed_settings = cls.__COLT_BUILDER__(settings, cls)

        # load environment variables
        environment = formed_settings.environment
        os.environ.update(environment)

        # Setup loggers
        for logger_name, logger_settings in formed_settings.logging.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging._nameToLevel[logger_settings.level])
            for handler in logger_settings.handlers:
                logger.addHandler(handler)

        return formed_settings


def load_formed_settings(path: Optional[Union[str, PathLike]] = None) -> FormedSettings:
    if path is not None or DEFAULT_FORMED_SETTINGS_PATH.exists():
        path = path or DEFAULT_FORMED_SETTINGS_PATH
        logger.info(f"Load formed settings from {path}")
        return FormedSettings.from_file(path)
    return FormedSettings()
