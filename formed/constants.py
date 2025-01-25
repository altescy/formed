from pathlib import Path
from typing import Final

COLT_TYPEKEY: Final = "type"
COLT_ARGSKEY: Final = "*"

DEFAULT_WORKING_DIRECTORY: Final = Path.cwd()
DEFAULT_FORMED_DIRECTORY: Final = DEFAULT_WORKING_DIRECTORY / ".formed"
DEFAULT_FORMED_SETTINGS_PATH: Final = DEFAULT_WORKING_DIRECTORY / "formed.yml"
