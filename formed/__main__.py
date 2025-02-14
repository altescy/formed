import logging
import os

from rich.logging import RichHandler

from formed.commands import main

if os.environ.get("FORMED_DEBUG"):
    LEVEL = logging.DEBUG
else:
    level_name = os.environ.get("FORMED_LOG_LEVEL", "INFO")
    LEVEL = logging._nameToLevel.get(level_name, logging.INFO)

logging.basicConfig(
    level=LEVEL,
    handlers=[RichHandler()],
)


def run() -> None:
    main(prog="formed")


if __name__ == "__main__":
    run()
