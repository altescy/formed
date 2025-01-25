import logging
import os

from formed.commands import main

if os.environ.get("FORMED_DEBUG"):
    LEVEL = logging.DEBUG
else:
    level_name = os.environ.get("FORMED_LOG_LEVEL", "INFO")
    LEVEL = logging._nameToLevel.get(level_name, logging.INFO)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL)


def run() -> None:
    main(prog="formed")


if __name__ == "__main__":
    run()
