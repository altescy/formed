import json
import logging
from types import TracebackType
from typing import Generic, Literal, Optional, TextIO, Type, TypeVar, Union

T_TextIO = TypeVar("T_TextIO", bound=TextIO)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_regord = {
            "time": self.formatTime(record),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(log_regord, ensure_ascii=False)


class LogCapture(Generic[T_TextIO]):
    def __init__(
        self,
        file: T_TextIO,
        logger: Optional[Union[str, logging.Logger]] = None,
        formatter: Optional[logging.Formatter] = None,
    ) -> None:
        formatter = formatter or JsonFormatter()
        logger = logging.getLogger(logger) if isinstance(logger, str) else (logger or logging.getLogger())

        self._file = file
        self._logger = logger
        self._formatter = formatter
        self._stream_handler: Optional[logging.StreamHandler] = None

    @property
    def stream(self) -> T_TextIO:
        return self._file

    def start(self) -> None:
        if not self._stream_handler:
            self._stream_handler = logging.StreamHandler(self._file)
            self._stream_handler.setFormatter(self._formatter)
            self._logger.addHandler(self._stream_handler)

    def stop(self) -> None:
        if self._stream_handler:
            self._logger.removeHandler(self._stream_handler)
            self._stream_handler = None

    def __enter__(self) -> None:
        self.start()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Literal[False]:
        self.stop()
        return False
