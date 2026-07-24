from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import ParamSpec, TextIO, TypeVar

T_TextIO = TypeVar("T_TextIO", bound=TextIO)
P = ParamSpec("P")
R = TypeVar("R")

_current_log_capture: contextvars.ContextVar["LogCapture | None"] = contextvars.ContextVar(
    "_formed_current_log_capture",
    default=None,
)

_log_record_factory_installed: bool = False
_capture_handler_installed: bool = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "time": self.formatTime(record),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(log_record, ensure_ascii=False)


@dataclass(slots=True)
class LogCapture:
    """Stores log records emitted inside an active capture context.

    ``LogCapture`` works together with :func:`install_log_capture` and the
    ``_formed_current_log_capture`` context variable. Any log record created
    while the context variable points to a ``LogCapture`` instance is tagged
    with that instance and appended to it by the globally installed
    ``CaptureHandler``.

    A capture may optionally have a ``parent`` capture. Records appended to a
    child are also forwarded to its parent, which makes it possible to keep a
    unified execution log while also maintaining per-step logs.

    This makes it possible to capture **all** logging output produced inside a
    region (e.g. a workflow step) regardless of which logger was used.

    Examples:
        >>> import logging
        >>> from formed.common.logutils import capture_logs
        >>> logger = logging.getLogger("my.module")
        >>> with capture_logs() as capture:
        ...     logger.info("hello")
        >>> len(capture.records)
        1

    """

    capture_id: str = field(
        default_factory=lambda: uuid.uuid4().hex[:8],
    )

    records: list[logging.LogRecord] = field(
        default_factory=list,
    )

    parent: "LogCapture | None" = field(
        default=None,
        repr=False,
    )

    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )

    def append(self, record: logging.LogRecord) -> None:
        """Append a record.

        Args:
            record: Log record.

        """
        with self._lock:
            self.records.append(record)
            parent = self.parent
        if parent is not None:
            parent.append(record)

    def clear(self) -> None:
        """Remove all captured records."""
        with self._lock:
            self.records.clear()

    @property
    def messages(self) -> list[str]:
        """Return formatted messages."""
        with self._lock:
            return [f"{r.levelname:<7} [{r.name}] {r.getMessage()}" for r in self.records]

    def format_records(
        self,
        formatter: logging.Formatter | None = None,
    ) -> str:
        """Format all captured records using ``formatter``.

        Args:
            formatter: Formatter to use. Defaults to :class:`JsonFormatter`.

        Returns:
            Concatenated formatted log lines.

        """
        with self._lock:
            fmt = formatter or JsonFormatter()
            return "".join(fmt.format(r) + "\n" for r in self.records)

    def write_to(self, file: TextIO) -> None:
        """Write formatted records to ``file``."""
        file.write(self.format_records())

    def __len__(self) -> int:
        with self._lock:
            return len(self.records)


class CaptureHandler(logging.Handler):
    """Routes log records tagged with an active capture to that capture."""

    def emit(self, record: logging.LogRecord) -> None:
        """Append the record to the active capture, if any."""
        capture: LogCapture | None = getattr(record, "capture", None)
        if capture is not None:
            capture.append(record)


def install_log_record_factory() -> None:
    """Install a LogRecord factory that tags records with the active capture."""
    global _log_record_factory_installed
    if _log_record_factory_installed:
        return

    old_factory = logging.getLogRecordFactory()

    def factory(
        *args: object,
        **kwargs: object,
    ) -> logging.LogRecord:
        record = old_factory(*args, **kwargs)

        capture = _current_log_capture.get()
        record.capture = capture
        record.capture_id = "-" if capture is None else capture.capture_id

        return record

    logging.setLogRecordFactory(factory)
    _log_record_factory_installed = True


def install_capture_handler() -> None:
    """Install a handler on the root logger that stores tagged records."""
    global _capture_handler_installed
    if _capture_handler_installed:
        return

    root = logging.getLogger()
    if not any(isinstance(handler, CaptureHandler) for handler in root.handlers):
        root.addHandler(CaptureHandler())

    _capture_handler_installed = True


def install_log_capture() -> None:
    """Install the log capture machinery.

    This installs a process-wide :class:`logging.LogRecord` factory and a
    :class:`CaptureHandler` on the root logger. It is safe to call multiple
    times; the installation is performed only once.
    """
    install_log_record_factory()
    install_capture_handler()


@contextlib.contextmanager
def capture_logs(capture: LogCapture | None = None) -> Iterator[LogCapture]:
    """Capture all log records emitted in the current context.

    Args:
        capture: Optional capture instance to use. A new instance is created
            when omitted.

    Yields:
        The active :class:`LogCapture` instance.

    """
    install_log_capture()

    if capture is None:
        capture = LogCapture()

    token = _current_log_capture.set(capture)
    try:
        yield capture
    finally:
        _current_log_capture.reset(token)


def get_current_log_capture() -> LogCapture | None:
    """Return the active :class:`LogCapture` for the current context, if any."""
    return _current_log_capture.get()


def submit_with_context(
    executor: ThreadPoolExecutor,
    fn: Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Future[R]:
    """Submit a function to ``executor`` while preserving context variables.

    Use this helper when a step spawns worker threads and you want logs emitted
    in those threads to be captured by the same :class:`LogCapture`.

    Args:
        executor: Thread pool to submit to.
        fn: Function to run.
        *args: Positional arguments for ``fn``.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        A :class:`concurrent.futures.Future` representing the result.

    """
    ctx = contextvars.copy_context()
    return executor.submit(ctx.run, fn, *args, **kwargs)
