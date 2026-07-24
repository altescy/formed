from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TextIO

_current_log_capture: contextvars.ContextVar["LogCapture | None"] = contextvars.ContextVar(
    "_formed_current_log_capture",
    default=None,
)

_install_lock = threading.Lock()
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

    A capture may also be given a ``sink`` (an open text stream). Records are
    then written to it as they arrive rather than only when the capture is
    drained, so logs are persisted incrementally and survive a non-graceful
    termination. Pair a ``sink`` with ``retain=False`` to stream records to the
    stream without accumulating them in memory.

    This makes it possible to capture **all** logging output produced inside a
    region (e.g. a workflow step) regardless of which logger was used.

    Only records that are actually emitted are captured: the emitting logger
    must be enabled for the record's level and must propagate to the root
    logger (``propagate=True``, the default), since the capturing handler is
    installed on the root logger.

    Examples:
        >>> import logging
        >>> from formed.common.logutils import capture_logs
        >>> logger = logging.getLogger("my.module")
        >>> logger.setLevel(logging.INFO)
        >>> with capture_logs() as capture:
        ...     logger.info("hello")
        >>> len(capture.records)
        1

    """

    parent: "LogCapture | None" = field(
        default=None,
        repr=False,
    )

    sink: TextIO | None = field(
        default=None,
        repr=False,
    )

    formatter: logging.Formatter | None = field(
        default=None,
        repr=False,
    )

    retain: bool = True

    records: list[logging.LogRecord] = field(
        default_factory=list,
        repr=False,
    )

    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )

    _formatter: logging.Formatter = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self._formatter = self.formatter or JsonFormatter()

    def append(self, record: logging.LogRecord) -> None:
        """Append a record.

        When ``retain`` is set the record is kept in :attr:`records`. When a
        ``sink`` is configured, the record is also formatted and written to it
        immediately (and flushed), so that logs are persisted incrementally
        rather than only when the capture is drained. The record is then
        forwarded to the ``parent`` capture, if any.

        Args:
            record: Log record.

        """
        with self._lock:
            if self.retain:
                self.records.append(record)
            if self.sink is not None:
                self.sink.write(self._formatter.format(record) + "\n")
                self.sink.flush()
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
            fmt = formatter or self._formatter
            return "".join(fmt.format(r) + "\n" for r in self.records)

    def __len__(self) -> int:
        with self._lock:
            return len(self.records)


class CaptureHandler(logging.Handler):
    """Routes log records tagged with an active capture to that capture."""

    def emit(self, record: logging.LogRecord) -> None:
        """Append the record to the active capture, if any."""
        capture: LogCapture | None = getattr(record, "capture", None)
        if capture is None:
            return
        try:
            capture.append(record)
        except Exception:  # pragma: no cover - mirror stdlib handlers on I/O errors
            self.handleError(record)


def install_log_record_factory() -> None:
    """Install a LogRecord factory that tags records with the active capture."""
    global _log_record_factory_installed
    if _log_record_factory_installed:
        return

    with _install_lock:
        if _log_record_factory_installed:
            return

        old_factory = logging.getLogRecordFactory()

        def factory(
            *args: object,
            **kwargs: object,
        ) -> logging.LogRecord:
            record = old_factory(*args, **kwargs)

            record.capture = _current_log_capture.get()

            return record

        logging.setLogRecordFactory(factory)
        _log_record_factory_installed = True


def install_capture_handler() -> None:
    """Install a handler on the root logger that stores tagged records."""
    global _capture_handler_installed
    if _capture_handler_installed:
        return

    with _install_lock:
        if _capture_handler_installed:
            return

        root = logging.getLogger()
        if not any(isinstance(handler, CaptureHandler) for handler in root.handlers):
            root.addHandler(CaptureHandler())

        _capture_handler_installed = True


def install_log_capture() -> None:
    """Install the log capture machinery.

    This installs a process-wide :class:`logging.LogRecord` factory and a
    :class:`CaptureHandler` on the root logger. Both are global side effects
    that affect all logging in the host process and are never uninstalled. The
    factory wraps (and preserves) whatever factory is currently installed. It
    is safe to call multiple times and from multiple threads; the installation
    is performed only once.
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
