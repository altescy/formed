import asyncio
import json
import logging
from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import pytest

from formed.common.logutils import LogCapture, capture_logs, install_log_capture
from formed.workflow import (
    AsyncWorkflowExecutor,
    DefaultWorkflowExecutor,
    FilesystemWorkflowOrganizer,
    WorkflowGraph,
    step,
)


class TestLogCaptureContext:
    @pytest.fixture(autouse=True)
    def _enable_logging(self) -> Iterator[None]:
        root = logging.getLogger()
        original_level = root.level
        root.setLevel(logging.DEBUG)
        yield
        root.setLevel(original_level)

    def test_capture_logs_collects_records_from_any_logger(self) -> None:
        logger_a = logging.getLogger("test.module.a")
        logger_b = logging.getLogger("test.module.b")

        with capture_logs() as capture:
            logger_a.info("message from a")
            logger_b.warning("message from b")

        assert len(capture.records) == 2
        assert [r.getMessage() for r in capture.records] == [
            "message from a",
            "message from b",
        ]

    def test_capture_logs_is_isolated_between_contexts(self) -> None:
        logger = logging.getLogger("test.isolated")

        with capture_logs() as outer:
            logger.info("outer")
            with capture_logs() as inner:
                logger.info("inner")

        assert [r.getMessage() for r in outer.records] == ["outer"]
        assert [r.getMessage() for r in inner.records] == ["inner"]

    def test_capture_logs_propagates_through_awaits(self) -> None:
        logger = logging.getLogger("test.async")
        logging.getLogger("asyncio").setLevel(logging.WARNING)

        async def task() -> None:
            logger.info("before await")
            await asyncio.sleep(0)
            logger.info("after await")

        with capture_logs() as capture:
            asyncio.run(task())

        assert [r.getMessage() for r in capture.records] == [
            "before await",
            "after await",
        ]


class TestWorkflowLogCapture:
    @pytest.fixture(autouse=True)
    def _reset_loggers(self) -> Iterator[None]:
        # Ensure INFO-level log records reach the root handler installed by
        # install_log_capture, but avoid DEBUG noise from dependencies.
        root = logging.getLogger()
        original_level = root.level
        root.setLevel(logging.INFO)
        install_log_capture()
        yield
        root.setLevel(original_level)

    def _read_step_messages(self, step_log_path: Path) -> list[str]:
        assert step_log_path.exists()
        log_lines = [line for line in step_log_path.read_text().splitlines() if line]
        return [json.loads(line)["message"] for line in log_lines]

    def test_filesystem_organizer_captures_generic_logger_in_step(self, tmp_path: Path) -> None:
        step_logger = logging.getLogger("my.custom.step.logger")

        @step("test_log_capture::generic_logger", version="1")
        def _() -> str:
            step_logger.info("hello from generic logger")
            return "ok"

        graph = WorkflowGraph.from_config({"steps": {"result": {"type": "test_log_capture::generic_logger"}}})

        formed_dir = tmp_path / "formed"
        organizer = FilesystemWorkflowOrganizer(directory=formed_dir)
        organizer.run(DefaultWorkflowExecutor(), graph)

        executions_dir = formed_dir / "executions"
        execution_dirs = list(executions_dir.iterdir())
        assert len(execution_dirs) == 1

        step_log_path = execution_dirs[0] / "steps" / "result" / "out.log"
        messages = self._read_step_messages(step_log_path)
        assert "hello from generic logger" in messages

    def test_filesystem_organizer_captures_async_step_logs(self, tmp_path: Path) -> None:
        step_logger = logging.getLogger("my.async.step.logger")

        @step("test_log_capture::async_logger", version="1")
        async def _() -> str:
            step_logger.info("before await")
            await asyncio.sleep(0)
            step_logger.info("after await")
            return "ok"

        graph = WorkflowGraph.from_config({"steps": {"result": {"type": "test_log_capture::async_logger"}}})

        formed_dir = tmp_path / "formed"
        organizer = FilesystemWorkflowOrganizer(directory=formed_dir)
        organizer.run(DefaultWorkflowExecutor(), graph)

        executions_dir = formed_dir / "executions"
        execution_dirs = list(executions_dir.iterdir())
        step_log_path = execution_dirs[0] / "steps" / "result" / "out.log"
        messages = self._read_step_messages(step_log_path)
        assert messages == ["before await", "after await"]

    def test_filesystem_organizer_captures_async_executor_step_logs(self, tmp_path: Path) -> None:
        step_logger = logging.getLogger("my.async.executor.logger")

        @step("test_log_capture::async_executor_logger", version="1")
        async def _() -> str:
            step_logger.info("async executor log")
            return "ok"

        graph = WorkflowGraph.from_config({"steps": {"result": {"type": "test_log_capture::async_executor_logger"}}})

        formed_dir = tmp_path / "formed"
        organizer = FilesystemWorkflowOrganizer(directory=formed_dir)
        organizer.run(AsyncWorkflowExecutor(), graph)

        executions_dir = formed_dir / "executions"
        execution_dirs = list(executions_dir.iterdir())
        step_log_path = execution_dirs[0] / "steps" / "result" / "out.log"
        messages = self._read_step_messages(step_log_path)
        assert messages == ["async executor log"]

    def test_execution_log_includes_step_logs(self, tmp_path: Path) -> None:
        step_logger = logging.getLogger("test.execution.inclusion")

        @step("test_log_capture::inclusion", version="1")
        def _() -> str:
            step_logger.info("step log")
            return "ok"

        graph = WorkflowGraph.from_config({"steps": {"result": {"type": "test_log_capture::inclusion"}}})

        formed_dir = tmp_path / "formed"
        organizer = FilesystemWorkflowOrganizer(directory=formed_dir)
        organizer.run(DefaultWorkflowExecutor(), graph)

        executions_dir = formed_dir / "executions"
        execution_dirs = list(executions_dir.iterdir())
        execution_log_path = execution_dirs[0] / "out.log"
        step_log_path = execution_dirs[0] / "steps" / "result" / "out.log"

        assert execution_log_path.exists()
        step_messages = self._read_step_messages(step_log_path)
        assert step_messages == ["step log"]

        # Step logs are forwarded to the parent execution log so that the
        # execution log contains a unified view of the whole run.
        execution_messages = self._read_step_messages(execution_log_path)
        assert "step log" in execution_messages

    def test_log_capture_object_storage(self) -> None:
        capture = LogCapture()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="stored",
            args=(),
            exc_info=None,
        )
        capture.append(record)
        assert len(capture) == 1
        assert capture.messages == ["INFO    [test] stored"]

    def _make_record(self, msg: str) -> logging.LogRecord:
        return logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_sink_receives_records_incrementally_without_retaining(self) -> None:
        sink = StringIO()
        capture = LogCapture(sink=sink, retain=False)

        capture.append(self._make_record("first"))
        # Written to the sink as soon as it is appended, not only at the end.
        assert json.loads(sink.getvalue().splitlines()[0])["message"] == "first"
        # retain=False keeps memory bounded: nothing is accumulated in-memory.
        assert len(capture) == 0
        assert capture.records == []

        capture.append(self._make_record("second"))
        messages = [json.loads(line)["message"] for line in sink.getvalue().splitlines()]
        assert messages == ["first", "second"]

    def test_child_sink_forwards_to_parent_sink(self) -> None:
        parent_sink = StringIO()
        child_sink = StringIO()
        parent = LogCapture(sink=parent_sink, retain=False)
        child = LogCapture(parent=parent, sink=child_sink, retain=False)

        child.append(self._make_record("hello"))

        assert json.loads(child_sink.getvalue())["message"] == "hello"
        # Forwarded to the parent's sink too, so the execution log stays unified.
        assert json.loads(parent_sink.getvalue())["message"] == "hello"
