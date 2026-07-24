"""Workflow organizers for managing execution state and artifacts.

This module provides organizers that coordinate workflow execution,
managing caches, callbacks, and persistent storage of execution state.

Key Components:
    - `WorkflowOrganizer`: Abstract base class for organizers
    - `MemoryWorkflowOrganizer`: In-memory organizer for testing
    - `FilesystemWorkflowOrganizer`: Persistent filesystem-based organizer

Features:
    - Execution lifecycle management
    - Integration with caches and callbacks
    - Persistent storage of execution metadata and results
    - Thread-safe execution with file locking
    - Automatic log capture and organization

Examples:
    >>> from formed.workflow import FilesystemWorkflowOrganizer, DefaultWorkflowExecutor
    >>>
    >>> # Create filesystem organizer
    >>> organizer = FilesystemWorkflowOrganizer(
    ...     directory=".formed",
    ...     callbacks=[my_callback]
    ... )
    >>>
    >>> # Run workflow
    >>> executor = DefaultWorkflowExecutor()
    >>> context = organizer.run(executor, workflow_graph)
    >>>
    >>> # Access results
    >>> print(context.state)
    >>> print(context.results)

"""

import contextvars
import json
import shutil
import uuid
from collections.abc import Sequence
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import ClassVar, Optional, TypeVar, Union

from colt import Registrable
from filelock import FileLock

from formed.common.logutils import (
    LogCapture,
    _current_log_capture,
    install_log_capture,
)

from .cache import FilesystemWorkflowCache, MemoryWorkflowCache, WorkflowCache
from .callback import MultiWorkflowCallback, WorkflowCallback
from .constants import WORKFLOW_DEFAULT_DIRECTORY
from .executor import (
    WorkflowExecutionContext,
    WorkflowExecutionID,
    WorkflowExecutionInfo,
    WorkflowExecutionState,
    WorkflowExecutor,
)
from .graph import WorkflowGraph
from .step import WorkflowStepContext, WorkflowStepInfo
from .utils import WorkflowJSONDecoder, WorkflowJSONEncoder

logger = getLogger(__name__)

T_WorkflowOrganizer = TypeVar("T_WorkflowOrganizer", bound="WorkflowOrganizer")


class WorkflowOrganizer(Registrable):
    def __init__(
        self,
        cache: "WorkflowCache",
        callbacks: Optional[Union[WorkflowCallback, Sequence[WorkflowCallback]]],
    ) -> None:
        if isinstance(callbacks, WorkflowCallback):
            callbacks = [callbacks]

        self.cache = cache
        self.callback = MultiWorkflowCallback(callbacks or [])

    def run(
        self,
        executor: "WorkflowExecutor",
        execution: Union[WorkflowGraph, WorkflowExecutionInfo],
    ) -> WorkflowExecutionContext:
        with executor:
            return executor(
                execution,
                cache=self.cache,
                callback=self.callback,
            )

    def get(self, execution_id: WorkflowExecutionID) -> Optional[WorkflowExecutionContext]:
        return None

    def exists(self, execution_id: WorkflowExecutionID) -> bool:
        return self.get(execution_id) is not None

    def remove(self, execution_id: WorkflowExecutionID) -> None:
        pass


@WorkflowOrganizer.register("memory")
class MemoryWorkflowOrganizer(WorkflowOrganizer):
    def __init__(
        self,
        callbacks: Optional[Union[WorkflowCallback, Sequence[WorkflowCallback]]] = None,
    ) -> None:
        super().__init__(MemoryWorkflowCache(), callbacks)


@WorkflowOrganizer.register("filesystem")
class FilesystemWorkflowOrganizer(WorkflowOrganizer):
    _CACHE_DIRNAME: ClassVar[str] = "cache"
    _EXECUTIONS_DIRNAME: ClassVar[str] = "executions"
    _DEFAULT_DIRECTORY: ClassVar[Path] = WORKFLOW_DEFAULT_DIRECTORY

    class _Callback(WorkflowCallback):
        _LOG_FILENAME: ClassVar[str] = "out.log"
        _STEPS_DIRNAME: ClassVar[str] = "steps"
        _LOCK_FILENAME: ClassVar[str] = "__lock__"
        _STEP_FILENAME: ClassVar[str] = "step.json"
        _STATE_FILENAME: ClassVar[str] = "state.json"
        _RESULT_FILENAME: ClassVar[str] = "result"
        _EXECUTION_FILENAME: ClassVar[str] = "execution.json"
        _TAGS_FILENAME: ClassVar[str] = "tags.json"

        def __init__(self, organizer: "FilesystemWorkflowOrganizer") -> None:
            self._organizer = organizer
            self._execution_log: Optional[LogCapture] = None
            self._execution_log_token: Optional[contextvars.Token] = None
            self._step_log: dict[WorkflowStepInfo, LogCapture] = {}
            self._step_log_tokens: dict[WorkflowStepInfo, contextvars.Token] = {}

        def _generate_new_execution_id(self) -> WorkflowExecutionID:
            execution_id = uuid.uuid4().hex[:8]
            while (self._organizer.executions_directory / execution_id).exists():
                execution_id = uuid.uuid4().hex[:8]
            return WorkflowExecutionID(execution_id)

        def _get_execution_directory(self, execution_info: "WorkflowExecutionInfo") -> Path:
            if execution_info.id is None:
                raise ValueError("Execution ID is not set")
            return self._organizer.executions_directory / execution_info.id

        def _get_step_directory(
            self,
            execution_info: "WorkflowExecutionInfo",
            step_info: "WorkflowStepInfo",
        ) -> Path:
            execution_directory = self._get_execution_directory(execution_info)
            return execution_directory / self._STEPS_DIRNAME / step_info.name

        def on_execution_start(
            self,
            execution_context: "WorkflowExecutionContext",
        ) -> None:
            execution_info = execution_context.info
            if execution_info.id is None:
                execution_info.id = self._generate_new_execution_id()

            execution_directory = self._get_execution_directory(execution_info)
            execution_directory.mkdir(parents=True, exist_ok=True)

            install_log_capture()
            self._execution_log = LogCapture()
            self._execution_log_token = _current_log_capture.set(self._execution_log)

            with FileLock(execution_directory / self._LOCK_FILENAME):
                # Use helper method to save execution info
                with (execution_directory / self._EXECUTION_FILENAME).open("w") as jsonfile:
                    json.dump(
                        execution_info,
                        jsonfile,
                        cls=WorkflowJSONEncoder,
                        indent=2,
                        ensure_ascii=False,
                    )

                # Use helper method to save execution state
                with (execution_directory / self._STATE_FILENAME).open("w") as jsonfile:
                    json.dump(
                        execution_context.state,
                        jsonfile,
                        cls=WorkflowJSONEncoder,
                        indent=2,
                        ensure_ascii=False,
                    )

                # Save execution tags for easy discovery/filtering
                with (execution_directory / self._TAGS_FILENAME).open("w") as jsonfile:
                    json.dump(
                        list(execution_info.metadata.tags),
                        jsonfile,
                        indent=2,
                        ensure_ascii=False,
                    )

        def on_execution_end(
            self,
            execution_context: "WorkflowExecutionContext",
        ) -> None:
            execution_info = execution_context.info
            execution_directory = self._get_execution_directory(execution_info)

            if self._execution_log_token is not None:
                _current_log_capture.reset(self._execution_log_token)
                self._execution_log_token = None

            with FileLock(execution_directory / self._LOCK_FILENAME):
                if self._execution_log is not None:
                    with (execution_directory / self._LOG_FILENAME).open("w") as logfile:
                        self._execution_log.write_to(logfile)
                    self._execution_log = None

                # Use helper method to save execution state
                with (execution_directory / self._STATE_FILENAME).open("w") as jsonfile:
                    json.dump(
                        execution_context.state,
                        jsonfile,
                        cls=WorkflowJSONEncoder,
                        indent=2,
                        ensure_ascii=False,
                    )

        def on_step_start(
            self,
            step_context: "WorkflowStepContext",
            execution_context: "WorkflowExecutionContext",
        ) -> None:
            step_info = step_context.info
            execution_info = execution_context.info

            step_directory = self._get_step_directory(execution_info, step_info)
            step_directory.mkdir(parents=True, exist_ok=True)

            install_log_capture()
            parent_capture = _current_log_capture.get()
            self._step_log[step_info] = LogCapture(parent=parent_capture)
            self._step_log_tokens[step_info] = _current_log_capture.set(self._step_log[step_info])

            with FileLock(step_directory / self._LOCK_FILENAME):
                with open(step_directory / self._STEP_FILENAME, "w") as jsonfile:
                    json.dump(
                        step_info,
                        jsonfile,
                        cls=WorkflowJSONEncoder,
                        indent=2,
                        ensure_ascii=False,
                    )
                with open(step_directory / self._STATE_FILENAME, "w") as jsonfile:
                    json.dump(
                        step_context.state,
                        jsonfile,
                        cls=WorkflowJSONEncoder,
                        indent=2,
                        ensure_ascii=False,
                    )
                with open(step_directory / self._TAGS_FILENAME, "w") as jsonfile:
                    json.dump(
                        list(step_info.tags),
                        jsonfile,
                        indent=2,
                        ensure_ascii=False,
                    )

        def on_step_end(
            self,
            step_context: "WorkflowStepContext",
            execution_context: "WorkflowExecutionContext",
        ) -> None:
            step_info = step_context.info
            execution_info = execution_context.info
            cache = execution_context.cache

            step_directory = self._get_step_directory(execution_info, step_info)
            cache_directory = self._organizer.cache_directory / step_info.fingerprint
            result_path = step_directory / self._RESULT_FILENAME

            if (token := self._step_log_tokens.pop(step_info, None)) is not None:
                _current_log_capture.reset(token)

            with FileLock(step_directory / self._LOCK_FILENAME):
                if (step_log := self._step_log.pop(step_info, None)) is not None:
                    with (step_directory / self._LOG_FILENAME).open("w") as logfile:
                        step_log.write_to(logfile)
                with open(step_directory / self._STATE_FILENAME, "w") as jsonfile:
                    json.dump(
                        step_context.state,
                        jsonfile,
                        cls=WorkflowJSONEncoder,
                        indent=2,
                        ensure_ascii=False,
                    )
                if step_info in cache and not result_path.exists():
                    result_path.symlink_to(cache_directory)

    def __init__(
        self,
        directory: Optional[Union[str, PathLike]] = None,
        callbacks: Optional[Union[WorkflowCallback, Sequence[WorkflowCallback]]] = None,
    ) -> None:
        self._directory = Path(directory or self._DEFAULT_DIRECTORY).expanduser().resolve().absolute()

        if isinstance(callbacks, WorkflowCallback):
            callbacks = [callbacks]
        callbacks = [self._Callback(self)] + list(callbacks or [])

        super().__init__(FilesystemWorkflowCache(self.cache_directory), callbacks)

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def cache_directory(self) -> Path:
        return self._directory / self._CACHE_DIRNAME

    @property
    def executions_directory(self) -> Path:
        return self._directory / self._EXECUTIONS_DIRNAME

    def get(self, execution_id: WorkflowExecutionID) -> Optional[WorkflowExecutionContext]:
        execution_directory = self.executions_directory / execution_id
        if not execution_directory.exists():
            return None

        execution_path = execution_directory / self._Callback._EXECUTION_FILENAME
        if not execution_path.exists():
            return None

        # Use helper method to load execution info
        execution_info = WorkflowExecutionInfo.from_json(json.loads(execution_path.read_text()))

        # Use helper method to load execution state
        state_path = execution_directory / self._Callback._STATE_FILENAME
        if state_path.exists():
            with state_path.open("r") as jsonfile:
                execution_state = json.load(jsonfile, cls=WorkflowJSONDecoder)
        else:
            logger.warning(f"State file not found for execution {execution_id}")
            execution_state = WorkflowExecutionState(execution_id=execution_info.id)

        return WorkflowExecutionContext(execution_info, execution_state, self.cache, self.callback)

    def exists(self, execution_id: WorkflowExecutionID) -> bool:
        execution_directory = self.executions_directory / execution_id
        return execution_directory.exists()

    def remove(self, execution_id: WorkflowExecutionID) -> None:
        execution_directory = self.executions_directory / execution_id
        if not execution_directory.exists():
            return

        shutil.rmtree(execution_directory)
