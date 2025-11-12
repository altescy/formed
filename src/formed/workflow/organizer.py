"""Workflow organizers for managing execution state and artifacts.

This module provides organizers that coordinate workflow execution,
managing caches, callbacks, and persistent storage of execution state.

Key Components:
    - WorkflowOrganizer: Abstract base class for organizers
    - MemoryWorkflowOrganizer: In-memory organizer for testing
    - FilesystemWorkflowOrganizer: Persistent filesystem-based organizer

Features:
    - Execution lifecycle management
    - Integration with caches and callbacks
    - Persistent storage of execution metadata and results
    - Thread-safe execution with file locking
    - Automatic log capture and organization

Example:
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

import json
import shutil
import uuid
from collections.abc import Sequence
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import ClassVar, TypeVar

from colt import Registrable
from filelock import FileLock

from formed.common.logutils import LogCapture

from .cache import FilesystemWorkflowCache, MemoryWorkflowCache, WorkflowCache
from .callback import MultiWorkflowCallback, WorkflowCallback
from .colt import COLT_BUILDER
from .constants import WORKFLOW_DEFAULT_DIRECTORY
from .executor import (
    WorkflowExecutionContext,
    WorkflowExecutionID,
    WorkflowExecutionInfo,
    WorkflowExecutionState,
    WorkflowExecutor,
)
from .graph import WorkflowGraph
from .step import WorkflowStepContext, WorkflowStepInfo, get_step_logger_from_info
from .utils import WorkflowJSONEncoder

logger = getLogger(__name__)

T_WorkflowOrganizer = TypeVar("T_WorkflowOrganizer", bound="WorkflowOrganizer")


class WorkflowOrganizer(Registrable):
    """Abstract base class for workflow organizers.

    Organizers coordinate workflow execution by managing caches, callbacks,
    and execution state. They provide the high-level interface for running
    workflows and accessing execution results.

    Args:
        cache: Cache for storing step results.
        callbacks: Optional callback or sequence of callbacks.

    """

    def __init__(
        self,
        cache: "WorkflowCache",
        callbacks: WorkflowCallback | Sequence[WorkflowCallback] | None,
    ) -> None:
        if isinstance(callbacks, WorkflowCallback):
            callbacks = [callbacks]

        self.cache = cache
        self.callback = MultiWorkflowCallback(callbacks or [])

    def run(
        self,
        executor: "WorkflowExecutor",
        execution: WorkflowGraph | WorkflowExecutionInfo,
    ) -> WorkflowExecutionContext:
        """Run a workflow execution.

        Args:
            executor: Executor to use for running the workflow.
            execution: Workflow graph or execution info to run.

        Returns:
            Execution context containing results and metadata.

        """
        with executor:
            return executor(
                execution,
                cache=self.cache,
                callback=self.callback,
            )

    def get(self, execution_id: WorkflowExecutionID) -> WorkflowExecutionContext | None:
        """Retrieve a previous execution by ID.

        Args:
            execution_id: Unique execution identifier.

        Returns:
            Execution context if found, None otherwise.

        """
        return None

    def exists(self, execution_id: WorkflowExecutionID) -> bool:
        """Check if an execution exists.

        Args:
            execution_id: Unique execution identifier.

        Returns:
            True if execution exists.

        """
        return self.get(execution_id) is not None

    def remove(self, execution_id: WorkflowExecutionID) -> None:
        """Remove an execution.

        Args:
            execution_id: Unique execution identifier.

        """
        pass


@WorkflowOrganizer.register("memory")
class MemoryWorkflowOrganizer(WorkflowOrganizer):
    """In-memory workflow organizer for testing and development.

    This organizer stores everything in memory and doesn't persist
    any state to disk. Useful for testing and rapid iteration.

    Args:
        callbacks: Optional callback or sequence of callbacks.

    Example:
        >>> organizer = MemoryWorkflowOrganizer()
        >>> context = organizer.run(executor, graph)

    """

    def __init__(
        self,
        callbacks: WorkflowCallback | Sequence[WorkflowCallback] | None = None,
    ) -> None:
        super().__init__(MemoryWorkflowCache(), callbacks)


@WorkflowOrganizer.register("filesystem")
class FilesystemWorkflowOrganizer(WorkflowOrganizer):
    """Filesystem-based workflow organizer with persistent storage.

    This organizer stores all execution state, logs, and artifacts to
    the filesystem. It provides persistent storage of workflow results
    and metadata, with thread-safe execution using file locks.

    Directory structure:
        {directory}/
            cache/              # Step result cache
            executions/         # Execution records
                {exec_id}/
                    out.log         # Execution log
                    execution.json  # Execution metadata
                    state.json      # Final state
                    steps/          # Step artifacts
                        {step_name}/
                            result      # Step result
                            step.json   # Step metadata
                            out.log     # Step log

    Args:
        directory: Base directory for workflow storage.
        callbacks: Optional callback or sequence of callbacks.

    Example:
        >>> organizer = FilesystemWorkflowOrganizer(".formed")
        >>> context = organizer.run(executor, graph)
        >>>
        >>> # Access previous execution
        >>> old_context = organizer.get(execution_id)

    """

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

        def __init__(self, organizer: "FilesystemWorkflowOrganizer") -> None:
            self._organizer = organizer
            self._execution_log: LogCapture | None = None
            self._step_log: dict[WorkflowStepInfo, LogCapture] = {}

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

            with FileLock(execution_directory / self._LOCK_FILENAME):
                self._execution_log = LogCapture((execution_directory / self._LOG_FILENAME).open("w"))
                self._execution_log.start()

                with open(execution_directory / self._EXECUTION_FILENAME, "w") as jsonfile:
                    json.dump(
                        execution_info,
                        jsonfile,
                        cls=WorkflowJSONEncoder,
                        indent=2,
                        ensure_ascii=False,
                    )
                with open(execution_directory / self._STATE_FILENAME, "w") as jsonfile:
                    json.dump(
                        execution_context.state,
                        jsonfile,
                        cls=WorkflowJSONEncoder,
                        indent=2,
                        ensure_ascii=False,
                    )

        def on_execution_end(
            self,
            execution_context: "WorkflowExecutionContext",
        ) -> None:
            execution_info = execution_context.info
            execution_directory = self._get_execution_directory(execution_info)

            with FileLock(execution_directory / self._LOCK_FILENAME):
                if self._execution_log is not None:
                    self._execution_log.stop()
                    self._execution_log.stream.close()
                    self._execution_log = None
                with open(execution_directory / self._STATE_FILENAME, "w") as jsonfile:
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

            with FileLock(step_directory / self._LOCK_FILENAME):
                self._step_log[step_info] = LogCapture(
                    (step_directory / self._LOG_FILENAME).open("w"),
                    logger=get_step_logger_from_info(step_info),
                )
                self._step_log[step_info].start()

                with open(step_directory / self._STEP_FILENAME, "w") as jsonfile:
                    json.dump(
                        step_info.to_dict(),
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
            with FileLock(step_directory / self._LOCK_FILENAME):
                if (step_log := self._step_log.pop(step_info, None)) is not None:
                    step_log.stop()
                    step_log.stream.close()
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
        directory: str | PathLike | None = None,
        callbacks: WorkflowCallback | Sequence[WorkflowCallback] | None = None,
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

    def get(self, execution_id: WorkflowExecutionID) -> WorkflowExecutionContext | None:
        execution_directory = self.executions_directory / execution_id
        if not execution_directory.exists():
            return None

        execution_path = execution_directory / self._Callback._EXECUTION_FILENAME
        if not execution_path.exists():
            return None

        with open(execution_path) as jsonfile:
            execution_info = COLT_BUILDER(json.load(jsonfile), WorkflowExecutionInfo)

        state_path = execution_directory / self._Callback._STATE_FILENAME
        if state_path.exists():
            with open(state_path) as jsonfile:
                execution_state = COLT_BUILDER(json.load(jsonfile), WorkflowExecutionState)
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
