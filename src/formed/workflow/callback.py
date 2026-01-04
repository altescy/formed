"""Callback system for workflow execution monitoring.

This module provides a callback interface for monitoring and responding to
workflow execution events. Callbacks can be used for logging, metrics collection,
checkpointing, or custom workflow orchestration logic.

Key Components:

    - `WorkflowCallback`: Abstract base class for all callbacks
    - `EmptyWorkflowCallback`: No-op callback
    - `MultiWorkflowCallback`: Combines multiple callbacks

Features:
    - Hook points at execution and step start/end
    - Access to execution and step contexts
    - Composable callback system
    - Registrable for configuration-based instantiation

Examples:
    >>> from formed.workflow import WorkflowCallback
    >>>
    >>> @WorkflowCallback.register("custom")
    ... class CustomCallback(WorkflowCallback):
    ...     def on_step_start(self, step_context, execution_context):
    ...         print(f"Starting step: {step_context.info.name}")
    ...
    ...     def on_step_end(self, step_context, execution_context):
    ...         print(f"Finished step: {step_context.info.name}")

"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from colt import Registrable

if TYPE_CHECKING:
    from .executor import WorkflowExecutionContext
    from .step import WorkflowStepContext


class WorkflowCallback(Registrable):
    """Abstract base class for workflow execution callbacks.

    Callbacks provide hooks to execute custom logic at various points
    during workflow execution. Subclasses can override hook methods to
    implement custom monitoring, logging, or orchestration behavior.

    Hook execution order:
        1. on_execution_start - once at workflow start
        2. on_step_start - before each step execution
        3. on_step_end - after each step execution
        4. on_execution_end - once at workflow end

    Examples:
        >>> class LoggingCallback(WorkflowCallback):
        ...     def on_execution_start(self, execution_context):
        ...         print("Workflow started")
        ...
        ...     def on_step_end(self, step_context, execution_context):
        ...         print(f"Step {step_context.info.name} completed")

    """

    def on_execution_start(
        self,
        execution_context: "WorkflowExecutionContext",
    ) -> None:
        """Called once at the start of workflow execution.

        Args:
            execution_context: Context containing execution metadata and state.

        """
        pass

    def on_execution_end(
        self,
        execution_context: "WorkflowExecutionContext",
    ) -> None:
        """Called once at the end of workflow execution.

        Args:
            execution_context: Context containing execution metadata and state.

        """
        pass

    def on_step_start(
        self,
        step_context: "WorkflowStepContext",
        execution_context: "WorkflowExecutionContext",
    ) -> None:
        """Called before each step execution.

        Args:
            step_context: Context for the step about to execute.
            execution_context: Context for the overall execution.

        """
        pass

    def on_step_end(
        self,
        step_context: "WorkflowStepContext",
        execution_context: "WorkflowExecutionContext",
    ) -> None:
        """Called after each step execution.

        Args:
            step_context: Context for the step that just executed.
            execution_context: Context for the overall execution.

        """
        pass


@WorkflowCallback.register("empty")
class EmptyWorkflowCallback(WorkflowCallback):
    """No-op callback that does nothing.

    This callback can be used as a placeholder or default when no
    callback behavior is needed.

    """

    ...


@WorkflowCallback.register("multi")
class MultiWorkflowCallback(WorkflowCallback):
    """Callback that executes multiple callbacks in sequence.

    This callback allows composing multiple callbacks together, calling
    each one in order for every hook.

    Args:
        callbacks: Sequence of callbacks to execute.

    Examples:
        >>> callback1 = LoggingCallback()
        >>> callback2 = MetricsCallback()
        >>> multi = MultiWorkflowCallback([callback1, callback2])
        >>> # Both callbacks will be called for each hook

    """

    def __init__(self, callbacks: Sequence["WorkflowCallback"]) -> None:
        self._callbacks = callbacks

    def on_execution_start(
        self,
        execution_context: "WorkflowExecutionContext",
    ) -> None:
        for callback in self._callbacks:
            callback.on_execution_start(execution_context)

    def on_execution_end(
        self,
        execution_context: "WorkflowExecutionContext",
    ) -> None:
        for callback in self._callbacks:
            callback.on_execution_end(execution_context)

    def on_step_start(
        self,
        step_context: "WorkflowStepContext",
        execution_context: "WorkflowExecutionContext",
    ) -> None:
        for callback in self._callbacks:
            callback.on_step_start(step_context, execution_context)

    def on_step_end(
        self,
        step_context: "WorkflowStepContext",
        execution_context: "WorkflowExecutionContext",
    ) -> None:
        for callback in self._callbacks:
            callback.on_step_end(step_context, execution_context)
