from collections.abc import Sequence
from typing import TYPE_CHECKING

from colt import Registrable

if TYPE_CHECKING:
    from .executor import WorkflowExecutionContext
    from .step import WorkflowStepContext


class WorkflowCallback(Registrable):
    def on_execution_start(
        self,
        execution_context: "WorkflowExecutionContext",
    ) -> None:
        pass

    def on_execution_end(
        self,
        execution_context: "WorkflowExecutionContext",
    ) -> None:
        pass

    def on_step_start(
        self,
        step_context: "WorkflowStepContext",
        execution_context: "WorkflowExecutionContext",
    ) -> None:
        pass

    def on_step_end(
        self,
        step_context: "WorkflowStepContext",
        execution_context: "WorkflowExecutionContext",
    ) -> None:
        pass


@WorkflowCallback.register("empty")
class EmptyWorkflowCallback(WorkflowCallback): ...


@WorkflowCallback.register("multi")
class MultiWorkflowCallback(WorkflowCallback):
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
