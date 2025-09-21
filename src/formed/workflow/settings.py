import dataclasses

from .executor import DefaultWorkflowExecutor, WorkflowExecutor
from .organizer import FilesystemWorkflowOrganizer, WorkflowOrganizer


def _default_executor() -> WorkflowExecutor:
    return DefaultWorkflowExecutor()


def _default_organizer() -> WorkflowOrganizer:
    return FilesystemWorkflowOrganizer()


@dataclasses.dataclass(frozen=True)
class WorkflowSettings:
    executor: WorkflowExecutor = dataclasses.field(default_factory=_default_executor)
    organizer: WorkflowOrganizer = dataclasses.field(default_factory=_default_organizer)
