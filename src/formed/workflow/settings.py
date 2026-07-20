import dataclasses
from collections.abc import Sequence

from .executor import DefaultWorkflowExecutor, WorkflowExecutionMetadata, WorkflowExecutor
from .organizer import FilesystemWorkflowOrganizer, WorkflowOrganizer


def _default_executor() -> WorkflowExecutor:
    return DefaultWorkflowExecutor()


def _default_organizer() -> WorkflowOrganizer:
    return FilesystemWorkflowOrganizer()


@dataclasses.dataclass(frozen=True)
class WorkflowSettings:
    executor: WorkflowExecutor = dataclasses.field(default_factory=_default_executor)
    organizer: WorkflowOrganizer = dataclasses.field(default_factory=_default_organizer)
    tags: Sequence[str] = dataclasses.field(default_factory=tuple)

    def build_execution_metadata(self) -> WorkflowExecutionMetadata:
        return WorkflowExecutionMetadata(tags=tuple(self.tags))
