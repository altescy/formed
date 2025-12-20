from formed.types import IJsonSerializable
from formed.workflow.step import WorkflowStepInfo


class TestWorkflowStepInfo:
    def test_workflow_step_info_json_serializable(self) -> None:
        assert issubclass(WorkflowStepInfo, IJsonSerializable)
