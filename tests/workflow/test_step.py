from formed.types import IJsonSerializable
from formed.workflow import WorkflowGraph, step
from formed.workflow.step import WorkflowStepInfo


class TestWorkflowStepInfo:
    def test_workflow_step_info_json_serializable(self) -> None:
        assert issubclass(WorkflowStepInfo, IJsonSerializable)


class TestStepTags:
    def test_step_tags_are_exposed_on_step_info(self) -> None:
        @step("test_step_tags::tagged_step", tags=["foo", "bar"])
        def tagged_step(value: int) -> int:
            return value

        graph = WorkflowGraph.from_config({"steps": {"test": {"type": "test_step_tags::tagged_step", "value": 42}}})
        step_info = graph["test"]

        assert step_info.tags == ("bar", "foo")

    def test_step_tags_affect_fingerprint(self) -> None:
        @step("test_step_tags::tagged_a", tags=["tag"])
        def tagged_a(value: int) -> int:
            return value

        @step("test_step_tags::tagged_b", tags=["different"])
        def tagged_b(value: int) -> int:
            return value

        graph_a = WorkflowGraph.from_config({"steps": {"test": {"type": "test_step_tags::tagged_a", "value": 42}}})
        graph_b = WorkflowGraph.from_config({"steps": {"test": {"type": "test_step_tags::tagged_b", "value": 42}}})

        assert graph_a["test"].fingerprint != graph_b["test"].fingerprint

    def test_step_without_tags_has_empty_tags(self) -> None:
        @step("test_step_tags::no_tags")
        def no_tags(value: int) -> int:
            return value

        graph = WorkflowGraph.from_config({"steps": {"test": {"type": "test_step_tags::no_tags", "value": 42}}})

        assert graph["test"].tags == ()
