from formed.types import IJsonCompatible
from formed.workflow import (
    DefaultWorkflowExecutor,
    MemoryWorkflowCache,
    WorkflowExecutionID,
    WorkflowExecutionInfo,
    WorkflowGraph,
    step,
)


class TestWorkflowExecutor:
    def test_default_executor_with_fieldref(self) -> None:
        @step("test_default_executor::generate_data")
        def _() -> dict:
            return {"count": 1}

        @step("test_default_executor::increment")
        def _(count: int) -> int:
            return count + 1

        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "data": {"type": "test_default_executor::generate_data"},
                    "result": {
                        "type": "test_default_executor::increment",
                        "count": {"type": "ref", "ref": "data.count"},
                    },
                }
            }
        )

        cache = MemoryWorkflowCache()
        executor = DefaultWorkflowExecutor()
        context = executor(graph, cache=cache)
        result = context.cache[context.info.graph["result"]]
        assert result == 2

    def test_default_executor_with_ref_of_dict_having_type_key(self) -> None:
        @step("test_default_executor::generate_state")
        def _() -> dict:
            return {"learner_profile": {"type": "Learner"}}

        @step("test_default_executor::consume_state")
        def _(state: dict) -> str:
            return state["learner_profile"]["type"]

        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "state": {"type": "test_default_executor::generate_state"},
                    "result": {
                        "type": "test_default_executor::consume_state",
                        "state": {"type": "ref", "ref": "state"},
                    },
                }
            }
        )

        cache = MemoryWorkflowCache()
        executor = DefaultWorkflowExecutor()
        context = executor(graph, cache=cache)
        result = context.cache[context.info.graph["result"]]
        assert result == "Learner"


class TestWorkflowExecutionInfo:
    def test_execution_info_json_compatibility(self) -> None:
        assert issubclass(WorkflowExecutionInfo, IJsonCompatible)

    def test_execution_metadata_tags(self) -> None:
        from formed.workflow.executor import WorkflowExecutionMetadata

        metadata = WorkflowExecutionMetadata(tags=["foo", "bar"])
        assert metadata.tags == ("foo", "bar")

    def test_execution_info_preserves_tags_through_json(self) -> None:
        from formed.workflow.executor import WorkflowExecutionMetadata

        @step("test_execution_info::dummy")
        def dummy() -> int:
            return 1

        graph = WorkflowGraph.from_config({"steps": {"test": {"type": "test_execution_info::dummy"}}})
        execution = WorkflowExecutionInfo(
            graph,
            id=WorkflowExecutionID("test-id"),
            metadata=WorkflowExecutionMetadata(tags=["foo", "bar"]),
        )

        data = execution.json()
        restored = WorkflowExecutionInfo.from_json(data)

        assert restored.metadata.tags == ("foo", "bar")
