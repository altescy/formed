from formed.types import IJsonCompatible
from formed.workflow import DefaultWorkflowExecutor, MemoryWorkflowCache, WorkflowExecutionInfo, WorkflowGraph, step


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


class TestWorkflowExecutionInfo:
    def test_execution_info_json_compatibility(self) -> None:
        assert issubclass(WorkflowExecutionInfo, IJsonCompatible)
