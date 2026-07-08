import pytest

from formed.workflow import MemoryWorkflowCache, WorkflowGraph, step

pytest.importorskip("burr")

from formed.integrations.burr import BurrWorkflowExecutor  # noqa: E402


class TestBurrWorkflowExecutor:
    def test_executes_dag_with_fieldref(self) -> None:
        @step("test_burr::generate_data")
        def _() -> dict:
            return {"count": 1}

        @step("test_burr::increment")
        def _(count: int) -> int:
            return count + 1

        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "data": {"type": "test_burr::generate_data"},
                    "result": {
                        "type": "test_burr::increment",
                        "count": {"type": "ref", "ref": "data.count"},
                    },
                }
            }
        )

        cache = MemoryWorkflowCache()
        executor = BurrWorkflowExecutor(tracking=False)
        context = executor(graph, cache=cache)

        result = context.cache[context.info.graph["result"]]
        assert result == 2

    def test_executes_multi_dependency_dag(self) -> None:
        @step("test_burr::value")
        def _(x: int) -> int:
            return x

        @step("test_burr::add")
        def _(a: int, b: int) -> int:
            return a + b

        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "a": {"type": "test_burr::value", "x": 3},
                    "b": {"type": "test_burr::value", "x": 4},
                    "sum": {
                        "type": "test_burr::add",
                        "a": {"type": "ref", "ref": "a"},
                        "b": {"type": "ref", "ref": "b"},
                    },
                }
            }
        )

        cache = MemoryWorkflowCache()
        executor = BurrWorkflowExecutor(tracking=False)
        context = executor(graph, cache=cache)

        assert context.cache[context.info.graph["sum"]] == 7

    def test_reuses_cached_results(self) -> None:
        call_count = {"n": 0}

        @step("test_burr::counted")
        def _() -> int:
            call_count["n"] += 1
            return call_count["n"]

        graph = WorkflowGraph.from_config({"steps": {"c": {"type": "test_burr::counted"}}})

        cache = MemoryWorkflowCache()
        executor = BurrWorkflowExecutor(tracking=False)

        executor(graph, cache=cache)
        first = cache[graph["c"]]

        executor(graph, cache=cache)
        second = cache[graph["c"]]

        assert first == second == 1
        assert call_count["n"] == 1
