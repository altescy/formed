import asyncio
import time
from collections.abc import AsyncIterator

from formed.types import IJsonCompatible
from formed.workflow import (
    AsyncWorkflowExecutor,
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


class TestAsyncSupportInDefaultExecutor:
    def test_runs_mixed_sync_and_async_steps(self) -> None:
        @step("test_default_async::left", version="1")
        async def _() -> int:
            await asyncio.sleep(0.01)
            return 1

        @step("test_default_async::right", version="1")
        def _() -> int:
            return 2

        @step("test_default_async::sum", version="1")
        def _(left: int, right: int) -> int:
            return left + right

        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "left": {"type": "test_default_async::left"},
                    "right": {"type": "test_default_async::right"},
                    "result": {
                        "type": "test_default_async::sum",
                        "left": {"type": "ref", "ref": "left"},
                        "right": {"type": "ref", "ref": "right"},
                    },
                }
            }
        )

        context = DefaultWorkflowExecutor()(graph, cache=MemoryWorkflowCache())
        assert context.cache[context.info.graph["result"]] == 3

    def test_caches_async_iterator_results(self) -> None:
        call_count = {"source": 0}

        @step("test_default_async_iterator::source", version="1")
        async def _() -> AsyncIterator[int]:
            call_count["source"] += 1

            async def _iterator() -> AsyncIterator[int]:
                for i in range(4):
                    yield i

            return _iterator()

        @step("test_default_async_iterator::collect", version="1")
        async def _(source: AsyncIterator[int]) -> list[int]:
            return [item async for item in source]

        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "source": {"type": "test_default_async_iterator::source"},
                    "result": {
                        "type": "test_default_async_iterator::collect",
                        "source": {"type": "ref", "ref": "source"},
                    },
                }
            }
        )

        cache = MemoryWorkflowCache()
        context1 = DefaultWorkflowExecutor()(graph, cache=cache)
        context2 = DefaultWorkflowExecutor()(graph, cache=cache)

        assert context1.cache[context1.info.graph["result"]] == [0, 1, 2, 3]
        assert context2.cache[context2.info.graph["result"]] == [0, 1, 2, 3]
        assert call_count["source"] == 1


class TestAsyncWorkflowExecutor:
    def test_runs_independent_async_steps_concurrently(self) -> None:
        @step("test_async_executor::left", version="1")
        async def _() -> int:
            await asyncio.sleep(0.2)
            return 1

        @step("test_async_executor::right", version="1")
        async def _() -> int:
            await asyncio.sleep(0.2)
            return 2

        @step("test_async_executor::sum", version="1")
        async def _(left: int, right: int) -> int:
            return left + right

        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "left": {"type": "test_async_executor::left"},
                    "right": {"type": "test_async_executor::right"},
                    "result": {
                        "type": "test_async_executor::sum",
                        "left": {"type": "ref", "ref": "left"},
                        "right": {"type": "ref", "ref": "right"},
                    },
                }
            }
        )

        sequential_executor = AsyncWorkflowExecutor(max_concurrency=1)
        parallel_executor = AsyncWorkflowExecutor(max_concurrency=2)

        start = time.perf_counter()
        sequential_executor(graph, cache=MemoryWorkflowCache())
        sequential_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        context = parallel_executor(graph, cache=MemoryWorkflowCache())
        parallel_elapsed = time.perf_counter() - start

        assert context.cache[context.info.graph["result"]] == 3
        assert parallel_elapsed < sequential_elapsed

    def test_reuses_async_iterator_for_multiple_dependents(self) -> None:
        @step("test_async_executor_async_iterator::source", version="1", cacheable=False)
        async def _() -> AsyncIterator[int]:
            async def _iterator() -> AsyncIterator[int]:
                for i in range(5):
                    yield i

            return _iterator()

        @step("test_async_executor_async_iterator::sum", version="1")
        async def _(source: AsyncIterator[int]) -> int:
            total = 0
            async for item in source:
                total += item
            return total

        @step("test_async_executor_async_iterator::count", version="1")
        async def _(source: AsyncIterator[int]) -> int:
            count = 0
            async for _ in source:
                count += 1
            return count

        @step("test_async_executor_async_iterator::combine", version="1")
        async def _(total: int, count: int) -> tuple[int, int]:
            return total, count

        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "source": {"type": "test_async_executor_async_iterator::source"},
                    "total": {
                        "type": "test_async_executor_async_iterator::sum",
                        "source": {"type": "ref", "ref": "source"},
                    },
                    "count": {
                        "type": "test_async_executor_async_iterator::count",
                        "source": {"type": "ref", "ref": "source"},
                    },
                    "result": {
                        "type": "test_async_executor_async_iterator::combine",
                        "total": {"type": "ref", "ref": "total"},
                        "count": {"type": "ref", "ref": "count"},
                    },
                }
            }
        )

        context = AsyncWorkflowExecutor(max_concurrency=4)(graph, cache=MemoryWorkflowCache())
        assert context.cache[context.info.graph["result"]] == (10, 5)
