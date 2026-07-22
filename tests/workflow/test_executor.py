import asyncio
import time
from collections.abc import AsyncIterator, Iterator

import pytest

from formed.types import IJsonCompatible
from formed.workflow import (
    AsyncWorkflowExecutor,
    DefaultWorkflowExecutor,
    MemoryWorkflowCache,
    WorkflowCallback,
    WorkflowExecutionID,
    WorkflowExecutionInfo,
    WorkflowGraph,
    step,
    use_step_context,
)


class TestWorkflowExecutor:
    def test_callback_receives_non_cached_step_result(self) -> None:
        class ResultCallback(WorkflowCallback):
            result = None

            def on_step_end(self, step_context, execution_context):
                self.result = step_context.result

        @step("test_default_executor::non_cached_result", cacheable=False)
        def _() -> dict[str, float]:
            return {"loss": 0.5}

        graph = WorkflowGraph.from_config({"steps": {"metrics": {"type": "test_default_executor::non_cached_result"}}})
        callback = ResultCallback()

        DefaultWorkflowExecutor()(graph, callback=callback)

        assert callback.result == {"loss": 0.5}

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
    def test_async_step_keeps_step_context_after_await(self) -> None:
        @step("test_default_async_context::step", version="1")
        async def _() -> str:
            await asyncio.sleep(0)
            context = use_step_context()
            assert context is not None
            return context.info.name

        graph = WorkflowGraph.from_config({"steps": {"result": {"type": "test_default_async_context::step"}}})

        context = DefaultWorkflowExecutor()(graph, cache=MemoryWorkflowCache())
        assert context.cache[context.info.graph["result"]] == "result"

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
    def test_async_executor_with_ref_of_dict_having_type_key(self) -> None:
        @step("test_async_executor::generate_state")
        async def _() -> dict:
            return {"learner_profile": {"type": "Learner"}}

        @step("test_async_executor::consume_state")
        async def _(state: dict) -> str:
            return state["learner_profile"]["type"]

        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "state": {"type": "test_async_executor::generate_state"},
                    "result": {
                        "type": "test_async_executor::consume_state",
                        "state": {"type": "ref", "ref": "state"},
                    },
                }
            }
        )

        cache = MemoryWorkflowCache()
        context = AsyncWorkflowExecutor()(graph, cache=cache)
        result = context.cache[context.info.graph["result"]]
        assert result == "Learner"

    def test_callback_receives_forced_non_cached_step_result(self) -> None:
        class ResultCallback(WorkflowCallback):
            result = None

            def on_step_end(self, step_context, execution_context):
                self.result = step_context.result

        @step("test_async_executor::forced_non_cached_result")
        async def _() -> dict[str, float]:
            return {"loss": 0.25}

        graph = WorkflowGraph.from_config(
            {"steps": {"metrics!": {"type": "test_async_executor::forced_non_cached_result"}}}
        )
        callback = ResultCallback()

        AsyncWorkflowExecutor()(graph, callback=callback)

        assert callback.result == {"loss": 0.25}

    def test_cancellation_finishes_execution_as_canceled(self) -> None:
        from formed.workflow.callback import WorkflowCallback
        from formed.workflow.executor import WorkflowExecutionContext, WorkflowExecutionStatus

        started = asyncio.Event()

        @step("test_async_executor_cancellation::step", version="1")
        async def _() -> None:
            started.set()
            await asyncio.Event().wait()

        class RecordingCallback(WorkflowCallback):
            ended_context: WorkflowExecutionContext | None = None

            def on_execution_end(self, execution_context: WorkflowExecutionContext) -> None:
                self.ended_context = execution_context

        graph = WorkflowGraph.from_config({"steps": {"running": {"type": "test_async_executor_cancellation::step"}}})
        callback = RecordingCallback()
        executor = AsyncWorkflowExecutor()

        async def cancel_execution() -> None:
            task = asyncio.create_task(executor._run_workflow(graph, callback=callback))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_execution())

        assert callback.ended_context is not None
        assert callback.ended_context.state.status is WorkflowExecutionStatus.CANCELED
        assert callback.ended_context.state.finished_at is not None

    def test_async_step_keeps_step_context_after_await(self) -> None:
        @step("test_async_executor_context::step", version="1")
        async def _() -> str:
            await asyncio.sleep(0)
            context = use_step_context()
            assert context is not None
            return context.info.name

        graph = WorkflowGraph.from_config({"steps": {"result": {"type": "test_async_executor_context::step"}}})

        context = AsyncWorkflowExecutor()(graph, cache=MemoryWorkflowCache())
        assert context.cache[context.info.graph["result"]] == "result"

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

    def test_dependent_step_starts_after_dependencies_resolve(self) -> None:
        from formed.workflow.callback import WorkflowCallback

        class TimingCallback(WorkflowCallback):
            def __init__(self) -> None:
                self.events: list[tuple[str, str, float]] = []

            def on_step_start(self, step_context, execution_context):
                self.events.append((step_context.info.name, "start", time.perf_counter()))

            def on_step_end(self, step_context, execution_context):
                self.events.append((step_context.info.name, "end", time.perf_counter()))

        @step("test_async_executor_dependency_order::source", version="1")
        async def _() -> int:
            await asyncio.sleep(0.2)
            return 1

        @step("test_async_executor_dependency_order::dependent", version="1")
        async def _(source: int) -> int:
            await asyncio.sleep(0.1)
            return source + 1

        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "source": {"type": "test_async_executor_dependency_order::source"},
                    "result": {
                        "type": "test_async_executor_dependency_order::dependent",
                        "source": {"type": "ref", "ref": "source"},
                    },
                }
            }
        )

        callback = TimingCallback()
        executor = AsyncWorkflowExecutor()
        executor(graph, cache=MemoryWorkflowCache(), callback=callback)

        source_end = next(t for name, event, t in callback.events if name == "source" and event == "end")
        dependent_start = next(t for name, event, t in callback.events if name == "result" and event == "start")
        assert dependent_start > source_end

    def test_running_steps_get_on_step_end_on_failure(self) -> None:
        from formed.workflow.callback import WorkflowCallback
        from formed.workflow.step import WorkflowStepStatus

        class StatusCallback(WorkflowCallback):
            def __init__(self) -> None:
                self.events: list[tuple[str, str, WorkflowStepStatus]] = []

            def on_step_start(self, step_context, execution_context):
                self.events.append((step_context.info.name, "start", step_context.state.status))

            def on_step_end(self, step_context, execution_context):
                self.events.append((step_context.info.name, "end", step_context.state.status))

        @step("test_async_executor_failure::a", version="1")
        async def _() -> int:
            await asyncio.sleep(0.3)
            return 1

        @step("test_async_executor_failure::b", version="1")
        async def _() -> int:
            await asyncio.sleep(0.1)
            raise RuntimeError("b failed")

        @step("test_async_executor_failure::c", version="1")
        async def _() -> int:
            await asyncio.sleep(0.3)
            return 3

        @step("test_async_executor_failure::d", version="1")
        async def _(a: int, b: int, c: int) -> int:
            return a + b + c

        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "a": {"type": "test_async_executor_failure::a"},
                    "b": {"type": "test_async_executor_failure::b"},
                    "c": {"type": "test_async_executor_failure::c"},
                    "d": {
                        "type": "test_async_executor_failure::d",
                        "a": {"type": "ref", "ref": "a"},
                        "b": {"type": "ref", "ref": "b"},
                        "c": {"type": "ref", "ref": "c"},
                    },
                }
            }
        )

        callback = StatusCallback()
        executor = AsyncWorkflowExecutor()
        with pytest.raises(Exception):
            executor(graph, cache=MemoryWorkflowCache(), callback=callback)

        end_statuses = {name: status for name, event, status in callback.events if event == "end"}
        assert "b" in end_statuses and end_statuses["b"] == WorkflowStepStatus.FAILURE
        assert "a" in end_statuses, "independent step 'a' should receive on_step_end"
        assert "c" in end_statuses, "independent step 'c' should receive on_step_end"
        assert end_statuses["a"] == WorkflowStepStatus.COMPLETED
        assert end_statuses["c"] == WorkflowStepStatus.COMPLETED
        assert "d" not in end_statuses, "dependent step 'd' should not start and therefore not receive on_step_end"

    def test_max_concurrency_bounds_async_iterator_work(self) -> None:
        # The real work of a streaming step happens while its async iterator is
        # consumed. That consumption must be bounded by max_concurrency, so two
        # streaming steps must not iterate at the same time when concurrency is 1.
        state = {"active": 0, "max_active": 0}

        def _make_source(name: str) -> None:
            @step(name, version="1", cacheable=False)
            async def _() -> AsyncIterator[int]:
                async def _iterator() -> AsyncIterator[int]:
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                    try:
                        for i in range(3):
                            await asyncio.sleep(0.02)
                            yield i
                    finally:
                        state["active"] -= 1

                return _iterator()

        _make_source("test_async_executor_concurrency::a")
        _make_source("test_async_executor_concurrency::b")

        @step("test_async_executor_concurrency::combine", version="1")
        async def _(a: AsyncIterator[int], b: AsyncIterator[int]) -> int:
            total = 0
            async for item in a:
                total += item
            async for item in b:
                total += item
            return total

        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "a": {"type": "test_async_executor_concurrency::a"},
                    "b": {"type": "test_async_executor_concurrency::b"},
                    "result": {
                        "type": "test_async_executor_concurrency::combine",
                        "a": {"type": "ref", "ref": "a"},
                        "b": {"type": "ref", "ref": "b"},
                    },
                }
            }
        )

        context = AsyncWorkflowExecutor(max_concurrency=1)(graph, cache=MemoryWorkflowCache())

        assert context.cache[context.info.graph["result"]] == 6
        assert state["max_active"] == 1

    def test_shared_async_iterator_delivers_full_data_to_concurrent_dependents(self) -> None:
        # The source is slow enough that both dependents resolve it while it is
        # still in flight (the `running_tasks` path). Each dependent must receive
        # an independent iterator over the full stream.
        @step("test_async_executor_shared_async::source", version="1")
        async def _() -> AsyncIterator[int]:
            await asyncio.sleep(0.1)

            async def _iterator() -> AsyncIterator[int]:
                for i in range(5):
                    yield i

            return _iterator()

        @step("test_async_executor_shared_async::sum", version="1")
        async def _(source: AsyncIterator[int]) -> int:
            return sum([item async for item in source])

        @step("test_async_executor_shared_async::count", version="1")
        async def _(source: AsyncIterator[int]) -> int:
            return len([item async for item in source])

        @step("test_async_executor_shared_async::combine", version="1")
        async def _(total: int, cnt: int) -> tuple[int, int]:
            return total, cnt

        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "source": {"type": "test_async_executor_shared_async::source"},
                    "total": {
                        "type": "test_async_executor_shared_async::sum",
                        "source": {"type": "ref", "ref": "source"},
                    },
                    "count": {
                        "type": "test_async_executor_shared_async::count",
                        "source": {"type": "ref", "ref": "source"},
                    },
                    "result": {
                        "type": "test_async_executor_shared_async::combine",
                        "total": {"type": "ref", "ref": "total"},
                        "cnt": {"type": "ref", "ref": "count"},
                    },
                }
            }
        )

        context = AsyncWorkflowExecutor(max_concurrency=4)(graph, cache=MemoryWorkflowCache())
        assert context.cache[context.info.graph["result"]] == (10, 5)

    def test_shared_sync_iterator_delivers_full_data_to_concurrent_dependents(self) -> None:
        # A non-cached step returning a plain (sync) generator, shared by two
        # dependents resolving concurrently. Without per-consumer buffering the
        # two would share a single generator and split its items.
        @step("test_async_executor_shared_sync::source", version="1", cacheable=False)
        async def _() -> Iterator[int]:
            await asyncio.sleep(0.1)
            return iter(range(5))

        @step("test_async_executor_shared_sync::sum", version="1")
        async def _(source: Iterator[int]) -> int:
            return sum(list(source))

        @step("test_async_executor_shared_sync::count", version="1")
        async def _(source: Iterator[int]) -> int:
            return len(list(source))

        @step("test_async_executor_shared_sync::combine", version="1")
        async def _(total: int, cnt: int) -> tuple[int, int]:
            return total, cnt

        graph = WorkflowGraph.from_config(
            {
                "steps": {
                    "source": {"type": "test_async_executor_shared_sync::source"},
                    "total": {
                        "type": "test_async_executor_shared_sync::sum",
                        "source": {"type": "ref", "ref": "source"},
                    },
                    "count": {
                        "type": "test_async_executor_shared_sync::count",
                        "source": {"type": "ref", "ref": "source"},
                    },
                    "result": {
                        "type": "test_async_executor_shared_sync::combine",
                        "total": {"type": "ref", "ref": "total"},
                        "cnt": {"type": "ref", "ref": "count"},
                    },
                }
            }
        )

        context = AsyncWorkflowExecutor(max_concurrency=4)(graph, cache=MemoryWorkflowCache())
        assert context.cache[context.info.graph["result"]] == (10, 5)
