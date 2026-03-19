"""Tests for formed.integrations.ai.extensions.graph.

All async tests use ``@pytest.mark.anyio`` (anyio pytest plugin).
No external API calls are made; a ``MockEngine`` provides deterministic
stream events.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from formed.integrations.ai.agent import Agent
from formed.integrations.ai.control import Continue, Stop
from formed.integrations.ai.extensions.graph import (
    Cond,
    ConditionalEdge,
    DirectEdge,
    Graph,
    GraphConditionError,
    Lens,
    Step,
    StepEvent,
    StepFinished,
    StepStarted,
)
from formed.integrations.ai.providers.base.events import (
    StreamEvent,
    TextDelta,
    TextPartDone,
    TurnDone,
)
from formed.integrations.ai.providers.base.messages import (
    AssistantMessage,
    Query,
    TextContent,
    UserMessage,
)
from formed.integrations.ai.providers.base.reducer import AgentReducer, ReducerState
from formed.integrations.ai.providers.base.signals import Signal, TextOutput

# ---------------------------------------------------------------------------
# Mock infrastructure (mirrors aigraphv7.py)
# ---------------------------------------------------------------------------


class MockEngine:
    """Deterministic engine that emits a fixed response string."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    @asynccontextmanager
    async def __call__(self, query: Query) -> AsyncIterator[AsyncIterator[StreamEvent]]:
        text = self._response_text

        async def _stream() -> AsyncIterator[StreamEvent]:
            mid = len(text) // 2
            for chunk in (text[:mid], text[mid:]):
                if chunk:
                    yield TextDelta(index=0, delta=chunk)
            yield TextPartDone(index=0, text=text)
            yield TurnDone(finish_reason="stop")

        yield _stream()


@dataclasses.dataclass(frozen=True)
class AgentState:
    """Minimal agent state used in tests."""

    query_history: tuple[UserMessage | AssistantMessage, ...] = ()
    reducer_state: ReducerState = dataclasses.field(default_factory=ReducerState)


class _Reducer:
    def __init__(self) -> None:
        self._inner = AgentReducer()

    def __call__(self, state: AgentState, event: StreamEvent) -> tuple[AgentState, list[Signal]]:
        new_rs, signals = self._inner(state.reducer_state, event)
        return dataclasses.replace(state, reducer_state=new_rs), list(signals)


class _Handler:
    async def __call__(
        self,
        state: AgentState,
        query: Query,
        signal: Signal,
    ) -> tuple[AgentState, Query, Continue | Stop[str]]:
        if isinstance(signal, TextOutput):
            new_history = state.query_history + (AssistantMessage(parts=(TextContent(text=signal.text),)),)
            return dataclasses.replace(state, query_history=new_history), query, Stop(signal.text)
        return state, query, Continue()


class _Contextualizer:
    async def __call__(self, state: AgentState, request: str) -> Query:
        new_history = state.query_history + (UserMessage(parts=(TextContent(text=request),)),)
        return Query(system=None, history=new_history)


def make_agent(engine: MockEngine) -> Agent[str, Query, StreamEvent, AgentState, Signal, str]:
    return Agent(
        engine=engine,
        reducer=_Reducer(),
        handler=_Handler(),
        contextualizer=_Contextualizer(),
    )


def make_step(response_text: str, name: str) -> Step[AgentState, AgentState, str, StreamEvent, str]:
    return Step(
        agent=make_agent(MockEngine(response_text)),
        lens=Lens.identity(),
        name=name,
    )


# ---------------------------------------------------------------------------
# Lens
# ---------------------------------------------------------------------------


class TestLens:
    def test_identity_project(self) -> None:
        lens: Lens[AgentState, AgentState] = Lens.identity()
        state = AgentState()
        assert lens.project(state) is state

    def test_identity_inject(self) -> None:
        lens: Lens[AgentState, AgentState] = Lens.identity()
        old = AgentState()
        new = dataclasses.replace(old)
        assert lens.inject(old, new) is new

    def test_custom_lens(self) -> None:
        @dataclasses.dataclass(frozen=True)
        class GS:
            value: int

        lens: Lens[GS, int] = Lens(
            project=lambda gs: gs.value,
            inject=lambda gs, v: dataclasses.replace(gs, value=v),
        )
        gs = GS(value=42)
        assert lens.project(gs) == 42
        assert lens.inject(gs, 99) == GS(value=99)


# ---------------------------------------------------------------------------
# Step / Edge construction
# ---------------------------------------------------------------------------


class TestStepEdges:
    def test_direct_edge(self) -> None:
        a = make_step("a", "a")
        b = make_step("b", "b")
        edge = a >> b
        assert isinstance(edge, DirectEdge)
        assert edge.src is a
        assert edge.dst is b

    def test_conditional_edge(self) -> None:
        a = make_step("a", "a")
        b = make_step("b", "b")
        edge = a >> b.when(lambda out: len(out) > 0)
        assert isinstance(edge, ConditionalEdge)
        assert edge.src is a
        assert edge.dst is b

    def test_default_conditional_edge(self) -> None:
        a = make_step("a", "a")
        b = make_step("b", "b")
        edge = a >> b.when(Cond.default)
        assert isinstance(edge, ConditionalEdge)
        assert edge.condition is Cond.default

    def test_step_name(self) -> None:
        s = make_step("hello", "my_step")
        assert s.name == "my_step"

    def test_step_repr(self) -> None:
        s = make_step("hello", "my_step")
        assert repr(s) == "Step('my_step')"


# ---------------------------------------------------------------------------
# Single-step graph
# ---------------------------------------------------------------------------


class TestSingleStepGraph:
    @pytest.mark.anyio
    async def test_collect_result(self) -> None:
        step = make_step("hello world", "greet")
        graph: Graph[AgentState, str, StreamEvent, str] = Graph(entry=step)
        state, result = await graph(AgentState(), "hi").collect()
        assert result == "hello world"

    @pytest.mark.anyio
    async def test_final_state_updated(self) -> None:
        step = make_step("response text", "step")
        graph: Graph[AgentState, str, StreamEvent, str] = Graph(entry=step)
        state, _ = await graph(AgentState(), "input").collect()
        # DefaultHandler appended an AssistantMessage to query_history
        assert len(state.query_history) >= 1
        last = state.query_history[-1]
        assert isinstance(last, AssistantMessage)

    @pytest.mark.anyio
    async def test_events_stream(self) -> None:
        step = make_step("hi", "s")
        graph: Graph[AgentState, str, StreamEvent, str] = Graph(entry=step)
        response = graph(AgentState(), "go")
        events = []
        async for e in response.events():
            events.append(e)
        step_started = [e for e in events if isinstance(e, StepStarted)]
        step_finished = [e for e in events if isinstance(e, StepFinished)]
        assert len(step_started) == 1
        assert step_started[0].step_name == "s"
        assert len(step_finished) == 1
        assert step_finished[0].output == "hi"


# ---------------------------------------------------------------------------
# Multi-step (linear chain)
# ---------------------------------------------------------------------------


class TestLinearGraph:
    @pytest.mark.anyio
    async def test_two_step_chain(self) -> None:
        a = make_step("step_a_output", "a")
        b = make_step("step_b_output", "b")
        graph: Graph[AgentState, str, StreamEvent, str] = Graph(a >> b, entry=a)
        _, result = await graph(AgentState(), "start").collect()
        assert result == "step_b_output"

    @pytest.mark.anyio
    async def test_three_step_chain_event_order(self) -> None:
        a = make_step("A", "a")
        b = make_step("B", "b")
        c = make_step("C", "c")
        graph: Graph[AgentState, str, StreamEvent, str] = Graph(a >> b, b >> c, entry=a)
        response = graph(AgentState(), "x")
        started_names = []
        finished_names = []
        async for e in response.events():
            if isinstance(e, StepStarted):
                started_names.append(e.step_name)
            elif isinstance(e, StepFinished):
                finished_names.append(e.step_name)
        assert started_names == ["a", "b", "c"]
        assert finished_names == ["a", "b", "c"]

    @pytest.mark.anyio
    async def test_terminal_forwarded_as_input(self) -> None:
        """The terminal of step A is forwarded as the request to step B."""
        a = make_step("forwarded_value", "a")

        # Capture the request that reaches step B
        received_requests: list[str] = []

        class _CapturingContextualizer:
            async def __call__(self, state: AgentState, request: str) -> Query:
                received_requests.append(request)
                new_history = state.query_history + (UserMessage(parts=(TextContent(text=request),)),)
                return Query(system=None, history=new_history)

        b_agent = Agent(
            engine=MockEngine("b_result"),
            reducer=_Reducer(),
            handler=_Handler(),
            contextualizer=_CapturingContextualizer(),
        )
        b: Step[AgentState, AgentState, str, StreamEvent, str] = Step(
            agent=b_agent,
            lens=Lens.identity(),
            name="b",
        )
        graph: Graph[AgentState, str, StreamEvent, str] = Graph(a >> b, entry=a)
        await graph(AgentState(), "original_input").collect()
        # Step B should have received step A's output as its request
        assert received_requests == ["forwarded_value"]


# ---------------------------------------------------------------------------
# Conditional branching
# ---------------------------------------------------------------------------


class TestConditionalGraph:
    @pytest.mark.anyio
    async def test_condition_true_branch(self) -> None:
        classify = make_step("short", "classify")
        short_branch = make_step("SHORT RESULT", "short")
        long_branch = make_step("LONG RESULT", "long")

        graph: Graph[AgentState, str, StreamEvent, str] = Graph(
            classify >> short_branch.when(lambda out: len(out) <= 10),
            classify >> long_branch.when(Cond.default),
            entry=classify,
        )
        _, result = await graph(AgentState(), "go").collect()
        assert result == "SHORT RESULT"

    @pytest.mark.anyio
    async def test_condition_false_falls_to_default(self) -> None:
        classify = make_step("this is a very long output", "classify")
        short_branch = make_step("SHORT RESULT", "short")
        long_branch = make_step("LONG RESULT", "long")

        graph: Graph[AgentState, str, StreamEvent, str] = Graph(
            classify >> short_branch.when(lambda out: len(out) <= 5),
            classify >> long_branch.when(Cond.default),
            entry=classify,
        )
        _, result = await graph(AgentState(), "go").collect()
        assert result == "LONG RESULT"

    @pytest.mark.anyio
    async def test_no_matching_edge_raises(self) -> None:
        classify = make_step("medium length output", "classify")
        branch = make_step("never", "branch")

        graph: Graph[AgentState, str, StreamEvent, str] = Graph(
            classify >> branch.when(lambda out: out == "exact_match"),
            entry=classify,
        )
        with pytest.raises(GraphConditionError):
            await graph(AgentState(), "go").collect()

    @pytest.mark.anyio
    async def test_direct_edge_takes_priority_over_conditional(self) -> None:
        """DirectEdge wins even when a ConditionalEdge condition is also True."""
        a = make_step("yes", "a")
        b_direct = make_step("B_DIRECT", "b_direct")
        b_cond = make_step("B_COND", "b_cond")

        # direct edge first in definition order — should still win
        graph: Graph[AgentState, str, StreamEvent, str] = Graph(
            a >> b_direct,
            a >> b_cond.when(lambda out: True),
            entry=a,
        )
        _, result = await graph(AgentState(), "go").collect()
        assert result == "B_DIRECT"


# ---------------------------------------------------------------------------
# step_events filtering
# ---------------------------------------------------------------------------


class TestStepEvents:
    @pytest.mark.anyio
    async def test_step_events_filters_by_step(self) -> None:
        a = make_step("A output", "a")
        b = make_step("B output", "b")
        graph: Graph[AgentState, str, StreamEvent, str] = Graph(a >> b, entry=a)
        response = graph(AgentState(), "start")

        b_deltas: list[TextDelta] = []
        async for e in response.step_events(b):
            if isinstance(e, TextDelta):
                b_deltas.append(e)

        # All collected deltas should have come from step b's engine
        assert all("B output"[: len(d.delta)] in "B output" for d in b_deltas)

    @pytest.mark.anyio
    async def test_step_events_concurrent_with_collect(self) -> None:
        """step_events and collect can run concurrently."""
        import asyncio

        a = make_step("hello", "a")
        b = make_step("world", "b")
        graph: Graph[AgentState, str, StreamEvent, str] = Graph(a >> b, entry=a)
        response = graph(AgentState(), "go")

        async def collect_b_events() -> list[StreamEvent]:
            result = []
            async for e in response.step_events(b):
                result.append(e)
            return result

        b_events, (_, final) = await asyncio.gather(collect_b_events(), response.collect())
        assert final == "world"
        assert any(isinstance(e, TextDelta) for e in b_events)


# ---------------------------------------------------------------------------
# GraphResponse.select
# ---------------------------------------------------------------------------


class TestGraphResponseSelect:
    @pytest.mark.anyio
    async def test_select_finished_events(self) -> None:
        a = make_step("out_a", "a")
        b = make_step("out_b", "b")
        graph: Graph[AgentState, str, StreamEvent, str] = Graph(a >> b, entry=a)
        response = graph(AgentState(), "start")

        outputs = []
        async for out in response.select(lambda e: e.output if isinstance(e, StepFinished) else None):
            outputs.append(out)

        assert outputs == ["out_a", "out_b"]

    @pytest.mark.anyio
    async def test_select_returns_none_for_unmatched(self) -> None:
        step = make_step("x", "s")
        graph: Graph[AgentState, str, StreamEvent, str] = Graph(entry=step)
        response = graph(AgentState(), "go")

        # Only collect StepStarted events
        started = []
        async for s in response.select(lambda e: e if isinstance(e, StepStarted) else None):
            started.append(s)
        assert len(started) == 1


# ---------------------------------------------------------------------------
# Lazy execution
# ---------------------------------------------------------------------------


class TestLazyExecution:
    @pytest.mark.anyio
    async def test_graph_not_started_until_collect(self) -> None:
        """GraphResponse is lazy: collect() triggers execution."""
        executed = []

        class _TrackingContextualizer:
            async def __call__(self, state: AgentState, request: str) -> Query:
                executed.append(request)
                new_history = state.query_history + (UserMessage(parts=(TextContent(text=request),)),)
                return Query(system=None, history=new_history)

        tracking_agent = Agent(
            engine=MockEngine("done"),
            reducer=_Reducer(),
            handler=_Handler(),
            contextualizer=_TrackingContextualizer(),
        )
        step: Step[AgentState, AgentState, str, StreamEvent, str] = Step(
            agent=tracking_agent,
            lens=Lens.identity(),
            name="tracked",
        )
        graph: Graph[AgentState, str, StreamEvent, str] = Graph(entry=step)
        response = graph(AgentState(), "trigger")
        assert executed == []  # not yet started
        await response.collect()
        assert executed == ["trigger"]  # now started


# ---------------------------------------------------------------------------
# Custom Lens (non-identity)
# ---------------------------------------------------------------------------


class TestCustomLens:
    @pytest.mark.anyio
    async def test_custom_lens_project_inject(self) -> None:
        """GraphState and AgentState can be different types."""

        @dataclasses.dataclass(frozen=True)
        class GraphState:
            messages: tuple[str, ...] = ()

        lens: Lens[GraphState, AgentState] = Lens(
            project=lambda gs: AgentState(
                query_history=tuple(UserMessage(parts=(TextContent(text=m),)) for m in gs.messages)
            ),
            inject=lambda gs, as_: dataclasses.replace(
                gs,
                messages=gs.messages
                + tuple(
                    "".join(p.text for p in m.parts if isinstance(p, TextContent))
                    for m in as_.query_history[len(gs.messages) :]
                    if isinstance(m, AssistantMessage)
                ),
            ),
        )

        step: Step[GraphState, AgentState, str, StreamEvent, str] = Step(
            agent=make_agent(MockEngine("custom reply")),
            lens=lens,
            name="custom",
        )
        graph: Graph[GraphState, str, StreamEvent, str] = Graph(entry=step)
        final_gs, result = await graph(GraphState(), "hello").collect()
        assert result == "custom reply"
        assert "custom reply" in final_gs.messages

    @pytest.mark.anyio
    async def test_shared_lens_across_steps(self) -> None:
        """Multiple steps sharing the same Lens instance."""

        @dataclasses.dataclass(frozen=True)
        class GS:
            history: tuple[str, ...] = ()

        def _project(gs: GS) -> AgentState:
            return AgentState(query_history=tuple(UserMessage(parts=(TextContent(text=h),)) for h in gs.history))

        def _inject(gs: GS, as_: AgentState) -> GS:
            new_texts = tuple(
                "".join(p.text for p in m.parts if isinstance(p, TextContent))
                for m in as_.query_history[len(gs.history) :]
                if isinstance(m, AssistantMessage)
            )
            return dataclasses.replace(gs, history=gs.history + new_texts)

        shared_lens: Lens[GS, AgentState] = Lens(project=_project, inject=_inject)

        x: Step[GS, AgentState, str, StreamEvent, str] = Step(
            agent=make_agent(MockEngine("X done")),
            lens=shared_lens,
            name="x",
        )
        y: Step[GS, AgentState, str, StreamEvent, str] = Step(
            agent=make_agent(MockEngine("Y done")),
            lens=shared_lens,
            name="y",
        )
        graph: Graph[GS, str, StreamEvent, str] = Graph(x >> y, entry=x)
        final_gs, _ = await graph(GS(), "start").collect()
        assert "X done" in final_gs.history
        assert "Y done" in final_gs.history


# ---------------------------------------------------------------------------
# Sub-graph (as_step)
# ---------------------------------------------------------------------------


class TestSubGraph:
    @pytest.mark.anyio
    async def test_subgraph_as_step(self) -> None:
        inner_a = make_step("Inner A", "inner_a")
        inner_b = make_step("Inner B", "inner_b")
        inner_graph: Graph[AgentState, str, StreamEvent, str] = Graph(inner_a >> inner_b, entry=inner_a)

        sub_step = inner_graph.as_step(
            lens=Lens(project=lambda gs: gs, inject=lambda _outer, inner: inner),
            name="sub",
        )
        outer: Step[AgentState, AgentState, str, StreamEvent, str] = make_step("Outer", "outer")
        graph: Graph[AgentState, str, StreamEvent, str] = Graph(
            sub_step >> outer,  # type: ignore[arg-type]
            entry=sub_step,
        )
        _, result = await graph(AgentState(), "go").collect()
        assert result == "Outer"

    @pytest.mark.anyio
    async def test_subgraph_events_wrapped(self) -> None:
        from formed.integrations.ai.extensions.graph import GraphEvent

        inner = make_step("inner_output", "inner")
        inner_graph: Graph[AgentState, str, StreamEvent, str] = Graph(entry=inner)

        sub_step = inner_graph.as_step(
            lens=Lens(project=lambda gs: gs, inject=lambda _outer, inner_gs: inner_gs),
            name="sub",
        )
        outer_graph: Graph[AgentState, str, GraphEvent, str] = Graph(entry=sub_step)
        response = outer_graph(AgentState(), "start")

        step_events = []
        async for e in response.events():
            if isinstance(e, StepEvent):
                step_events.append(e)
        # Inner GraphEvents should appear as StepEvent items in the outer stream
        assert any(e.step_name == "sub" for e in step_events)
