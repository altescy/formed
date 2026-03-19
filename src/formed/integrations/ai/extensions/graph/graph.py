"""Graph execution engine and response handle.

:class:`Graph` is a directed graph of :class:`~step.Step` instances.  Each
step's terminal value is forwarded as the request to the next step, and the
``GraphState`` is threaded through every step via each step's
:class:`~lens.Lens`.

:class:`GraphResponse` is the lazy handle returned by :meth:`Graph.__call__`.
Execution does not start until one of its streaming or collection methods is
called, so multiple subscribers can safely register before the graph begins.

Routing
-------
Edges are evaluated in the following order for each step:

1. :class:`~edges.DirectEdge` — wins immediately.
2. :class:`~edges.ConditionalEdge` — evaluated in definition order.
3. :attr:`~edges.Cond.default` — final fallback.
4. No match → :exc:`GraphConditionError`.

Sub-graphs
----------
:meth:`Graph.as_step` converts a :class:`Graph` into a :class:`~step.Step`
that can be embedded in an outer graph.  Inner
:data:`~events.GraphEvent` items are emitted as the step's ``EventT`` so the
outer :class:`GraphResponse` can expose them.

Examples:
    >>> graph: Graph[MyState, str, StreamEvent, str] = Graph(
    ...     step_a >> step_b,
    ...     step_b >> step_c.when(lambda r: r == "yes"),
    ...     step_b >> step_d.when(Cond.default),
    ...     entry=step_a,
    ... )
    >>> response = graph(state=MyState(), input="hello")
    >>> final_state, result = await response.collect()
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, Generic, TypeVar

from ...response import Response
from ...source import EventSource
from .edges import ConditionalEdge, DirectEdge, Edge, _DefaultCondition
from .events import GraphEvent, StepEvent, StepFinished, StepStarted
from .lens import Lens
from .step import Step

EventT = TypeVar("EventT")
FinalT = TypeVar("FinalT")
GraphStateT = TypeVar("GraphStateT")
InnerGraphStateT = TypeVar("InnerGraphStateT")
ItemT = TypeVar("ItemT")
RequestT = TypeVar("RequestT")


class GraphConditionError(Exception):
    """Raised when no edge condition matches and no :attr:`~edges.Cond.default` is set.

    Add a :class:`~edges.ConditionalEdge` with :attr:`~edges.Cond.default` as
    the last edge from any step that uses conditional routing to avoid this
    error.
    """


# ---------------------------------------------------------------------------
# GraphResponse
# ---------------------------------------------------------------------------


class GraphResponse(Generic[EventT, GraphStateT, FinalT]):
    """Lazy handle to a graph execution.

    Execution does not begin until :meth:`events`, :meth:`step_events`,
    :meth:`select`, or :meth:`collect` is called.  Multiple subscribers can
    safely register before the graph starts.

    Each call to :meth:`events`, :meth:`step_events`, or :meth:`select`
    creates an independent subscription so multiple concurrent consumers can
    observe the same run.

    Args:
        source: The :class:`~formed.integrations.ai.source.EventSource` that
            the graph publishes :data:`~events.GraphEvent` items to.
        coro: Zero-argument coroutine factory that runs the graph and returns
            ``(final_graph_state, final_output)``.
    """

    def __init__(
        self,
        source: EventSource[GraphEvent],
        coro: Callable[[], Coroutine[Any, Any, tuple[GraphStateT, FinalT]]],
    ) -> None:
        self._source = source
        self._coro = coro
        self._task: asyncio.Task[tuple[GraphStateT, FinalT]] | None = None

    def _ensure_started(self) -> asyncio.Task[tuple[GraphStateT, FinalT]]:
        if self._task is None:
            self._task = asyncio.create_task(self._coro())
        return self._task

    def events(self, replay: int = 0) -> AsyncIterator[GraphEvent]:
        """Yield all :data:`~events.GraphEvent` items in chronological order.

        Starts the graph if it has not started yet.

        Args:
            replay: Number of already-emitted events to replay before live
                events.  Passed to
                :meth:`~formed.integrations.ai.source.EventSource.subscribe`.

        Returns:
            An :class:`~collections.abc.AsyncIterator` of
            :data:`~events.GraphEvent` items.
        """
        self._ensure_started()

        async def _iter() -> AsyncIterator[GraphEvent]:
            async with self._source.subscribe(replay=replay) as stream:
                async for e in stream:
                    yield e

        return _iter()

    def step_events(
        self,
        step: Step[Any, Any, Any, EventT, Any],
    ) -> AsyncIterator[EventT]:
        """Yield only the inner ``EventT`` events emitted by *step*.

        Matched by :attr:`~step.Step.name`; the ``EventT`` type is inferred
        from the step object so callers get full type safety.

        Args:
            step: The :class:`~step.Step` whose events to observe.

        Returns:
            An :class:`~collections.abc.AsyncIterator` of the raw inner events
            for the given step.
        """

        async def _iter() -> AsyncIterator[EventT]:
            async for e in self.events():
                if isinstance(e, StepEvent) and e.step_name == step.name:
                    yield e.event

        return _iter()

    def select(
        self,
        fn: Callable[[GraphEvent], ItemT | None],
    ) -> AsyncIterator[ItemT]:
        """Map and filter :data:`~events.GraphEvent` items with *fn*.

        Items for which *fn* returns ``None`` are dropped.

        Args:
            fn: Projection function.  Return a value to include it; return
                ``None`` to skip the event.

        Returns:
            An :class:`~collections.abc.AsyncIterator` of the non-``None``
            values returned by *fn*.
        """

        async def _iter() -> AsyncIterator[ItemT]:
            async for e in self.events():
                if (x := fn(e)) is not None:
                    yield x

        return _iter()

    async def collect(self) -> tuple[GraphStateT, FinalT]:
        """Await graph completion and return ``(final_graph_state, final_output)``.

        Starts the graph if it has not started yet.

        Returns:
            A tuple of ``(final_graph_state, final_output)`` where
            *final_graph_state* is the :class:`~lens.Lens`-merged state after
            the last step and *final_output* is the last step's terminal value.

        Raises:
            Exception: Any exception raised inside the graph coroutine,
                including :exc:`GraphConditionError`.
        """
        return await self._ensure_started()


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class Graph(Generic[GraphStateT, RequestT, EventT, FinalT]):
    """Directed graph of :class:`~step.Step` instances.

    Each step's terminal value is forwarded as the *request* to the next step.
    ``GraphState`` is threaded through every step via each step's
    :class:`~lens.Lens`.

    Calling an instance returns a lazy :class:`GraphResponse`; execution does
    not begin until the response is consumed.

    Args:
        *edges: :class:`~edges.DirectEdge` or :class:`~edges.ConditionalEdge`
            instances produced by ``step_a >> step_b`` or
            ``step_a >> step_b.when(cond)``.
        entry: The step where execution begins.

    Note:
        Step names must be unique within the graph;
        :meth:`GraphResponse.step_events` matches by name.

    Examples:
        >>> graph: Graph[MyState, str, StreamEvent, str] = Graph(
        ...     step_a >> step_b,
        ...     step_b >> step_c.when(lambda r: r == "yes"),
        ...     step_b >> step_d.when(Cond.default),
        ...     entry=step_a,
        ... )
        >>> response = graph(state=MyState(), input="hello")
        >>> final_state, result = await response.collect()
    """

    def __init__(
        self,
        *edges: Edge,
        entry: Step[GraphStateT, Any, RequestT, EventT, Any],
    ) -> None:
        self._edges = list(edges)
        self._entry = entry

    def _next_step(
        self,
        current: Step[Any, Any, Any, Any, Any],
        output: Any,
    ) -> Step[Any, Any, Any, Any, Any] | None:
        direct = [e for e in self._edges if isinstance(e, DirectEdge) and e.src is current]
        if direct:
            return direct[0].dst

        conditional = [e for e in self._edges if isinstance(e, ConditionalEdge) and e.src is current]
        if not conditional:
            return None

        default_edge: ConditionalEdge[Any, Any, Any, Any, Any, Any] | None = None
        for edge in conditional:
            if isinstance(edge.condition, _DefaultCondition):
                default_edge = edge
            elif edge.condition(output):
                return edge.dst

        if default_edge is not None:
            return default_edge.dst

        raise GraphConditionError(
            f"No matching edge from {current.name!r} for output {output!r}. Add Cond.default as a fallback."
        )

    def __call__(
        self,
        graph_state: GraphStateT,
        input: RequestT,
    ) -> GraphResponse[EventT, GraphStateT, FinalT]:
        """Execute the graph and return a lazy :class:`GraphResponse` handle.

        Execution is deferred until :meth:`GraphResponse.collect` or one of
        the streaming methods is called.

        Args:
            graph_state: Initial shared state passed to the first step via its
                :class:`~lens.Lens`.
            input: Initial request forwarded to the entry step's agent.

        Returns:
            A :class:`GraphResponse` for streaming events and collecting the
            final ``(graph_state, result)`` tuple.
        """
        source: EventSource[GraphEvent] = EventSource()
        initial_graph_state = graph_state
        initial_input = input

        async def _run() -> tuple[GraphStateT, FinalT]:
            graph_state = initial_graph_state
            current_step: Step[Any, Any, Any, Any, Any] = self._entry
            current_input: Any = initial_input
            exc: BaseException | None = None

            try:
                while True:
                    await source.publish(StepStarted(step_name=current_step.name, input=current_input))

                    agent_state = current_step.lens.project(graph_state)
                    agent_response = current_step.agent(agent_state, current_input)

                    async def _forward(
                        resp: Response[Any, Any, Any],
                        sname: str,
                    ) -> None:
                        async for e in resp.events():
                            await source.publish(StepEvent(step_name=sname, event=e))

                    forward_task = asyncio.create_task(_forward(agent_response, current_step.name))

                    try:
                        final_agent_state, terminal = await agent_response.collect()
                    except BaseException:
                        forward_task.cancel()
                        raise

                    await forward_task

                    graph_state = current_step.lens.inject(graph_state, final_agent_state)

                    await source.publish(StepFinished(step_name=current_step.name, output=terminal))

                    next_step = self._next_step(current_step, terminal)
                    if next_step is None:
                        return graph_state, terminal

                    current_input = terminal
                    current_step = next_step

            except BaseException as e:
                exc = e if not isinstance(e, asyncio.CancelledError) else None
                raise
            finally:
                await source.aclose(exception=exc)

        return GraphResponse(source=source, coro=_run)

    def as_step(
        self,
        lens: Lens[GraphStateT, InnerGraphStateT],
        name: str,
    ) -> Step[GraphStateT, InnerGraphStateT, RequestT, GraphEvent, FinalT]:
        """Wrap this graph as a :class:`~step.Step` for use in an outer graph.

        The inner ``GraphState`` is treated as the step's ``AgentStateT`` and
        connected to the outer ``GraphState`` via *lens*.  Inner
        :data:`~events.GraphEvent` items are emitted as the step's ``EventT``
        so the outer :meth:`GraphResponse.step_events` can expose them.

        Args:
            lens: Maps outer ``GraphState`` ↔ inner ``GraphState``.
            name: Step name used in the outer graph.

        Returns:
            A :class:`~step.Step` that runs this entire graph as a single node.

        Note:
            Inner events appear as :class:`~events.StepEvent` ``[GraphEvent]``
            in the outer stream.  Flatten them with
            :meth:`GraphResponse.select` if a flat event stream is needed.

        Examples:
            >>> sub = inner_graph.as_step(
            ...     lens=Lens(project=lambda gs: gs.inner, inject=lambda gs, s: replace(gs, inner=s)),
            ...     name="sub",
            ... )
            >>> outer_graph = Graph(sub >> next_step, entry=sub)
        """
        graph = self

        class _GraphAsAgent:
            def __call__(
                _self,
                inner_state: InnerGraphStateT,
                request: RequestT,
            ) -> Response[GraphEvent, InnerGraphStateT, FinalT]:
                gr = graph(inner_state, request)  # type: ignore[arg-type]
                inner_source: EventSource[GraphEvent] = EventSource()

                async def _run() -> tuple[InnerGraphStateT, FinalT]:
                    exc: BaseException | None = None
                    try:

                        async def _forward() -> None:
                            async for e in gr.events():
                                await inner_source.publish(e)

                        forward_task = asyncio.create_task(_forward())
                        try:
                            inner_gs, result = await gr.collect()
                        except BaseException:
                            forward_task.cancel()
                            raise
                        await forward_task
                        return inner_gs, result  # type: ignore[return-value]
                    except BaseException as e:
                        exc = e if not isinstance(e, asyncio.CancelledError) else None
                        raise
                    finally:
                        await inner_source.aclose(exception=exc)

                return Response(source=inner_source, coro=_run)

        return Step(
            agent=_GraphAsAgent(),  # type: ignore[arg-type]
            lens=lens,
            name=name,
        )


__all__ = [
    "Graph",
    "GraphConditionError",
    "GraphResponse",
]
