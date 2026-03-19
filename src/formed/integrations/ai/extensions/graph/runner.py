"""Streaming execution engine for agent graphs.

This module provides :class:`GraphRunner`, which wraps a :class:`~formed.integrations.ai.extensions.graph.graph.Graph`
and exposes a unified event stream that emits every event from every node in
execution order.  This is useful when you need both live streaming output
*and* the final result from a multi-node workflow.

Examples:
    >>> from formed.integrations.ai.extensions.graph.runner import GraphRunner
    >>> runner = GraphRunner(graph=my_graph)
    >>> async for event in runner.events(state=state, request=request):
    ...     print(event)
    >>> result = await runner.collect(state=state, request=request)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Generic, TypeVar

from ...source import EventSource
from .graph import Graph

StateT = TypeVar("StateT")
RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class GraphRunner(Generic[StateT, RequestT, ResultT]):
    """Streaming execution engine for an agent :class:`~formed.integrations.ai.extensions.graph.graph.Graph`.

    :class:`GraphRunner` drives a graph to completion while publishing each
    node's events to a shared :class:`~formed.integrations.ai.source.EventSource`.
    Consumers can subscribe to the unified stream via :meth:`events` while the
    graph runs in the background.

    Node execution order is determined by each node's
    :attr:`~formed.integrations.ai.extensions.graph.graph.Node.edges` callable,
    exactly as in :meth:`~formed.integrations.ai.extensions.graph.graph.Graph.run`.

    Args:
        graph: The :class:`~formed.integrations.ai.extensions.graph.graph.Graph`
            to execute.

    Examples:
        >>> runner = GraphRunner(graph=my_graph)
        >>> # Stream all events from all nodes
        >>> async for event in runner.events(state=state, request=request):
        ...     print(event)
        >>> # Or simply collect the final result
        >>> result = await runner.collect(state=state, request=request)
    """

    def __init__(self, graph: Graph[StateT, RequestT, ResultT]) -> None:
        self._graph = graph

    def events(self, state: StateT, request: RequestT) -> AsyncIterator[Any]:
        """Return a unified async stream of events from all graph nodes.

        The graph runs as a background task.  Events from each node are
        published to a shared :class:`~formed.integrations.ai.source.EventSource`
        and forwarded to the caller in the order they are produced.  The
        stream closes automatically when the last node finishes (or when an
        exception is raised).

        Args:
            state: Shared mutable state passed to every agent.
            request: Initial request handed to the entry node.

        Returns:
            An :class:`~collections.abc.AsyncIterator` that yields events from
            all nodes in execution order.
        """
        source: EventSource[Any] = EventSource()

        async def _run() -> None:
            exc: BaseException | None = None
            try:
                current_node_name: str | None = self._graph.entry
                current_request: Any = request

                while current_node_name is not None:
                    node = self._graph.nodes[current_node_name]
                    response = node.agent(state, current_request)
                    async for event in response.events():
                        await source.publish(event)
                    _, result = await response.collect()
                    current_node_name = node.edges(result)
                    current_request = result
            except BaseException as e:
                exc = e
                raise
            finally:
                await source.aclose(exception=exc)

        asyncio.create_task(_run())

        async def _iter() -> AsyncIterator[Any]:
            async with source.subscribe() as stream:
                async for e in stream:
                    yield e

        return _iter()

    async def collect(self, state: StateT, request: RequestT) -> ResultT:
        """Execute the entire graph and return the final result.

        This is a convenience wrapper around
        :meth:`~formed.integrations.ai.extensions.graph.graph.Graph.run` for
        callers that do not need the event stream.

        Args:
            state: Shared mutable state passed to every agent.
            request: Initial request handed to the entry node.

        Returns:
            The result produced by the last node in the graph.
        """
        return await self._graph.run(state, request)
