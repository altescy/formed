"""Agent graph for composing multi-step AI workflows.

This module provides :class:`Graph` and its supporting types for wiring
:class:`~formed.integrations.ai.agent.Agent` instances into a directed
execution graph.  Each node holds one agent and an *edges* function that
inspects the agent's result and returns the name of the next node to run
(or ``None`` to terminate).  This lets you express conditional branching and
loops without any special framework support.

The module also exposes :class:`GraphExecution`, a thin handle returned by
:meth:`Graph.responses` that you can use when you need access to the final
result without running the graph yourself.

Examples:
    >>> from formed.integrations.ai.extensions.graph.graph import Graph, Node
    >>> graph = Graph(
    ...     nodes={
    ...         "analyze": Node(
    ...             agent=analyze_agent,
    ...             edges=lambda r: "respond" if r.needs_response else None,
    ...         ),
    ...         "respond": Node(agent=respond_agent, edges=lambda r: None),
    ...     },
    ...     entry="analyze",
    ... )
    >>> result = await graph.run(state=MyState(), request=my_request)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from ...agent import Agent
from ...response import Response

StateT = TypeVar("StateT")
RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")
EventT = TypeVar("EventT")


@dataclasses.dataclass
class Node(Generic[StateT, RequestT, ResultT, EventT]):
    """A single node in an agent graph.

    A node pairs one :class:`~formed.integrations.ai.agent.Agent` with an
    *edges* callable that decides which node to visit next based on the
    agent's result.

    Attributes:
        agent: The agent executed at this node.
        edges: Callable that receives the agent's result and returns the name
            of the next node, or ``None`` to end graph execution.
    """

    agent: Agent[RequestT, Any, EventT, StateT, Any, ResultT]
    edges: Callable[[ResultT], str | None]


@dataclasses.dataclass
class Graph(Generic[StateT, RequestT, ResultT]):
    """Directed agent graph that executes nodes sequentially.

    Each node's :attr:`~Node.edges` function inspects the preceding result and
    returns the next node name, enabling conditional branching and loops.
    Execution terminates when ``edges`` returns ``None``.

    The result of each node is forwarded as the *request* to the next node,
    allowing nodes to build on each other's output.

    Args:
        nodes: Mapping from node name to :class:`Node` instance.
        entry: Name of the node where execution begins.

    Examples:
        >>> graph = Graph(
        ...     nodes={
        ...         "classify": Node(
        ...             agent=classify_agent,
        ...             edges=lambda r: "answer" if r.confident else "clarify",
        ...         ),
        ...         "clarify": Node(agent=clarify_agent, edges=lambda r: "classify"),
        ...         "answer": Node(agent=answer_agent, edges=lambda r: None),
        ...     },
        ...     entry="classify",
        ... )
        >>> result = await graph.run(state=state, request=question)
    """

    nodes: dict[str, Node[StateT, Any, Any, Any]]
    entry: str

    async def run(self, state: StateT, request: RequestT) -> ResultT:
        """Execute the graph from the entry node and return the final result.

        Nodes are executed sequentially.  Each node's result is passed as the
        request to the next node selected by :attr:`~Node.edges`.  Execution
        stops when ``edges`` returns ``None``.

        Args:
            state: Shared mutable state passed to every agent in the graph.
            request: Initial request handed to the entry node.

        Returns:
            The result produced by the last node executed.
        """
        current_node_name: str | None = self.entry
        current_request: Any = request
        result: Any = None

        while current_node_name is not None:
            node = self.nodes[current_node_name]
            response: Response[Any, Any] = node.agent(state, current_request)
            result = await response.collect()
            current_node_name = node.edges(result)
            current_request = result

        return result  # type: ignore[return-value]

    def responses(self, state: StateT, request: RequestT) -> GraphExecution[StateT, RequestT, ResultT]:
        """Return a :class:`GraphExecution` handle for this graph run.

        Use the returned handle when you want to drive execution yourself or
        when you need a :class:`~formed.integrations.ai.response.Response`-
        compatible object to pass to downstream consumers.

        Args:
            state: Shared mutable state passed to every agent in the graph.
            request: Initial request handed to the entry node.

        Returns:
            A :class:`GraphExecution` bound to this graph, *state*, and
            *request*.
        """
        return GraphExecution(graph=self, state=state, request=request)


class GraphExecution(Generic[StateT, RequestT, ResultT]):
    """Execution handle for a single :class:`Graph` run.

    Created by :meth:`Graph.responses`; provides a :meth:`run` method that
    delegates to the underlying :class:`Graph`.

    Args:
        graph: The :class:`Graph` to execute.
        state: Shared mutable state passed to every agent.
        request: Initial request handed to the entry node.
    """

    def __init__(self, graph: Graph[StateT, RequestT, ResultT], state: StateT, request: RequestT) -> None:
        self._graph = graph
        self._state = state
        self._request = request

    async def run(self) -> ResultT:
        """Execute the graph and return the final result.

        Returns:
            The result produced by the last node in the graph.
        """
        return await self._graph.run(self._state, self._request)
