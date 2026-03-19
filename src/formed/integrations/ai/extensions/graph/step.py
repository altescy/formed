"""Single node in an agent graph.

A :class:`Step` owns one :class:`~formed.integrations.ai.agent.Agent` and a
:class:`~lens.Lens` that bridges the graph's shared state and the agent's
local state.  The agent knows nothing about the graph; the :class:`Lens`
carries all mapping responsibility.

Edge construction
-----------------
Use ``>>`` to wire steps together:

- ``step_a >> step_b`` — unconditional (:class:`~edges.DirectEdge`).
- ``step_a >> step_b.when(pred)`` — conditional (:class:`~edges.ConditionalEdge`).
- ``step_a >> step_b.when(Cond.default)`` — catch-all fallback.

The ``>>`` operator is type-checked by pyright: the source step's
``TerminalT`` must match the destination step's ``RequestT``.

Sub-graph nodes
---------------
:meth:`~graph.Graph.as_step` returns a :class:`Step` backed by an entire
:class:`~graph.Graph`, enabling graph composition.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from ...agent import Agent
from .edges import ConditionalEdge, ConditionalStep, DirectEdge, _DefaultCondition
from .lens import Lens

AgentStateT = TypeVar("AgentStateT")
EventT = TypeVar("EventT")
GraphStateT = TypeVar("GraphStateT")
NextTerminalT = TypeVar("NextTerminalT")
RequestT = TypeVar("RequestT")
TerminalT = TypeVar("TerminalT")


class Step(Generic[GraphStateT, AgentStateT, RequestT, EventT, TerminalT]):
    """A single node in the agent graph.

    Holds one :class:`~formed.integrations.ai.agent.Agent` and a
    :class:`~lens.Lens` that connects the graph's shared ``GraphState`` to
    the agent's local ``AgentState``.  The agent is fully decoupled from
    the graph structure and can be reused across multiple graphs.

    Args:
        agent: The agent executed at this node.
        lens: Bidirectional ``GraphState ↔ AgentState`` mapping.
            The same instance can be shared across multiple steps.
        name: Unique step name within the graph.  Used in
            :data:`~events.GraphEvent` payloads and matched by
            :meth:`~graph.GraphResponse.step_events`.

    Examples:
        >>> step_a: Step[GS, AS, str, E, int] = Step(agent_a, lens, "a")
        >>> step_b: Step[GS, AS, int, E, str] = Step(agent_b, lens, "b")
        >>> edge = step_a >> step_b   # TerminalT=int matches RequestT=int  ✓
    """

    def __init__(
        self,
        agent: Agent[RequestT, Any, EventT, AgentStateT, Any, TerminalT],
        lens: Lens[GraphStateT, AgentStateT],
        name: str,
    ) -> None:
        self._agent = agent
        self._lens = lens
        self._name = name

    @property
    def name(self) -> str:
        """Unique step name within the graph."""
        return self._name

    @property
    def agent(self) -> Agent[RequestT, Any, EventT, AgentStateT, Any, TerminalT]:
        """The agent executed at this step."""
        return self._agent

    @property
    def lens(self) -> Lens[GraphStateT, AgentStateT]:
        """The ``GraphState ↔ AgentState`` mapping for this step."""
        return self._lens

    def __rshift__(
        self,
        other: Step[GraphStateT, Any, TerminalT, EventT, NextTerminalT]
        | ConditionalStep[GraphStateT, Any, TerminalT, EventT, NextTerminalT],
    ) -> (
        DirectEdge[GraphStateT, AgentStateT, RequestT, EventT, TerminalT, NextTerminalT]
        | ConditionalEdge[GraphStateT, AgentStateT, RequestT, EventT, TerminalT, NextTerminalT]
    ):
        """Create an edge from this step to *other*.

        Args:
            other: Destination step or conditional step produced by
                :meth:`when`.  When a plain :class:`Step` is supplied a
                :class:`~edges.DirectEdge` is returned; a
                :class:`~edges.ConditionalStep` yields a
                :class:`~edges.ConditionalEdge`.

        Returns:
            A :class:`~edges.DirectEdge` or :class:`~edges.ConditionalEdge`
            ready to be passed to :class:`~graph.Graph`.
        """
        if isinstance(other, ConditionalStep):
            return ConditionalEdge(src=self, dst=other.step, condition=other.condition)
        return DirectEdge(src=self, dst=other)

    def when(
        self,
        condition: Callable[[TerminalT], bool] | _DefaultCondition,
    ) -> ConditionalStep[GraphStateT, AgentStateT, RequestT, EventT, TerminalT]:
        """Annotate this step as a conditional transition target.

        Wraps ``self`` in a :class:`~edges.ConditionalStep` so that
        ``step_a >> step_b.when(pred)`` produces a
        :class:`~edges.ConditionalEdge`.

        Args:
            condition: Predicate called with the *source* step's terminal
                value.  Pass :attr:`~edges.Cond.default` as a catch-all
                fallback for the last branch.

        Returns:
            A :class:`~edges.ConditionalStep` wrapping this step and the
            given condition.

        Examples:
            >>> step_a >> step_b.when(lambda out: out == "yes")
            >>> step_a >> step_c.when(Cond.default)
        """
        return ConditionalStep(step=self, condition=condition)

    def __repr__(self) -> str:
        return f"Step({self._name!r})"


__all__ = ["Step"]
