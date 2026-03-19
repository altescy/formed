"""Bidirectional state mapping between ``GraphState`` and ``AgentState``.

A :class:`Lens` is the bridge that lets a :class:`~step.Step` translate
between the *shared* graph state and the *local* agent state without either
knowing about the other.

Design::

    project : GraphState  → AgentState
    inject  : (GraphState, AgentState) → GraphState

``project`` extracts the slice of context that one agent needs.
``inject`` merges the agent's updated state back after the run.  The
``AgentState`` received by ``inject`` is the **post-run** value returned by
``Response.collect()``, so every update the handler accumulated is visible.

The terminal value is deliberately *not* passed to ``inject``.  When the
terminal needs to be written back into the graph state, the handler should
store it inside ``AgentState`` and ``inject`` should read it from there.

The same :class:`Lens` instance can be shared across multiple steps so that
the mapping is defined once and reused.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Generic, TypeVar

AgentStateT = TypeVar("AgentStateT")
GraphStateT = TypeVar("GraphStateT")


@dataclasses.dataclass(frozen=True)
class Lens(Generic[GraphStateT, AgentStateT]):
    """Bidirectional mapping between ``GraphState`` and ``AgentState``.

    Attributes:
        project: ``GraphState → AgentState``.
            Extract the slice of shared context that one agent needs.
        inject: ``(GraphState, AgentState) → GraphState``.
            Merge the agent's updated state back into the graph state.
            The *AgentState* received here is the **post-run** value returned
            by :meth:`~formed.integrations.ai.response.Response.collect`.

    Examples:
        >>> from dataclasses import replace
        >>> lens = Lens(
        ...     project=lambda gs: AgentState(history=gs.messages),
        ...     inject=lambda gs, as_: replace(gs, messages=as_.history),
        ... )
    """

    project: Callable[[GraphStateT], AgentStateT]
    inject: Callable[[GraphStateT, AgentStateT], GraphStateT]

    @staticmethod
    def identity() -> Lens[AgentStateT, AgentStateT]:
        """Return an identity :class:`Lens` for when ``GraphState == AgentState``.

        ``inject`` replaces the graph state wholesale with the updated agent
        state, which is correct when the two types are the same.

        Returns:
            A :class:`Lens` whose ``project`` is the identity function and
            whose ``inject`` discards the old graph state.

        Examples:
            >>> step: Step[MyState, MyState, str, E, str] = Step(
            ...     agent=agent,
            ...     lens=Lens.identity(),
            ...     name="s",
            ... )
        """
        return Lens(project=lambda s: s, inject=lambda _gs, as_: as_)


__all__ = ["Lens"]
