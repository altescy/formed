"""Edge types and condition helpers for graph routing.

Edges connect :class:`~step.Step` instances and encode the routing logic.
They are created exclusively via the ``>>`` operator on :class:`~step.Step`:

- ``step_a >> step_b`` — unconditional transition (:class:`DirectEdge`).
- ``step_a >> step_b.when(pred)`` — conditional transition
  (:class:`ConditionalEdge`).
- ``step_a >> step_b.when(Cond.default)`` — catch-all fallback.

:data:`Edge` is the union alias consumed by :class:`~graph.Graph`.

Condition evaluation order inside the graph:

1. :class:`DirectEdge` — wins immediately; no condition is evaluated.
2. :class:`ConditionalEdge` — evaluated in definition order.
3. :attr:`Cond.default` — used only if every other condition was ``False``.
4. No match → :exc:`~graph.GraphConditionError`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

# Forward reference — Step is defined in step.py which imports from here.
# We use TYPE_CHECKING to avoid a circular import at runtime.
if TYPE_CHECKING:
    from .step import Step

AgentStateT = TypeVar("AgentStateT")
EventT = TypeVar("EventT")
GraphStateT = TypeVar("GraphStateT")
NextTerminalT = TypeVar("NextTerminalT")
RequestT = TypeVar("RequestT")
TerminalT = TypeVar("TerminalT")


# ---------------------------------------------------------------------------
# Condition helpers
# ---------------------------------------------------------------------------


class _DefaultCondition:
    """Sentinel for :attr:`Cond.default`.

    Matches only when every non-default condition on the same source step has
    evaluated to ``False``.
    """

    def __repr__(self) -> str:
        return "Cond.default"


class Cond:
    """Namespace for graph edge condition helpers.

    Examples:
        >>> # Conditional edge — branch when the result flag is truthy
        >>> step_a >> step_b.when(lambda out: out.flag)
        >>> # Fallback edge — taken when no other condition matched
        >>> step_a >> step_c.when(Cond.default)
    """

    default = _DefaultCondition()
    """Catch-all fallback sentinel.

    Assign this as the condition of the last edge from a step to handle any
    terminal value not matched by the preceding :class:`ConditionalEdge`
    instances.
    """


# ---------------------------------------------------------------------------
# ConditionalStep
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ConditionalStep(Generic[GraphStateT, AgentStateT, RequestT, EventT, TerminalT]):
    """A step annotated with a routing condition.

    Created by :meth:`~step.Step.when`; consumed by
    :meth:`~step.Step.__rshift__` to produce a :class:`ConditionalEdge`.

    Attributes:
        step: The destination step.
        condition: Predicate applied to the source step's terminal value, or
            :attr:`Cond.default` for the catch-all fallback.
    """

    step: Step[GraphStateT, AgentStateT, RequestT, EventT, TerminalT]
    condition: Callable[[Any], bool] | _DefaultCondition


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DirectEdge(Generic[GraphStateT, AgentStateT, RequestT, EventT, TerminalT, NextTerminalT]):
    """Unconditional transition between two steps.

    Created by ``step_a >> step_b``.  A :class:`DirectEdge` always wins over
    any :class:`ConditionalEdge` originating from the same source step.

    Attributes:
        src: The step from which this edge originates.
        dst: The step to transition to unconditionally.
    """

    src: Step[GraphStateT, AgentStateT, RequestT, EventT, TerminalT]
    dst: Step[GraphStateT, Any, TerminalT, EventT, NextTerminalT]


@dataclasses.dataclass(frozen=True)
class ConditionalEdge(Generic[GraphStateT, AgentStateT, RequestT, EventT, TerminalT, NextTerminalT]):
    """Conditional transition between two steps.

    Created by ``step_a >> step_b.when(condition)``.  The graph evaluates
    ``condition(terminal)`` where *terminal* is the output of ``src``; if it
    returns ``True`` (or ``condition`` is :attr:`Cond.default` and no earlier
    condition matched), the graph transitions to ``dst``.

    Attributes:
        src: The step from which this edge originates.
        dst: The step to transition to when the condition holds.
        condition: Predicate on the source terminal, or :attr:`Cond.default`.
    """

    src: Step[GraphStateT, AgentStateT, RequestT, EventT, TerminalT]
    dst: Step[GraphStateT, Any, TerminalT, EventT, NextTerminalT]
    condition: Callable[[TerminalT], bool] | _DefaultCondition


Edge = DirectEdge[Any, Any, Any, Any, Any, Any] | ConditionalEdge[Any, Any, Any, Any, Any, Any]
"""Union of :class:`DirectEdge` and :class:`ConditionalEdge`.

Pass values of this type to :class:`~graph.Graph` to declare the routing
between steps.
"""

__all__ = [
    "Cond",
    "ConditionalEdge",
    "ConditionalStep",
    "DirectEdge",
    "Edge",
]
