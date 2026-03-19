"""Graph-level lifecycle events emitted during graph execution.

Each :class:`~step.Step` execution emits three events in order:

1. :class:`StepStarted` — before the agent runs.
2. :class:`StepEvent` — once per event from the inner agent stream.
3. :class:`StepFinished` — after the agent produces a terminal value.

The union alias :data:`GraphEvent` covers all three so that typed streams
can be annotated as ``AsyncIterator[GraphEvent]``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Generic, TypeVar

EventT_co = TypeVar("EventT_co", covariant=True)
RequestT = TypeVar("RequestT")
TerminalT = TypeVar("TerminalT")


@dataclasses.dataclass(frozen=True)
class StepStarted(Generic[RequestT]):
    """Published immediately before a step begins executing its agent.

    Attributes:
        step_name: Name of the step that is starting.
        input: Request value forwarded to the step's agent as its first input.
    """

    step_name: str
    input: RequestT


@dataclasses.dataclass(frozen=True)
class StepEvent(Generic[EventT_co]):
    """Wraps a single event emitted by the inner agent during a step.

    One :class:`StepEvent` is published for every event the agent emits so
    that outer consumers can observe the raw stream of each step without
    subscribing to each agent directly.

    Attributes:
        step_name: Name of the step whose agent produced ``event``.
        event: The raw event from the inner agent.
    """

    step_name: str
    event: EventT_co


@dataclasses.dataclass(frozen=True)
class StepFinished(Generic[TerminalT]):
    """Published after a step's agent returns its terminal value.

    Attributes:
        step_name: Name of the step that finished.
        output: Terminal value produced by the step's agent.
    """

    step_name: str
    output: TerminalT


GraphEvent = StepStarted[Any] | StepEvent[Any] | StepFinished[Any]
"""Union of all graph lifecycle events.

Used as the ``EventT`` parameter of :class:`~graph.GraphResponse` and
:class:`~graph.Graph` when the caller wants an unfiltered event stream.
"""

__all__ = [
    "GraphEvent",
    "StepEvent",
    "StepFinished",
    "StepStarted",
]
