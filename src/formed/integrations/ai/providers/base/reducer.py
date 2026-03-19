from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from .events import (
    StreamEvent,
    TextPartDone,
    ThinkingPartDone,
    ToolCallPartDone,
    TurnDone,
)
from .messages import ThinkingBlock, ToolCallRecord
from .signals import Signal, TextOutput


@dataclasses.dataclass(frozen=True)
class ReducerState:
    """Minimal state for :class:`AgentReducer`.

    With the part-lifecycle event model, all text / argument accumulation is
    done in the Engine layer before emitting ``*PartDone`` events.  The
    reducer only needs to track thinking blocks that should be round-tripped
    back to the provider on the next turn.

    Attributes:
        thinking_blocks: Thinking blocks accumulated in this turn, collected
            for round-tripping to the handler.
    """

    thinking_blocks: tuple[ThinkingBlock, ...] = ()


class AgentReducer:
    """Provider-agnostic reducer that maps part-lifecycle events to signals.

    :class:`TextPartDone` and :class:`ToolCallPartDone` events already carry
    fully assembled data, so the reducer simply re-packages them as
    :class:`~signals.Signal` values.  :class:`ThinkingPartDone` events with a
    non-``None`` ``opaque`` block are accumulated in ``ReducerState`` for
    round-tripping.  All other events pass through without emitting signals.

    Examples:
        >>> from formed.integrations.ai.providers.base import AgentReducer, ReducerState
        >>> from formed.integrations.ai.providers.base.events import (
        ...     TextPartStarted, TextDelta, TextPartDone, TurnDone,
        ... )
        >>> reducer = AgentReducer()
        >>> state = ReducerState()
        >>> state, sigs = reducer(state, TextPartStarted(index=0))
        >>> state, sigs = reducer(state, TextDelta(index=0, delta="Hello"))
        >>> state, sigs = reducer(state, TextPartDone(index=0, text="Hello"))
        >>> sigs  # [TextOutput(text="Hello")]
        >>> state, sigs = reducer(state, TurnDone(finish_reason="stop"))
    """

    def __call__(
        self,
        state: ReducerState,
        event: StreamEvent,
    ) -> tuple[ReducerState, Sequence[Signal]]:
        """Fold one event into state and return any newly completed signals.

        Args:
            state: Current reducer state.
            event: Incoming :class:`~events.StreamEvent` from the engine.

        Returns:
            A tuple of ``(new_state, signals)`` where ``signals`` is non-empty
            only when a ``*PartDone`` event has completed a meaningful unit.
        """
        if isinstance(event, TextPartDone):
            return state, [TextOutput(text=event.text)]

        if isinstance(event, ToolCallPartDone):
            return state, [
                ToolCallRecord(
                    id=event.tool_call_id,
                    name=event.tool_name,
                    args_json=event.args_json,
                )
            ]

        if isinstance(event, ThinkingPartDone) and event.opaque is not None:
            return (
                dataclasses.replace(
                    state,
                    thinking_blocks=(*state.thinking_blocks, event.opaque),
                ),
                [],
            )

        if isinstance(event, TurnDone):
            # Reset the thinking buffer at the end of a turn so the caller
            # can read the accumulated blocks from state before they're cleared.
            return dataclasses.replace(state, thinking_blocks=()), []

        # TextPartStarted, TextDelta, ToolCallPartStarted, ToolCallArgsDelta,
        # ThinkingPartStarted, ThinkingDelta — all pass through without signals.
        return state, []
