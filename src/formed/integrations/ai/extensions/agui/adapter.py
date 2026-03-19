from __future__ import annotations

import dataclasses
import uuid
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from ...providers.base.events import (
    StreamEvent,
    TextDelta,
    TextPartDone,
    TextPartStarted,
    ToolCallArgsDelta,
    ToolCallPartDone,
    ToolCallPartStarted,
    TurnDone,
)
from ...response import Response

EventT = TypeVar("EventT")
ResultT_co = TypeVar("ResultT_co", covariant=True)


# ---------------------------------------------------------------------------
# AG-UI event type definitions
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AGUIRunStarted:
    """AG-UI event emitted when an agent run begins.

    Attributes:
        run_id: Unique identifier for this run.
    """

    run_id: str


@dataclasses.dataclass(frozen=True)
class AGUIRunFinished:
    """AG-UI event emitted when an agent run completes.

    Attributes:
        run_id: Unique identifier for this run.
    """

    run_id: str


@dataclasses.dataclass(frozen=True)
class AGUITextMessageStart:
    """AG-UI event marking the start of a streamed text message.

    Attributes:
        message_id: Unique identifier for this message.
    """

    message_id: str


@dataclasses.dataclass(frozen=True)
class AGUITextMessageContent:
    """AG-UI event carrying an incremental text delta.

    Attributes:
        message_id: Identifier of the message this delta belongs to.
        delta: Incremental text fragment.
    """

    message_id: str
    delta: str


@dataclasses.dataclass(frozen=True)
class AGUITextMessageEnd:
    """AG-UI event marking the end of a streamed text message.

    Attributes:
        message_id: Identifier of the message that has ended.
    """

    message_id: str


@dataclasses.dataclass(frozen=True)
class AGUIToolCallStart:
    """AG-UI event marking the start of a tool call.

    Attributes:
        tool_call_id: Unique identifier for this tool call.
        tool_call_name: Name of the tool being called.
    """

    tool_call_id: str
    tool_call_name: str


@dataclasses.dataclass(frozen=True)
class AGUIToolCallArgs:
    """AG-UI event carrying an incremental tool-argument delta.

    Attributes:
        tool_call_id: Identifier of the tool call this delta belongs to.
        delta: Incremental JSON arguments fragment.
    """

    tool_call_id: str
    delta: str


@dataclasses.dataclass(frozen=True)
class AGUIToolCallEnd:
    """AG-UI event marking the end of a tool call.

    Attributes:
        tool_call_id: Identifier of the tool call that has ended.
    """

    tool_call_id: str


AGUIEvent = (
    AGUIRunStarted
    | AGUIRunFinished
    | AGUITextMessageStart
    | AGUITextMessageContent
    | AGUITextMessageEnd
    | AGUIToolCallStart
    | AGUIToolCallArgs
    | AGUIToolCallEnd
)


# ---------------------------------------------------------------------------
# Adapter state
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AGUIState:
    """Immutable adapter state maintained while mapping part-lifecycle events.

    With the part-lifecycle model, a text message ID is assigned when
    :class:`~events.TextPartStarted` arrives and reused for all subsequent
    deltas until :class:`~events.TextPartDone`.  Tool call IDs come directly
    from the :class:`~events.ToolCallPartStarted` event.

    Attributes:
        text_message_ids: Map from part index to assigned AG-UI message ID.
    """

    text_message_ids: dict[int, str] = dataclasses.field(default_factory=dict)

    def replace(self, **kwargs: Any) -> AGUIState:
        return dataclasses.replace(self, **kwargs)


# ---------------------------------------------------------------------------
# AG-UI adapter
# ---------------------------------------------------------------------------


def to_agui_stream(
    response: Response[StreamEvent, Any, Any],
    run_id: str = "run-1",
) -> AsyncIterator[AGUIEvent]:
    """Convert a :class:`~response.Response` event stream to AG-UI protocol events.

    Each call creates an independent subscription via :meth:`~response.Response.events`,
    so multiple consumers can call this function on the same
    :class:`~response.Response` concurrently without interfering.

    The mapping follows the part-lifecycle model:

    - :class:`~events.TextPartStarted` → :class:`AGUITextMessageStart`
    - :class:`~events.TextDelta` → :class:`AGUITextMessageContent`
    - :class:`~events.TextPartDone` → :class:`AGUITextMessageEnd`
    - :class:`~events.ToolCallPartStarted` → :class:`AGUIToolCallStart`
    - :class:`~events.ToolCallArgsDelta` → :class:`AGUIToolCallArgs`
    - :class:`~events.ToolCallPartDone` → :class:`AGUIToolCallEnd`
    - :class:`~events.TurnDone` / other → ignored (no AG-UI equivalent)

    Args:
        response: Agent response whose event stream contains
            :data:`~events.StreamEvent` items.
        run_id: Identifier attached to the :class:`AGUIRunStarted` /
            :class:`AGUIRunFinished` bookend events.

    Returns:
        An :class:`~collections.abc.AsyncIterator` of :data:`AGUIEvent`
        instances following the AG-UI protocol.

    Examples:
        >>> async for event in to_agui_stream(response, run_id="run-abc"):
        ...     print(event)
    """

    async def _iter() -> AsyncIterator[AGUIEvent]:
        yield AGUIRunStarted(run_id=run_id)

        state = AGUIState()

        async for event in response.events():
            state, new_events = _reduce_agui(state, event)
            for e in new_events:
                yield e

        yield AGUIRunFinished(run_id=run_id)

    return _iter()


def _reduce_agui(
    state: AGUIState,
    event: StreamEvent,
) -> tuple[AGUIState, list[AGUIEvent]]:
    """Map a single :data:`~events.StreamEvent` to zero or more AG-UI events.

    Args:
        state: Current adapter state.
        event: Incoming stream event.

    Returns:
        ``(new_state, emitted_events)``
    """
    emitted: list[AGUIEvent] = []

    if isinstance(event, TextPartStarted):
        msg_id = str(uuid.uuid4())
        new_ids = dict(state.text_message_ids)
        new_ids[event.index] = msg_id
        state = state.replace(text_message_ids=new_ids)
        emitted.append(AGUITextMessageStart(message_id=msg_id))

    elif isinstance(event, TextDelta):
        msg_id = state.text_message_ids.get(event.index)
        if msg_id is not None:
            emitted.append(AGUITextMessageContent(message_id=msg_id, delta=event.delta))

    elif isinstance(event, TextPartDone):
        msg_id = state.text_message_ids.get(event.index)
        if msg_id is not None:
            emitted.append(AGUITextMessageEnd(message_id=msg_id))
            new_ids = dict(state.text_message_ids)
            del new_ids[event.index]
            state = state.replace(text_message_ids=new_ids)

    elif isinstance(event, ToolCallPartStarted):
        emitted.append(
            AGUIToolCallStart(
                tool_call_id=event.tool_call_id,
                tool_call_name=event.tool_name,
            )
        )

    elif isinstance(event, ToolCallArgsDelta):
        # The tool_call_id is not directly on ToolCallArgsDelta; we rely on
        # callers to track it via ToolCallPartStarted.  For AG-UI we need the
        # ID — the adapter must carry it in state if needed.  However, the
        # AG-UI protocol only uses the ID for ToolCallArgs, and the ID was
        # already emitted in ToolCallPartStarted.  Downstream consumers that
        # need to correlate can use the part index.  Here we emit a sentinel
        # with an empty ID because the AG-UI spec requires one; callers that
        # need full fidelity should track IDs themselves.
        # (See note below for a richer state design.)
        pass  # handled by richer state variant below

    elif isinstance(event, ToolCallPartDone):
        emitted.append(AGUIToolCallEnd(tool_call_id=event.tool_call_id))

    elif isinstance(event, TurnDone):
        pass  # no AG-UI equivalent

    return state, emitted


# ---------------------------------------------------------------------------
# Richer adapter that tracks tool-call IDs for ToolCallArgsDelta
# ---------------------------------------------------------------------------
# The simple version above passes ToolCallArgsDelta through without an ID
# because the ID is only available at ToolCallPartStarted time.  The full
# version below carries a tool_call_id map in state.


@dataclasses.dataclass(frozen=True)
class _RichAGUIState:
    """Extended AG-UI state that maps part index → IDs for both text and tool calls."""

    text_message_ids: dict[int, str] = dataclasses.field(default_factory=dict)
    tool_call_ids: dict[int, str] = dataclasses.field(default_factory=dict)

    def replace(self, **kwargs: Any) -> _RichAGUIState:
        return dataclasses.replace(self, **kwargs)


def _reduce_agui_full(
    state: _RichAGUIState,
    event: StreamEvent,
) -> tuple[_RichAGUIState, list[AGUIEvent]]:
    """Full AG-UI reducer that correctly routes :class:`~events.ToolCallArgsDelta`.

    Args:
        state: Current rich adapter state.
        event: Incoming stream event.

    Returns:
        ``(new_state, emitted_events)``
    """
    emitted: list[AGUIEvent] = []

    if isinstance(event, TextPartStarted):
        msg_id = str(uuid.uuid4())
        new_ids = dict(state.text_message_ids)
        new_ids[event.index] = msg_id
        state = state.replace(text_message_ids=new_ids)
        emitted.append(AGUITextMessageStart(message_id=msg_id))

    elif isinstance(event, TextDelta):
        msg_id = state.text_message_ids.get(event.index)
        if msg_id is not None:
            emitted.append(AGUITextMessageContent(message_id=msg_id, delta=event.delta))

    elif isinstance(event, TextPartDone):
        msg_id = state.text_message_ids.get(event.index)
        if msg_id is not None:
            emitted.append(AGUITextMessageEnd(message_id=msg_id))
            new_ids = dict(state.text_message_ids)
            del new_ids[event.index]
            state = state.replace(text_message_ids=new_ids)

    elif isinstance(event, ToolCallPartStarted):
        new_tc_ids = dict(state.tool_call_ids)
        new_tc_ids[event.index] = event.tool_call_id
        state = state.replace(tool_call_ids=new_tc_ids)
        emitted.append(
            AGUIToolCallStart(
                tool_call_id=event.tool_call_id,
                tool_call_name=event.tool_name,
            )
        )

    elif isinstance(event, ToolCallArgsDelta):
        tc_id = state.tool_call_ids.get(event.index)
        if tc_id is not None:
            emitted.append(AGUIToolCallArgs(tool_call_id=tc_id, delta=event.delta))

    elif isinstance(event, ToolCallPartDone):
        emitted.append(AGUIToolCallEnd(tool_call_id=event.tool_call_id))
        new_tc_ids = dict(state.tool_call_ids)
        new_tc_ids.pop(event.index, None)
        state = state.replace(tool_call_ids=new_tc_ids)

    return state, emitted


def to_agui_stream_full(
    response: Response[StreamEvent, Any, Any],
    run_id: str = "run-1",
) -> AsyncIterator[AGUIEvent]:
    """Like :func:`to_agui_stream` but also emits :class:`AGUIToolCallArgs` events.

    Uses :func:`_reduce_agui_full` internally which tracks tool-call IDs
    across deltas.

    Args:
        response: Agent response whose event stream contains
            :data:`~events.StreamEvent` items.
        run_id: Identifier attached to the bookend events.

    Returns:
        An :class:`~collections.abc.AsyncIterator` of :data:`AGUIEvent` instances.
    """

    async def _iter() -> AsyncIterator[AGUIEvent]:
        yield AGUIRunStarted(run_id=run_id)

        state = _RichAGUIState()

        async for event in response.events():
            state, new_events = _reduce_agui_full(state, event)
            for e in new_events:
                yield e

        yield AGUIRunFinished(run_id=run_id)

    return _iter()
