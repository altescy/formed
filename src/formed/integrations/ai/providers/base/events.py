from __future__ import annotations

import dataclasses
from typing import Literal, Union

from .messages import ThinkingBlock

# ---------------------------------------------------------------------------
# Text part lifecycle
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TextPartStarted:
    """A new text generation part has begun.

    Attributes:
        index: Zero-based part index within this response turn.
    """

    index: int


@dataclasses.dataclass(frozen=True)
class TextDelta:
    """Incremental text fragment for the active text part.

    Attributes:
        index: Part index this delta belongs to.
        delta: Text fragment.
    """

    index: int
    delta: str


@dataclasses.dataclass(frozen=True)
class TextPartDone:
    """The active text part has completed.

    Attributes:
        index: Part index.
        text: Fully accumulated text of this part.
    """

    index: int
    text: str


# ---------------------------------------------------------------------------
# Tool call part lifecycle
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ToolCallPartStarted:
    """A new tool call part has begun.

    Attributes:
        index: Part index (distinct from any model-assigned call index).
        tool_call_id: Unique identifier assigned by the model for this call.
        tool_name: Name of the tool being called.
    """

    index: int
    tool_call_id: str
    tool_name: str


@dataclasses.dataclass(frozen=True)
class ToolCallArgsDelta:
    """Incremental JSON argument fragment for an in-progress tool call.

    Attributes:
        index: Part index.
        delta: Partial JSON string to be appended.
    """

    index: int
    delta: str


@dataclasses.dataclass(frozen=True)
class ToolCallPartDone:
    """A tool call part has completed.

    Attributes:
        index: Part index.
        tool_call_id: Model-assigned ID for this call.
        tool_name: Name of the tool.
        args_json: Fully assembled JSON arguments string.
    """

    index: int
    tool_call_id: str
    tool_name: str
    args_json: str


# ---------------------------------------------------------------------------
# Thinking part lifecycle
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ThinkingPartStarted:
    """A thinking (chain-of-thought) part has begun.

    Attributes:
        index: Part index.
    """

    index: int


@dataclasses.dataclass(frozen=True)
class ThinkingDelta:
    """Incremental thinking text fragment.

    Attributes:
        index: Part index.
        delta: Thinking text fragment.
    """

    index: int
    delta: str


@dataclasses.dataclass(frozen=True)
class ThinkingPartDone:
    """A thinking part has completed.

    The ``opaque`` field carries the provider-specific round-trip payload.
    Consumers that only need to display thinking text can use ``text``
    directly and ignore ``opaque``.

    Attributes:
        index: Part index.
        text: Fully accumulated thinking text.
        opaque: Opaque :class:`~messages.ThinkingBlock` for round-tripping.
            ``None`` for providers that do not require thinking to be sent
            back.
    """

    index: int
    text: str
    opaque: ThinkingBlock | None


# ---------------------------------------------------------------------------
# Turn termination
# ---------------------------------------------------------------------------

FinishReason = Literal["stop", "tool_calls", "length", "content_filter"]


@dataclasses.dataclass(frozen=True)
class Usage:
    """Token usage for one generation turn.

    All fields are optional; providers only populate the fields they report.

    Attributes:
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens generated.
        cache_read_tokens: Prompt tokens served from the provider cache.
        cache_write_tokens: Prompt tokens written to the provider cache.
        reasoning_tokens: Tokens spent on internal reasoning.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclasses.dataclass(frozen=True)
class TurnDone:
    """Sentinel marking the end of one model generation turn.

    Attributes:
        finish_reason: Normalized finish reason.
        usage: Token usage for this turn. ``None`` if the provider did not
            supply usage data.
    """

    finish_reason: FinishReason
    usage: Usage | None = None


# ---------------------------------------------------------------------------
# Union alias
# ---------------------------------------------------------------------------

StreamEvent = Union[
    TextPartStarted,
    TextDelta,
    TextPartDone,
    ToolCallPartStarted,
    ToolCallArgsDelta,
    ToolCallPartDone,
    ThinkingPartStarted,
    ThinkingDelta,
    ThinkingPartDone,
    TurnDone,
]
