from typing import Literal, NotRequired, TypedDict

from typing_extensions import TypeAlias

from formed.types import JsonValue

# ---------------------------------------------------------------------------
# Text part lifecycle
# ---------------------------------------------------------------------------


class TextPartStarted(TypedDict):
    kind: Literal["text_part_started"]
    index: int


class TextDelta(TypedDict):
    kind: Literal["text_delta"]
    index: int
    delta: str


class TextPartDone(TypedDict):
    kind: Literal["text_part_done"]
    index: int
    text: str


# ---------------------------------------------------------------------------
# Tool call part lifecycle
# ---------------------------------------------------------------------------


class ToolCallPartStarted(TypedDict):
    kind: Literal["tool_call_part_started"]
    index: int
    tool_call_id: str
    tool_name: str


class ToolCallArgsDelta(TypedDict):
    kind: Literal["tool_call_args_delta"]
    index: int
    delta: str


class ToolCallPartDone(TypedDict):
    kind: Literal["tool_call_part_done"]
    index: int
    tool_call_id: str
    tool_name: str
    args: JsonValue


# ---------------------------------------------------------------------------
# Thinking part lifecycle
# ---------------------------------------------------------------------------


class ThinkingPartStarted(TypedDict):
    kind: Literal["thinking_part_started"]
    index: int


class ThinkingDelta(TypedDict):
    kind: Literal["thinking_delta"]
    index: int
    delta: str


class ThinkingPartDone(TypedDict):
    kind: Literal["thinking_part_done"]
    index: int
    thoughts: str
    metadata: NotRequired[JsonValue]


# ---------------------------------------------------------------------------
# Turn termination
# ---------------------------------------------------------------------------


FinishReason: TypeAlias = Literal["stop", "tool_calls", "length", "content_filter"]


class Usage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int


class TurnDone(TypedDict):
    kind: Literal["turn_done"]
    finish_reason: FinishReason
    usage: Usage


# ---------------------------------------------------------------------------
# Union alias
# ---------------------------------------------------------------------------


StreamEvent: TypeAlias = (
    TextPartStarted
    | TextDelta
    | TextPartDone
    | ToolCallPartStarted
    | ToolCallArgsDelta
    | ToolCallPartDone
    | ThinkingPartStarted
    | ThinkingDelta
    | ThinkingPartDone
    | TurnDone
)
