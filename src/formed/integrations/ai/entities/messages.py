from collections.abc import Sequence
from typing import Literal, NotRequired, TypedDict

from typing_extensions import TypeAlias

from formed.types import JsonValue

# ---------------------------------------------------------------------------
# Content parts
# ---------------------------------------------------------------------------


class TextContent(TypedDict):
    kind: Literal["text"]
    text: str


ImageDetail: TypeAlias = Literal["auto", "low", "high"]


class ImageUrlContent(TypedDict):
    kind: Literal["image_url"]
    url: str
    detail: NotRequired[ImageDetail]


class ImageBytesContent(TypedDict):
    kind: Literal["image_bytes"]
    data: bytes
    media_type: str
    detail: NotRequired[ImageDetail]


ContentPart: TypeAlias = TextContent | ImageUrlContent | ImageBytesContent

# ---------------------------------------------------------------------------
# Thinking block (opaque round-trip payload)
# ---------------------------------------------------------------------------


class ThinkingBlock(TypedDict):
    kind: Literal["thinking"]
    thoughts: str
    metadata: NotRequired[JsonValue]


# ---------------------------------------------------------------------------
# Tool call record
# ---------------------------------------------------------------------------


class ToolCallRecord(TypedDict):
    id: str
    name: str
    args: JsonValue


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


class UserMessage(TypedDict):
    role: Literal["user"]
    parts: Sequence[ContentPart]


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    parts: Sequence[ContentPart]
    tool_calls: NotRequired[Sequence[ToolCallRecord]]
    thinking: NotRequired[Sequence[ThinkingBlock]]


class ToolResultMessage(TypedDict):
    role: Literal["tool_result"]
    tool_call_id: str
    parts: Sequence[ContentPart]


ChatMessage: TypeAlias = UserMessage | AssistantMessage | ToolResultMessage
