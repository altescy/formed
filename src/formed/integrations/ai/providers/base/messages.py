from __future__ import annotations

import dataclasses
from typing import Literal, Union

from .tools import ToolDefinition

# ---------------------------------------------------------------------------
# Content parts
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TextContent:
    """Plain text content part.

    Attributes:
        text: Text string.
    """

    text: str


@dataclasses.dataclass(frozen=True)
class ImageUrlContent:
    """Image referenced by URL.

    Attributes:
        url: HTTP(S) URL of the image.
        detail: Resolution hint passed to the provider.
            ``None`` means provider default (typically ``"auto"``).
    """

    url: str
    detail: Literal["auto", "low", "high"] | None = None


@dataclasses.dataclass(frozen=True)
class ImageBytesContent:
    """Image supplied as raw bytes.

    Attributes:
        data: Raw image bytes.
        media_type: MIME type (e.g. ``"image/png"``).
        detail: Resolution hint. ``None`` means provider default.
    """

    data: bytes
    media_type: str
    detail: Literal["auto", "low", "high"] | None = None


ContentPart = Union[TextContent, ImageUrlContent, ImageBytesContent]

# ---------------------------------------------------------------------------
# Thinking block (opaque round-trip payload)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ThinkingBlock:
    """Opaque thinking block for provider round-trip.

    The content is intentionally unstructured — ``data`` holds whatever bytes
    the originating provider requires (e.g. Anthropic's thinking text +
    signature as JSON, or OpenAI's encrypted reasoning bytes).  The
    ``provider`` field ensures the block is only sent back to the provider
    that issued it.

    Attributes:
        provider: Provider identifier (e.g. ``"anthropic"``, ``"openai"``).
        data: Opaque bytes. The provider layer is responsible for serialising
            and deserialising this field.
    """

    provider: str
    data: bytes


# ---------------------------------------------------------------------------
# Tool call record
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ToolCallRecord:
    """Record of a single tool call made by the assistant.

    This is **distinct** from :class:`ToolResultMessage`, which carries the
    tool's output.  ``ToolCallRecord`` represents the call *issued* by the
    model; ``ToolResultMessage`` represents the *response* from the tool.

    Attributes:
        id: Unique identifier assigned by the model.
        name: Name of the tool that was called.
        args_json: JSON-encoded arguments.
    """

    id: str
    name: str
    args_json: str


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class UserMessage:
    """User-role message.

    Attributes:
        parts: Content parts forming this turn.  Use
            ``(TextContent(text),)`` for a plain-text turn.
    """

    parts: tuple[ContentPart, ...]


@dataclasses.dataclass(frozen=True)
class AssistantMessage:
    """Assistant-role message.

    Attributes:
        parts: Content parts in this turn (text and/or images).
        tool_calls: Tool call records issued in this turn.  Kept separate from
            ``parts`` so that :data:`ContentPart` stays limited to
            text/image content and the field is type-safe without casts.
        thinking: Opaque thinking blocks to round-trip back to the provider.
            Each block carries raw bytes that the provider can interpret; the
            structure is intentionally provider-agnostic.
    """

    parts: tuple[ContentPart, ...]
    tool_calls: tuple[ToolCallRecord, ...] = ()
    thinking: tuple[ThinkingBlock, ...] = ()


@dataclasses.dataclass(frozen=True)
class ToolResultMessage:
    """Tool-role message carrying the result of a tool execution.

    Attributes:
        tool_call_id: Identifier of the tool call this result belongs to.
        parts: Result content parts.  Typically a single :class:`TextContent`
            but can include images for vision-capable providers.
    """

    tool_call_id: str
    parts: tuple[ContentPart, ...]


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Query:
    """Complete context passed to an Engine for one generation turn.

    Separating ``system``, ``history``, and ``tools`` as first-class fields
    lets Engines map them to whatever provider-specific format they need
    (e.g. a ``system`` parameter separate from ``messages`` in Anthropic's
    API) without inspecting the message list.

    Attributes:
        system: System prompt. ``None`` means no system prompt.
        history: Ordered conversation history (user / assistant / tool turns).
        tools: Tool definitions available in this turn.
    """

    system: str | None = None
    history: tuple[UserMessage | AssistantMessage | ToolResultMessage, ...] = ()
    tools: tuple[ToolDefinition, ...] = ()
