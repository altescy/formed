from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from ..base.events import (
    StreamEvent,
    TextDelta,
    TextPartDone,
    TextPartStarted,
    ToolCallArgsDelta,
    ToolCallPartDone,
    ToolCallPartStarted,
    TurnDone,
    Usage,
)
from ..base.messages import (
    AssistantMessage,
    ImageBytesContent,
    ImageUrlContent,
    Query,
    TextContent,
    ToolResultMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class OpenAIEngine:
    """Engine that wraps the OpenAI Chat Completions streaming API.

    Converts the provider's streaming response into a :data:`~events.StreamEvent`
    stream following the part-lifecycle model
    (``*PartStarted`` → ``*Delta`` → ``*PartDone`` → :class:`~events.TurnDone`).

    Requires the ``openai`` package.

    Args:
        client: Authenticated :class:`openai.AsyncOpenAI` client.
        model: Model identifier (e.g. ``"gpt-4o"``).
        **kwargs: Additional keyword arguments forwarded to
            ``client.chat.completions.stream``.

    Examples:
        >>> from openai import AsyncOpenAI
        >>> from formed.integrations.ai.providers.openai import OpenAIEngine
        >>>
        >>> engine = OpenAIEngine(client=AsyncOpenAI(), model="gpt-4o")
        >>> async with engine(query) as stream:
        ...     async for event in stream:
        ...         print(event)
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        **kwargs: Any,
    ) -> None:
        self._client = client
        self._model = model
        self._kwargs = kwargs

    @asynccontextmanager
    async def __call__(self, query: Query) -> AsyncIterator[AsyncIterator[StreamEvent]]:
        messages = _build_messages(query)
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream_options": {"include_usage": True},
            **self._kwargs,
        }
        if query.system:
            # system is already injected as first message by _build_messages;
            # no extra param needed for OpenAI.
            pass
        if query.tools:
            params["tools"] = [_tool_to_openai(t) for t in query.tools]

        async def _stream() -> AsyncIterator[StreamEvent]:
            # Per-part accumulators
            text_buf: str = ""
            text_part_open: bool = False
            text_part_index: int = 0

            @dataclasses.dataclass
            class _PartialTC:
                tool_call_id: str
                tool_name: str
                args_buf: str

            tc_parts: dict[int, _PartialTC] = {}
            tc_part_index_base: int = 1  # text part uses index 0

            finish_reason: str | None = None
            usage_data: dict[str, Any] | None = None

            async with await self._client.chat.completions.create(stream=True, **params) as stream:
                async for chunk in stream:
                    choice = chunk.choices[0] if chunk.choices else None

                    # Capture usage when provided (some models send it on the
                    # last chunk alongside an empty choices list).
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        u = chunk.usage
                        usage_data = {
                            "input_tokens": getattr(u, "prompt_tokens", None),
                            "output_tokens": getattr(u, "completion_tokens", None),
                        }

                    if choice is None:
                        continue

                    delta = choice.delta

                    # --- text ---
                    if delta.content:
                        if not text_part_open:
                            text_part_open = True
                            yield TextPartStarted(index=text_part_index)
                        text_buf += delta.content
                        yield TextDelta(index=text_part_index, delta=delta.content)

                    # --- tool calls ---
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            part_index = tc_part_index_base + idx

                            if idx not in tc_parts:
                                tc_id = tc.id or ""
                                tc_name = (tc.function.name if tc.function else None) or ""
                                tc_parts[idx] = _PartialTC(
                                    tool_call_id=tc_id,
                                    tool_name=tc_name,
                                    args_buf="",
                                )
                                yield ToolCallPartStarted(
                                    index=part_index,
                                    tool_call_id=tc_id,
                                    tool_name=tc_name,
                                )
                            else:
                                part = tc_parts[idx]
                                if tc.id and not part.tool_call_id:
                                    part.tool_call_id = tc.id
                                if tc.function and tc.function.name and not part.tool_name:
                                    part.tool_name = tc.function.name

                            args_frag = tc.function.arguments if tc.function and tc.function.arguments else ""
                            if args_frag:
                                tc_parts[idx].args_buf += args_frag
                                yield ToolCallArgsDelta(index=part_index, delta=args_frag)

                    if choice.finish_reason is not None:
                        finish_reason = choice.finish_reason

            # Flush open parts
            if text_part_open:
                yield TextPartDone(index=text_part_index, text=text_buf)

            for idx, part in tc_parts.items():
                part_index = tc_part_index_base + idx
                yield ToolCallPartDone(
                    index=part_index,
                    tool_call_id=part.tool_call_id,
                    tool_name=part.tool_name,
                    args_json=part.args_buf,
                )

            # Normalise finish reason
            fr = finish_reason or "stop"
            if fr not in ("stop", "tool_calls", "length", "content_filter"):
                fr = "stop"

            usage: Usage | None = None
            if usage_data:
                usage = Usage(
                    input_tokens=usage_data.get("input_tokens"),
                    output_tokens=usage_data.get("output_tokens"),
                )

            yield TurnDone(finish_reason=fr, usage=usage)  # type: ignore[arg-type]

        yield _stream()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_messages(query: Query) -> list[dict[str, Any]]:
    """Convert a :class:`~messages.Query` into the OpenAI API message list format.

    Args:
        query: Conversation context to convert.

    Returns:
        A list of message dicts accepted by ``client.chat.completions.stream``.
    """
    result: list[dict[str, Any]] = []

    if query.system:
        result.append({"role": "system", "content": query.system})

    for msg in query.history:
        if isinstance(msg, UserMessage):
            result.append({"role": "user", "content": _build_content(msg.parts)})

        elif isinstance(msg, AssistantMessage):
            entry: dict[str, Any] = {"role": "assistant"}
            if msg.parts:
                entry["content"] = "".join(p.text for p in msg.parts if isinstance(p, TextContent))
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.args_json},
                    }
                    for tc in msg.tool_calls
                ]
            result.append(entry)

        elif isinstance(msg, ToolResultMessage):
            # OpenAI expects a single string for tool results; join text parts.
            text = "".join(p.text for p in msg.parts if isinstance(p, TextContent))
            result.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": text})

    return result


def _build_content(parts: tuple[Any, ...]) -> list[dict[str, Any]] | str:
    """Convert content parts to OpenAI's ``content`` format.

    For a single plain-text part, returns a plain string (simpler).
    For mixed/image content, returns the list-of-dicts format.

    Args:
        parts: Sequence of :data:`~messages.ContentPart` items.

    Returns:
        Either a plain string or a list of OpenAI content-part dicts.
    """
    import base64

    if len(parts) == 1 and isinstance(parts[0], TextContent):
        return parts[0].text

    result: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextContent):
            result.append({"type": "text", "text": part.text})
        elif isinstance(part, ImageUrlContent):
            image_url: dict[str, Any] = {"url": part.url}
            if part.detail is not None:
                image_url["detail"] = part.detail
            result.append({"type": "image_url", "image_url": image_url})
        elif isinstance(part, ImageBytesContent):
            b64 = base64.b64encode(part.data).decode()
            data_url = f"data:{part.media_type};base64,{b64}"
            image_url = {"url": data_url}
            if part.detail is not None:
                image_url["detail"] = part.detail
            result.append({"type": "image_url", "image_url": image_url})
    return result


def _tool_to_openai(tool: Any) -> dict[str, Any]:
    """Convert a :class:`~tools.ToolDefinition` to OpenAI's tool param format.

    Args:
        tool: Tool definition to convert.

    Returns:
        Dict in OpenAI ``ChatCompletionToolParam`` shape.
    """
    func: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict is not None:
        func["strict"] = tool.strict
    return {"type": "function", "function": func}
