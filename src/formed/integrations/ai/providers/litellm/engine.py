from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from ..base.events import (
    StreamEvent,
    TextDelta,
    TextPartDone,
    TextPartStarted,
    ThinkingDelta,
    ThinkingPartDone,
    ThinkingPartStarted,
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
    ThinkingBlock,
    ToolResultMessage,
    UserMessage,
)


class LiteLLMEngine:
    """Engine that wraps the LiteLLM unified completion interface.

    LiteLLM exposes an OpenAI-compatible API and supports many providers
    (Anthropic, Gemini, Azure OpenAI, Cohere, etc.) through a single
    interface.  This engine converts the streaming response into a
    :data:`~events.StreamEvent` stream following the part-lifecycle model.

    For providers that surface thinking / reasoning content (e.g. Anthropic
    extended thinking via ``thinking_blocks`` in the delta), the engine emits
    :class:`~events.ThinkingPartStarted`, :class:`~events.ThinkingDelta`, and
    :class:`~events.ThinkingPartDone` with an opaque
    :class:`~messages.ThinkingBlock` for round-tripping.

    Requires the ``litellm`` package.

    Args:
        model: LiteLLM model string (e.g.
            ``"anthropic/claude-3-5-sonnet-20241022"``).
        **kwargs: Additional keyword arguments forwarded to
            ``litellm.acompletion``.

    Examples:
        >>> from formed.integrations.ai.providers.litellm import LiteLLMEngine
        >>>
        >>> engine = LiteLLMEngine(model="anthropic/claude-3-5-sonnet-20241022")
        >>> async with engine(query) as stream:
        ...     async for event in stream:
        ...         print(event)
    """

    def __init__(
        self,
        model: str,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._kwargs = kwargs

    @asynccontextmanager
    async def __call__(self, query: Query) -> AsyncIterator[AsyncIterator[StreamEvent]]:
        try:
            import litellm
        except ImportError as e:
            raise ImportError("litellm is required to use LiteLLMEngine. Install it with: pip install litellm") from e

        messages = _build_messages(query)
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            **self._kwargs,
        }
        if query.tools:
            params["tools"] = [_tool_to_litellm(t) for t in query.tools]

        async def _stream() -> AsyncIterator[StreamEvent]:
            @dataclasses.dataclass
            class _PartialTC:
                tool_call_id: str
                tool_name: str
                args_buf: str

            # Part state
            text_buf: str = ""
            text_part_open: bool = False
            text_part_index: int = 0

            thinking_buf: str = ""
            thinking_sig: str = ""
            thinking_part_open: bool = False
            thinking_part_index: int = -1  # will be assigned when opened

            tc_parts: dict[int, _PartialTC] = {}
            tc_part_index_base: int = 1  # thinking gets 0 if present, else text gets 0

            finish_reason: str | None = None
            usage_data: dict[str, Any] | None = None

            response = cast(AsyncIterator[Any], await litellm.acompletion(**params))
            async for chunk in response:
                # Usage (some providers send it in the last chunk)
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    u = chunk.usage
                    usage_data = {
                        "input_tokens": getattr(u, "prompt_tokens", None),
                        "output_tokens": getattr(u, "completion_tokens", None),
                    }

                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue

                delta = choice.delta

                # --- thinking (Anthropic extended thinking via LiteLLM) ---
                thinking_text: str | None = getattr(delta, "thinking", None) or getattr(
                    delta, "reasoning_content", None
                )
                thinking_signature: str | None = getattr(delta, "signature", None)

                if thinking_text:
                    if not thinking_part_open:
                        thinking_part_open = True
                        thinking_part_index = 0
                        # Push text part index up
                        text_part_index = 1
                        tc_part_index_base = 2
                        yield ThinkingPartStarted(index=thinking_part_index)
                    thinking_buf += thinking_text
                    yield ThinkingDelta(index=thinking_part_index, delta=thinking_text)

                if thinking_signature:
                    thinking_sig = thinking_signature

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
                            tc_id = getattr(tc, "id", None) or ""
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
                            if tc.id and not part.tool_call_id:  # type: ignore[attr-defined]
                                part.tool_call_id = tc.id  # type: ignore[attr-defined]
                            if tc.function and tc.function.name and not part.tool_name:
                                part.tool_name = tc.function.name

                        args_frag = tc.function.arguments if tc.function and tc.function.arguments else ""
                        if args_frag:
                            tc_parts[idx].args_buf += args_frag
                            yield ToolCallArgsDelta(index=part_index, delta=args_frag)

                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason

            # Flush open parts
            if thinking_part_open:
                opaque: ThinkingBlock | None = None
                if thinking_buf or thinking_sig:
                    import json

                    opaque = ThinkingBlock(
                        provider="litellm",
                        data=json.dumps({"text": thinking_buf, "signature": thinking_sig}).encode(),
                    )
                yield ThinkingPartDone(
                    index=thinking_part_index,
                    text=thinking_buf,
                    opaque=opaque,
                )

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
    """Convert a :class:`~messages.Query` into the LiteLLM message list format.

    Args:
        query: Conversation context to convert.

    Returns:
        A list of message dicts accepted by ``litellm.acompletion``.
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
            text = "".join(p.text for p in msg.parts if isinstance(p, TextContent))
            result.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": text})

    return result


def _build_content(parts: tuple[Any, ...]) -> list[dict[str, Any]] | str:
    """Convert content parts to LiteLLM's ``content`` format.

    Args:
        parts: Sequence of :data:`~messages.ContentPart` items.

    Returns:
        Either a plain string (single text part) or a list of content-part dicts.
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


def _tool_to_litellm(tool: Any) -> dict[str, Any]:
    """Convert a :class:`~tools.ToolDefinition` to LiteLLM's tool param format.

    Args:
        tool: Tool definition to convert.

    Returns:
        Dict in OpenAI-compatible ``tool`` shape.
    """
    func: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict is not None:
        func["strict"] = tool.strict
    return {"type": "function", "function": func}
