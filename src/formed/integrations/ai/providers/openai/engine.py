import asyncio
import base64
import json
import os
import ssl
import urllib.parse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import assert_never

from ...engine import BaseChatEngine, ChatQuery
from ...entities.events import FinishReason, StreamEvent, Usage
from ...entities.messages import ChatMessage, ContentPart

# ---------------------------------------------------------------------------
# Generic async HTTP request/response utilities
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _http_stream(
    host: str,
    port: int,
    use_ssl: bool,
    request_bytes: bytes,
    timeout: float = 10.0,
) -> AsyncIterator[tuple[int, dict[bytes, bytes], AsyncIterator[bytes]]]:
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host, port, ssl=use_ssl),
        timeout=timeout,
    )
    try:
        writer.write(request_bytes)
        await writer.drain()
        status_code, headers = await _read_response_headers(reader)
        body = _iter_body_chunks(reader, headers)
        yield status_code, headers, body
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except ssl.SSLError as e:
            # Some TLS peers send data after close_notify, which violates the
            # TLS spec but does not affect already-received application data.
            # Only ignore that specific shutdown error; re-raise anything else.
            if getattr(e, "reason", None) != "APPLICATION_DATA_AFTER_CLOSE_NOTIFY":
                raise


async def _read_response_headers(reader: asyncio.StreamReader) -> tuple[int, dict[bytes, bytes]]:
    status_line = await reader.readline()
    if not status_line:
        raise ConnectionError("Server closed connection before sending a response")
    parts = status_line.rstrip(b"\r\n").split(b" ", 2)
    if len(parts) < 2:
        raise ConnectionError(f"Invalid HTTP status line: {status_line!r}")
    try:
        status_code = int(parts[1])
    except ValueError as e:
        raise ConnectionError(f"Invalid HTTP status code: {parts[1]!r}") from e

    headers: dict[bytes, bytes] = {}
    while True:
        line = await reader.readline()
        if not line or line in (b"\r\n", b"\n"):
            break
        key, _, value = line.partition(b":")
        headers[key.lower().strip()] = value.strip()

    return status_code, headers


async def _read_body_bytes(body: AsyncIterator[bytes]) -> bytes:
    chunks: list[bytes] = [chunk async for chunk in body]
    return b"".join(chunks)


async def _iter_body_chunks(
    reader: asyncio.StreamReader,
    headers: dict[bytes, bytes],
) -> AsyncIterator[bytes]:
    transfer_encoding = headers.get(b"transfer-encoding", b"").lower()
    if b"chunked" in transfer_encoding:
        while True:
            size_line = await reader.readline()
            if not size_line:
                break
            size_hex = size_line.split(b";", 1)[0].strip()
            if not size_hex:
                continue
            try:
                chunk_size = int(size_hex, 16)
            except ValueError as e:
                raise ConnectionError(f"Invalid chunk size: {size_line!r}") from e
            if chunk_size == 0:
                # Consume trailing headers/CRLF
                while True:
                    line = await reader.readline()
                    if not line or line in (b"\r\n", b"\n"):
                        break
                break
            chunk = await reader.readexactly(chunk_size)
            # Consume trailing CRLF after chunk data
            await reader.readline()
            yield chunk
    else:
        content_length = headers.get(b"content-length")
        if content_length is not None:
            remaining = int(content_length)
            while remaining > 0:
                chunk = await reader.read(min(8192, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
        else:
            while True:
                chunk = await reader.read(8192)
                if not chunk:
                    break
                yield chunk


# ---------------------------------------------------------------------------
# Generic SSE parsing utilities
# ---------------------------------------------------------------------------


async def _iter_lines(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if line.endswith(b"\r"):
                line = line[:-1]
            yield line
    if buffer:
        yield buffer


async def _iter_sse_events(lines: AsyncIterator[bytes]) -> AsyncIterator[list[bytes]]:
    event_lines: list[bytes] = []
    async for line in lines:
        if line == b"":
            if event_lines:
                yield event_lines
                event_lines = []
        else:
            event_lines.append(line)
    if event_lines:
        yield event_lines


def _extract_sse_data(event_lines: list[bytes]) -> bytes | None:
    data_lines: list[bytes] = []
    for line in event_lines:
        if line.startswith(b"data: "):
            data_lines.append(line[6:])
        elif line.startswith(b"data:"):
            data_lines.append(line[5:])
    if not data_lines:
        return None
    return b"\n".join(data_lines)


# ---------------------------------------------------------------------------
# OpenAI-specific engine
# ---------------------------------------------------------------------------


@BaseChatEngine.register("openai")
class OpenAIChatEngine(BaseChatEngine):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = self._ensure_api_key(api_key)
        self._api_base = self._ensure_api_base(api_base)

    @asynccontextmanager
    async def arun(self, query: ChatQuery) -> AsyncIterator[AsyncIterator[StreamEvent]]:
        parsed_url = urllib.parse.urlsplit(self._api_base)
        request_bytes = self._build_request(parsed_url, query)

        port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
        async with _http_stream(
            host=parsed_url.hostname or "",
            port=port,
            use_ssl=parsed_url.scheme == "https",
            request_bytes=request_bytes,
            timeout=60.0,
        ) as (status_code, headers, body):
            if status_code != 200:
                body_bytes = await _read_body_bytes(body)
                raise RuntimeError(f"OpenAI API returned {status_code}: {body_bytes.decode('utf-8', errors='replace')}")

            sse_events = _iter_sse_events(_iter_lines(body))
            yield self._parse_stream(sse_events)

    def _build_request(self, parsed_url: urllib.parse.SplitResult, query: ChatQuery) -> bytes:
        path = parsed_url.path.rstrip("/") + "/chat/completions"

        payload: dict = {
            "model": self._model,
            "messages": [self._convert_message(message) for message in query["messages"]],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if "tools" in query:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["parameters"],
                    },
                }
                for tool in query["tools"]
            ]
        body = json.dumps(payload).encode("utf-8")

        headers = {
            "Host": parsed_url.netloc,
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Accept": "text/event-stream",
            "Connection": "close",
        }
        request = (
            f"POST {path} HTTP/1.1\r\n" + "\r\n".join(f"{k}: {v}" for k, v in headers.items()) + "\r\n\r\n"
        ).encode("utf-8")
        return request + body

    def _convert_message(self, message: ChatMessage) -> dict:
        converted: dict = {
            "role": self._convert_role(message["role"]),
            "content": [self._convert_part(part) for part in message["parts"]],
        }
        if message["role"] == "assistant" and "tool_calls" in message:
            converted["tool_calls"] = [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["args"]),
                    },
                }
                for tool_call in message["tool_calls"]
            ]
        if message["role"] == "tool_result":
            converted["tool_call_id"] = message["tool_call_id"]
        return converted

    @staticmethod
    def _convert_role(role: str) -> str:
        if role == "tool_result":
            return "tool"
        return role

    def _convert_part(self, part: ContentPart) -> dict:
        match part["kind"]:
            case "text":
                return {"type": "text", "text": part["text"]}
            case "image_url":
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": part["url"],
                        "detail": part.get("detail", "auto"),
                    },
                }
            case "image_bytes":
                encoded = base64.b64encode(part["data"]).decode("ascii")
                data_url = f"data:{part['media_type']};base64,{encoded}"
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                        "detail": part.get("detail", "auto"),
                    },
                }
            case _ as unknown:
                assert_never(unknown)

    async def _parse_stream(self, sse_events: AsyncIterator[list[bytes]]) -> AsyncIterator[StreamEvent]:
        state = _StreamState()
        turn_done_emitted = False

        async for event_lines in sse_events:
            data = _extract_sse_data(event_lines)
            if data is None:
                continue
            data_str = data.decode("utf-8").strip()
            if data_str == "[DONE]":
                for event in state.flush_done_events():
                    yield event
                yield {
                    "kind": "turn_done",
                    "finish_reason": state.finish_reason or "stop",
                    "usage": state.usage,
                }
                turn_done_emitted = True
                return

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to decode SSE data: {data_str}") from e

            for event in self._convert_chunk(chunk, state):
                yield event

        if not turn_done_emitted and state.finish_reason is not None:
            for event in state.flush_done_events():
                yield event
            yield {
                "kind": "turn_done",
                "finish_reason": state.finish_reason,
                "usage": state.usage,
            }

    def _convert_chunk(self, chunk: dict, state: "_StreamState") -> list[StreamEvent]:
        events: list[StreamEvent] = []
        choices = chunk.get("choices") or []

        for choice in choices:
            index = choice.get("index", 0)
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason")

            content = delta.get("content")
            if content:
                if index not in state.text_started:
                    events.append({"kind": "text_part_started", "index": index})
                    state.text_started.add(index)
                events.append({"kind": "text_delta", "index": index, "delta": content})
                state.text_buffer.setdefault(index, []).append(content)

            tool_calls = delta.get("tool_calls") or []
            for tool_call in tool_calls:
                tc_index = tool_call.get("index", 0)
                tc_id = tool_call.get("id")
                function = tool_call.get("function") or {}
                name = function.get("name")
                arguments = function.get("arguments", "")

                if tc_id is not None and name is not None:
                    events.append(
                        {
                            "kind": "tool_call_part_started",
                            "index": tc_index,
                            "tool_call_id": tc_id,
                            "tool_name": name,
                        }
                    )
                    state.tool_calls[tc_index] = {"id": tc_id, "name": name, "args": ""}

                if arguments and tc_index in state.tool_calls:
                    events.append(
                        {
                            "kind": "tool_call_args_delta",
                            "index": tc_index,
                            "delta": arguments,
                        }
                    )
                    state.tool_calls[tc_index]["args"] += arguments

            if finish_reason is not None:
                state.finish_reason = finish_reason
                events.extend(state.flush_done_events())

        usage = chunk.get("usage")
        if usage:
            state.usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }

        return events

    @staticmethod
    def _ensure_api_key(api_key: str | None) -> str:
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "OpenAI API key must be provided either as an argument or via the OPENAI_API_KEY environment variable."
            )
        return api_key

    @staticmethod
    def _ensure_api_base(api_base: str | None) -> str:
        if api_base is None:
            api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        return api_base


class _StreamState:
    def __init__(self) -> None:
        self.text_started: set[int] = set()
        self.text_buffer: dict[int, list[str]] = {}
        self.tool_calls: dict[int, dict[str, str]] = {}
        self.finish_reason: FinishReason | None = None
        self.usage: Usage = {}

    def flush_done_events(self) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        for index, parts in list(self.text_buffer.items()):
            events.append({"kind": "text_part_done", "index": index, "text": "".join(parts)})
            del self.text_buffer[index]
        for tc_index, tool_call in list(self.tool_calls.items()):
            args = tool_call["args"]
            try:
                parsed_args = json.loads(args) if args else None
            except json.JSONDecodeError:
                parsed_args = args
            events.append(
                {
                    "kind": "tool_call_part_done",
                    "index": tc_index,
                    "tool_call_id": tool_call["id"],
                    "tool_name": tool_call["name"],
                    "args": parsed_args,
                }
            )
            del self.tool_calls[tc_index]
        return events
