from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar, cast

from colt.builder import ColtBuilder

from ...control import Continue, Stop
from .messages import (
    AssistantMessage,
    ContentPart,
    ImageBytesContent,
    ImageUrlContent,
    Query,
    TextContent,
    ToolCallRecord,
    ToolResultMessage,
)
from .signals import Signal, TextOutput
from .tools import ToolArgsBuilder, Toolset

StateT = TypeVar("StateT")
TerminalT = TypeVar("TerminalT")


@dataclasses.dataclass(frozen=True)
class DefaultHandler(Generic[StateT, TerminalT]):
    """Provider-base handler for text responses and tool calls.

    The handler consumes all signals emitted in one engine turn at once,
    appends an :class:`AssistantMessage` to ``query.history``, executes tool
    calls when present, and decides whether the agent should continue.

    Args:
        toolset: Tool definitions and implementations.
        max_parallel_tool_calls: Maximum number of concurrent tool executions.
        raise_tool_errors: If ``True``, tool errors are raised and fail the run.
            If ``False``, errors are converted into tool result text.
        args_builder: Builder that converts JSON-like args to typed callable
            inputs. Defaults to ``ColtBuilder(strict=True)``.
        terminal_from_assistant: Converter from final assistant message to the
            terminal value surfaced by ``Stop``.
    """

    toolset: Toolset = dataclasses.field(default_factory=Toolset.empty)
    max_parallel_tool_calls: int | None = None
    raise_tool_errors: bool = False
    args_builder: ToolArgsBuilder = dataclasses.field(default_factory=lambda: ColtBuilder(strict=True), repr=False)
    terminal_from_assistant: Callable[[AssistantMessage], TerminalT] = dataclasses.field(
        default=lambda assistant: cast(TerminalT, assistant)
    )

    async def __call__(
        self,
        state: StateT,
        query: Query,
        signals: Sequence[Signal],
    ) -> tuple[StateT, Query, Continue | Stop[TerminalT]]:
        parts: list[ContentPart] = []
        tool_calls: list[ToolCallRecord] = []

        for signal in signals:
            if isinstance(signal, TextOutput):
                parts.append(TextContent(text=signal.text))
            elif isinstance(signal, ToolCallRecord):
                tool_calls.append(signal)

        assistant = AssistantMessage(
            parts=tuple(parts),
            tool_calls=tuple(tool_calls),
        )
        history = query.history
        if assistant.parts or assistant.tool_calls:
            history = (*history, assistant)

        if tool_calls:
            tool_results = await self._run_tool_calls(tool_calls)
            history = (*history, *tool_results)
            return state, dataclasses.replace(query, history=history), Continue()

        if not assistant.parts:
            return state, query, Continue()

        terminal = self.terminal_from_assistant(assistant)
        return state, dataclasses.replace(query, history=history), Stop(terminal)

    async def _run_tool_calls(self, tool_calls: Sequence[ToolCallRecord]) -> tuple[ToolResultMessage, ...]:
        if self.max_parallel_tool_calls is not None:
            semaphore = asyncio.Semaphore(self.max_parallel_tool_calls)
        else:
            semaphore = None

        async def _run(tc: ToolCallRecord) -> ToolResultMessage:
            if semaphore is None:
                return await self._run_tool_call(tc)
            async with semaphore:
                return await self._run_tool_call(tc)

        return tuple(await asyncio.gather(*(_run(tc) for tc in tool_calls)))

    async def _run_tool_call(self, tool_call: ToolCallRecord) -> ToolResultMessage:
        tool = self.toolset.get(tool_call.name)
        if tool is None:
            return ToolResultMessage(
                tool_call_id=tool_call.id,
                parts=(TextContent(text=f"Tool '{tool_call.name}' is not registered."),),
            )

        try:
            args = json.loads(tool_call.args_json) if tool_call.args_json else {}
            if args is None:
                args = {}
            result = self.args_builder(args, tool)

            if inspect.isawaitable(result):
                result = await result

            parts = _tool_result_to_parts(result)

        except BaseException as e:
            if self.raise_tool_errors:
                raise
            parts = (TextContent(text=f"Tool '{tool_call.name}' failed: {e}"),)

        return ToolResultMessage(tool_call_id=tool_call.id, parts=parts)


def _tool_result_to_parts(result: Any) -> tuple[ContentPart, ...]:
    if isinstance(result, (TextContent, ImageUrlContent, ImageBytesContent)):
        return (result,)

    if isinstance(result, str):
        return (TextContent(text=result),)

    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        if all(isinstance(p, (TextContent, ImageUrlContent, ImageBytesContent)) for p in result):
            return tuple(result)

    if result is None:
        return (TextContent(text=""),)

    return (TextContent(text=json.dumps(result, ensure_ascii=True, default=str)),)
