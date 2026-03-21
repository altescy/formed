from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Literal

import pytest

from formed.integrations.ai.agent import Agent
from formed.integrations.ai.control import Continue, Stop
from formed.integrations.ai.providers.base.events import StreamEvent, TextPartDone, TurnDone
from formed.integrations.ai.providers.base.handler import DefaultHandler
from formed.integrations.ai.providers.base.messages import (
    AssistantMessage,
    Query,
    TextContent,
    ToolCallRecord,
    ToolResultMessage,
)
from formed.integrations.ai.providers.base.reducer import AgentReducer, ReducerState
from formed.integrations.ai.providers.base.signals import Signal, TextOutput
from formed.integrations.ai.providers.base.tools import Toolset


class _BatchEngine:
    @asynccontextmanager
    async def __call__(self, query: Query) -> AsyncIterator[AsyncIterator[StreamEvent]]:
        async def _stream() -> AsyncIterator[StreamEvent]:
            yield TextPartDone(index=0, text="hello")
            yield TextPartDone(index=1, text=" world")
            yield TurnDone(finish_reason="stop")

        yield _stream()


class _Contextualizer:
    async def __call__(self, state: ReducerState, request: str) -> Query:
        return Query(history=())


class _BatchHandler:
    def __init__(self) -> None:
        self.calls: list[Sequence[Signal]] = []

    async def __call__(
        self,
        state: ReducerState,
        query: Query,
        signals: Sequence[Signal],
    ) -> tuple[ReducerState, Query, Continue | Stop[str]]:
        self.calls.append(tuple(signals))
        text = "".join(signal.text for signal in signals if isinstance(signal, TextOutput))
        return state, query, Stop(text)


@pytest.mark.anyio
async def test_agent_passes_signals_to_handler_in_single_batch() -> None:
    handler = _BatchHandler()
    agent: Agent[str, Query, StreamEvent, ReducerState, Signal, str] = Agent(
        engine=_BatchEngine(),
        reducer=AgentReducer(),
        handler=handler,
        contextualizer=_Contextualizer(),
    )

    _, result = await agent(ReducerState(), "ignored").collect()

    assert result == "hello world"
    assert len(handler.calls) == 1
    assert len(handler.calls[0]) == 2


@pytest.mark.anyio
async def test_default_handler_executes_tool_calls_in_parallel() -> None:
    async def sleep_tool(value: str) -> str:
        await asyncio.sleep(0.05)
        return f"done:{value}"

    handler: DefaultHandler[object, AssistantMessage] = DefaultHandler(
        toolset=Toolset.from_tools({"sleep_tool": sleep_tool}),
    )
    signals: tuple[Signal, ...] = (
        ToolCallRecord(id="tc-1", name="sleep_tool", args_json=json.dumps({"value": "a"})),
        ToolCallRecord(id="tc-2", name="sleep_tool", args_json=json.dumps({"value": "b"})),
    )

    started = time.perf_counter()
    _, query, control = await handler(object(), Query(), signals)
    elapsed = time.perf_counter() - started

    assert isinstance(control, Continue)
    assert elapsed < 0.09
    assert isinstance(query.history[0], AssistantMessage)
    assert isinstance(query.history[1], ToolResultMessage)
    assert isinstance(query.history[2], ToolResultMessage)


@pytest.mark.anyio
async def test_default_handler_stops_on_text_without_tools() -> None:
    handler: DefaultHandler[object, AssistantMessage] = DefaultHandler()

    _, query, control = await handler(object(), Query(), (TextOutput(text="final"),))

    assert isinstance(control, Stop)
    assert isinstance(control.result, AssistantMessage)
    assert control.result.parts == (TextContent(text="final"),)
    assert len(query.history) == 1


@dataclasses.dataclass(frozen=True)
class _Location:
    city: str


@dataclasses.dataclass(frozen=True)
class _WeatherQuery:
    location: _Location
    units: Literal["c", "f"] = "c"


@pytest.mark.anyio
async def test_default_handler_builds_complex_typed_tool_arguments() -> None:
    def weather(query: _WeatherQuery) -> str:
        return f"{query.location.city}:{query.units}:{type(query).__name__}"

    handler: DefaultHandler[object, AssistantMessage] = DefaultHandler(
        toolset=Toolset.from_tools({"weather": weather}),
    )
    signal = ToolCallRecord(
        id="tc-typed",
        name="weather",
        args_json=json.dumps({"query": {"location": {"city": "Tokyo"}, "units": "f"}}),
    )

    _, query, control = await handler(object(), Query(), (signal,))

    assert isinstance(control, Continue)
    result = query.history[1]
    assert isinstance(result, ToolResultMessage)
    assert result.parts == (TextContent(text="Tokyo:f:_WeatherQuery"),)


@pytest.mark.anyio
async def test_default_handler_accepts_callable_sequence_tools() -> None:
    def ping(message: str) -> str:
        return f"pong:{message}"

    handler: DefaultHandler[object, AssistantMessage] = DefaultHandler(toolset=Toolset.from_tools((ping,)))
    signal = ToolCallRecord(
        id="tc-ping",
        name="ping",
        args_json=json.dumps({"message": "hello"}),
    )

    _, query, control = await handler(object(), Query(), (signal,))

    assert isinstance(control, Continue)
    assert tuple(tool.name for tool in handler.toolset.definitions) == ("ping",)
    result = query.history[1]
    assert isinstance(result, ToolResultMessage)
    assert result.parts == (TextContent(text="pong:hello"),)


@pytest.mark.anyio
async def test_default_handler_supports_custom_args_builder() -> None:
    def concat(message: str) -> str:
        return f"ok:{message}"

    class _ArgsBuilder:
        def __call__(self, config: object, cls: object) -> object:
            assert callable(cls)
            if isinstance(config, dict):
                return cls(**config)  # type: ignore[misc]
            return cls(config)  # type: ignore[misc]

    handler: DefaultHandler[object, AssistantMessage] = DefaultHandler(
        toolset=Toolset.from_tools((concat,)),
        args_builder=_ArgsBuilder(),
    )

    _, query, control = await handler(
        object(),
        Query(),
        (ToolCallRecord(id="tc-concat", name="concat", args_json=json.dumps({"message": "x"})),),
    )

    assert isinstance(control, Continue)
    result = query.history[1]
    assert isinstance(result, ToolResultMessage)
    assert result.parts == (TextContent(text="ok:x"),)
