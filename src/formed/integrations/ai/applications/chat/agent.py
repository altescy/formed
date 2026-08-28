import asyncio
import dataclasses
import inspect
from collections.abc import Mapping, Sequence
from typing import Generic, NotRequired, TypedDict, cast

from typing_extensions import TypeVar

from ...agent import Agent, BaseHandler, BaseReducer
from ...engine import ChatQuery
from ...entities.controls import Continue, Control, Stop
from ...entities.events import StreamEvent
from ...entities.messages import AssistantMessage, ChatMessage, ToolCallRecord, ToolResultMessage
from ...entities.tools import ToolDefinition
from ...services.tooling import InvalidToolArgumentsError, execute_tool_with_params_json

FormatT = TypeVar("FormatT", default=str)


class ChatRequest(TypedDict):
    messages: Sequence[ChatMessage]
    tools: NotRequired[Sequence[ToolDefinition]]


@dataclasses.dataclass(frozen=True)
class ChatState:
    messages: list[ChatMessage] = dataclasses.field(default_factory=list)


@BaseReducer.register("chat")
class ChatReducer(BaseReducer[ChatState, ChatMessage]):
    async def __call__(self, state: ChatState, event: StreamEvent) -> tuple[ChatState, Sequence[ChatMessage]]:
        match event["kind"]:
            case "text_part_done":
                return state, [AssistantMessage(role="assistant", parts=[{"kind": "text", "text": event["text"]}])]
            case "tool_call_part_done":
                return state, [
                    AssistantMessage(
                        role="assistant",
                        parts=[],
                        tool_calls=[
                            ToolCallRecord(id=event["tool_call_id"], name=event["tool_name"], args=event["args"])
                        ],
                    )
                ]
        return state, []


@BaseHandler.register("chat")
class ChatHandler(Generic[FormatT], BaseHandler[ChatRequest, ChatQuery, ChatState, ChatMessage, FormatT]):
    def __init__(self, response_format: type[FormatT]) -> None:
        self._response_format = response_format

    async def __call__(
        self,
        request: ChatRequest,
        state: ChatState,
        signals: Sequence[ChatMessage] | None = None,
    ) -> tuple[ChatState, ChatQuery, Control[FormatT]]:
        if signals is None:
            state = dataclasses.replace(state, messages=[*state.messages, *request["messages"]])

        tool_definitions = {tool.name: tool for tool in request.get("tools", ())}
        tool_specs = [tool.spec for tool in request.get("tools", ())]
        next_query = ChatQuery(messages=state.messages, tools=tool_specs)

        if not signals:
            return state, next_query, Continue()

        responses: list[ChatMessage] = list(signals)

        last_assisntant_message: AssistantMessage | None = None
        for message in signals:
            match message["role"]:
                case "assistant":
                    if tool_calls := message.get("tool_calls"):
                        responses.extend(await self._handle_tool_calls(tool_calls, tool_definitions))
                    last_assisntant_message = message

        state = dataclasses.replace(state, messages=[*state.messages, *responses])
        next_query |= ChatQuery(messages=state.messages)

        if (
            not last_assisntant_message
            or last_assisntant_message.get("tool_calls")
            or last_assisntant_message.get("thinking")
            or not last_assisntant_message.get("parts")
        ):
            return state, next_query, Continue()

        terminal = self._format_response(last_assisntant_message)
        return state, next_query, Stop(terminal)

    def _get_text_content_from_message(self, message: ChatMessage) -> str:
        return "".join(part["text"] for part in message["parts"] if part["kind"] == "text")

    def _format_response(self, message: ChatMessage) -> FormatT:
        if self._response_format is str:
            return cast(FormatT, self._get_text_content_from_message(message))
        raise NotImplementedError("Response formatting is not implemented yet.")

    async def _handle_tool_calls(
        self,
        tool_calls: Sequence[ToolCallRecord],
        tool_definitions: Mapping[str, ToolDefinition],
    ) -> list[ToolResultMessage]:

        async def execute_tool(tool_call: ToolCallRecord) -> ToolResultMessage:
            tool_def = tool_definitions.get(tool_call["name"])
            if not tool_def:
                return ToolResultMessage(
                    role="tool_result",
                    tool_call_id=tool_call["id"],
                    parts=[{"kind": "text", "text": f"Tool '{tool_call['name']}' not found."}],
                )

            try:
                result = execute_tool_with_params_json(tool_def, tool_call["args"])
                if inspect.iscoroutine(result):
                    result = await result
                return ToolResultMessage(
                    role="tool_result",
                    tool_call_id=tool_call["id"],
                    parts=[{"kind": "text", "text": str(result)}],
                )
            except InvalidToolArgumentsError as e:
                return ToolResultMessage(
                    role="tool_result",
                    tool_call_id=tool_call["id"],
                    parts=[{"kind": "text", "text": f"Invalid arguments for tool '{tool_call['name']}': {e}"}],
                )
            except Exception:
                return ToolResultMessage(
                    role="tool_result",
                    tool_call_id=tool_call["id"],
                    parts=[{"kind": "text", "text": f"Error executing tool '{tool_call['name']}'."}],
                )

        return await asyncio.gather(*map(execute_tool, tool_calls))


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        from ...providers.openai import OpenAIChatEngine
        from ...services.tooling import build_tool_definition

        def get_weather(city: str) -> str:
            """Get the current weather for a given city."""
            return f"The current weather in {city} is sunny."

        engine = OpenAIChatEngine(model="gpt-5-mini")

        agent = Agent(
            engine,
            ChatReducer(),
            ChatHandler(str),
        )

        state = ChatState()

        while True:
            try:
                prompt = input("> ")
                if not prompt:
                    continue
                response = await agent.arun(
                    {
                        "messages": [{"role": "user", "parts": [{"kind": "text", "text": prompt}]}],
                        "tools": [build_tool_definition(get_weather)],
                    },
                    state,
                )
                async for event in response.events():
                    match event["kind"]:
                        case "tool_call_part_started":
                            print(f"🔧 Calling tool '{event['tool_name']}'...", flush=True)
                        case "text_delta":
                            print(event["delta"], end="", flush=True)
                        case "text_part_done":
                            print("\n")
                state, _ = await response.collect()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break

    asyncio.run(main())
