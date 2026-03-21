from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Generic, TypeAlias, TypeVar

from .messages import (
    AssistantMessage,
    ContentPart,
    Query,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from .tools import Toolset

StateT = TypeVar("StateT")

HistoryMessage: TypeAlias = UserMessage | AssistantMessage | ToolResultMessage
DefaultRequest: TypeAlias = str | UserMessage | tuple[ContentPart, ...]


def _empty_history(_: object) -> tuple[HistoryMessage, ...]:
    return ()


@dataclasses.dataclass(frozen=True)
class DefaultContextualizer(Generic[StateT]):
    """Default contextualizer for chat-like requests.

    It converts an incoming request into one user message and appends it to
    history extracted from state.
    """

    system: str | None = None
    toolset: Toolset = dataclasses.field(default_factory=Toolset.empty)
    history_getter: Callable[[StateT], tuple[HistoryMessage, ...]] = _empty_history

    async def __call__(self, state: StateT, request: DefaultRequest) -> Query:
        history = self.history_getter(state)
        user = _to_user_message(request)
        return Query(
            system=self.system,
            history=(*history, user),
            tools=self.toolset.definitions,
        )


def _to_user_message(request: DefaultRequest) -> UserMessage:
    if isinstance(request, UserMessage):
        return request
    if isinstance(request, str):
        return UserMessage(parts=(TextContent(text=request),))
    return UserMessage(parts=request)
