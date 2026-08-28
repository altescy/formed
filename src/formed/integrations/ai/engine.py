import abc
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Generic, NotRequired, TypedDict, TypeVar

from colt import Registrable

from .entities.events import StreamEvent
from .entities.messages import ChatMessage
from .entities.tools import ToolSpec

QueryT = TypeVar("QueryT")


class BaseEngine(Generic[QueryT], Registrable, abc.ABC):
    @abc.abstractmethod
    def arun(self, query: QueryT) -> AbstractAsyncContextManager[AsyncIterator[StreamEvent]]:
        raise NotImplementedError


class ChatQuery(TypedDict):
    messages: Sequence[ChatMessage]
    tools: NotRequired[Sequence[ToolSpec]]


class BaseChatEngine(BaseEngine[ChatQuery], abc.ABC): ...
