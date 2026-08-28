import abc
import asyncio
from collections.abc import Sequence
from functools import partial
from typing import Generic

from colt import Registrable
from typing_extensions import TypeVar

from formed.common.streaming import EventSource, StreamingResponse

from .engine import BaseEngine
from .entities.controls import Control, Stop
from .entities.events import StreamEvent

RequestT_contra = TypeVar("RequestT_contra", contravariant=True)
QueryT = TypeVar("QueryT")
QueryT_co = TypeVar("QueryT_co", covariant=True)
SignalT = TypeVar("SignalT")
SignalT_co = TypeVar("SignalT_co", covariant=True)
StateT = TypeVar("StateT")
StateT_contra = TypeVar("StateT_contra", contravariant=True)
ResultT = TypeVar("ResultT")
ResultT_co = TypeVar("ResultT_co", covariant=True)


class BaseReducer(Registrable, abc.ABC, Generic[StateT, SignalT_co]):
    @abc.abstractmethod
    async def __call__(self, state: StateT, event: StreamEvent) -> tuple[StateT, Sequence[SignalT_co]]:
        raise NotImplementedError


class BaseHandler(Registrable, abc.ABC, Generic[RequestT_contra, QueryT_co, StateT, SignalT, ResultT]):
    @abc.abstractmethod
    async def __call__(
        self,
        request: RequestT_contra,
        state: StateT,
        signals: Sequence[SignalT] | None = None,
    ) -> tuple[StateT, QueryT_co, Control[ResultT]]:
        raise NotImplementedError


class Agent(
    Registrable,
    Generic[
        RequestT_contra,
        QueryT,
        StateT,
        SignalT,
        ResultT_co,
    ],
):
    def __init__(
        self,
        engine: BaseEngine,
        reducer: BaseReducer[StateT, SignalT],
        handler: BaseHandler[RequestT_contra, QueryT, StateT, SignalT, ResultT_co],
    ) -> None:
        self._engine = engine
        self._reducer = reducer
        self._handler = handler

    async def arun(
        self,
        request: RequestT_contra,
        state: StateT,
    ) -> StreamingResponse[StreamEvent, StateT, ResultT_co]:
        source = EventSource[StreamEvent]()

        async def _run(state: StateT) -> tuple[StateT, ResultT_co]:
            state, query, control = await self._handler(request, state)
            match control:
                case Stop(result):
                    return state, result

            error: BaseException | None = None
            try:
                while True:
                    async with self._engine.arun(query) as stream:
                        async for event in stream:
                            async with asyncio.TaskGroup() as tg:
                                tg.create_task(source.publish(event))
                                reduce_task = tg.create_task(self._reducer(state, event))
                            state, signals = reduce_task.result()
                            state, query, control = await self._handler(request, state, signals)
                            match control:
                                case Stop(result):
                                    return state, result
            except BaseException as e:
                error = e
                raise
            finally:
                await source.aclose(error)

            raise RuntimeError("Agent run loop exited without returning a result")

        return StreamingResponse(source, partial(_run, state))
