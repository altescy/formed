from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, Generic, TypeVar

from .source import EventSource

EventT = TypeVar("EventT")
StateT = TypeVar("StateT")
ResultT_co = TypeVar("ResultT_co", covariant=True)
ResultT = TypeVar("ResultT")
ItemT = TypeVar("ItemT")


class Response(Generic[EventT, ResultT_co]):
    """Handle to a lazily-started agent execution.

    `Response` decouples *subscribing* to the event stream from *starting*
    the agent.  Execution begins the first time `events`, `select`, `scan`,
    or `collect` is called, so multiple subscribers can be registered before
    the engine produces any events — without a ``sleep(0)`` workaround.

    Each call to `events`, `select`, or `scan` creates an independent
    subscription, allowing multiple concurrent consumers of the same run.

    If an exception occurs inside the agent it is re-raised by `collect` *and*
    propagated to every active event stream, so stream-only consumers can also
    detect failures.

    Args:
        source: The `EventSource` that the agent publishes events to.
        coro: Zero-argument coroutine factory that runs the agent and returns
            the terminal result.
    """

    def __init__(
        self,
        source: EventSource[EventT],
        coro: Callable[[], Coroutine[Any, Any, ResultT_co]],
    ) -> None:
        self._source = source
        self._coro = coro
        self._task: asyncio.Task[ResultT_co] | None = None

    def _ensure_started(self) -> asyncio.Task[ResultT_co]:
        if self._task is None:
            self._task = asyncio.create_task(self._coro())
        return self._task

    def events(self, replay: int = 0) -> AsyncIterator[EventT]:
        """Return an async iterator over all events emitted by the agent.

        Starts the agent if it has not started yet.

        Args:
            replay: Number of already-emitted events to replay before live
                events.  Passed through to `EventSource.subscribe`.

        Returns:
            An `AsyncIterator` of raw events.
        """
        self._ensure_started()

        async def _iter() -> AsyncIterator[EventT]:
            async with self._source.subscribe(replay=replay) as stream:
                async for e in stream:
                    yield e

        return _iter()

    def select(self, fn: Callable[[EventT], ItemT | None]) -> AsyncIterator[ItemT]:
        """Return an async iterator of projected, non-`None` values.

        Args:
            fn: Projection function.  Events for which ``fn`` returns ``None``
                are skipped.

        Returns:
            An `AsyncIterator` of the non-``None`` values returned by ``fn``.
        """

        async def _iter() -> AsyncIterator[ItemT]:
            async for e in self.events():
                if (x := fn(e)) is not None:
                    yield x

        return _iter()

    def scan(
        self,
        init: StateT,
        reducer: Callable[[StateT, EventT], StateT],
    ) -> AsyncIterator[StateT]:
        """Return an async iterator of running states.

        Args:
            init: Initial state value.
            reducer: Function that folds each event into the current state.

        Returns:
            An `AsyncIterator` that yields the state after each event.
        """

        async def _iter() -> AsyncIterator[StateT]:
            state = init
            async for e in self.events():
                state = reducer(state, e)
                yield state

        return _iter()

    async def collect(self) -> ResultT_co:
        """Await the agent to completion and return the terminal result.

        Starts the agent if it has not started yet.

        Returns:
            The value wrapped in `Stop` by the `Handler`.

        Raises:
            Exception: Any exception raised inside the agent coroutine.
        """
        return await self._ensure_started()
