import asyncio
import enum
from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any, Generic, TypeVar

EventT = TypeVar("EventT")
ItemT = TypeVar("ItemT")
ResultT_co = TypeVar("ResultT_co", covariant=True)
StateT = TypeVar("StateT")
StateT_co = TypeVar("StateT_co", covariant=True)


class SlowSubscriberError(Exception):
    """Raised by `EventSource.publish` when a subscriber queue is full.

    Only raised when `SlowSubscriberPolicy.ERROR` is active.  Use
    `SlowSubscriberPolicy.DROP` if you want to silently skip slow subscribers
    instead.
    """


class SlowSubscriberPolicy(enum.Enum):
    """Policy applied when a subscriber queue is full.

    Attributes:
        DROP: Skip delivery to the full queue.  Other subscribers and the
            publisher continue unaffected.
        ERROR: Raise `SlowSubscriberError` from `publish` when any queue is full.
    """

    DROP = "drop"
    ERROR = "error"


class EventSource(Generic[EventT]):
    """Fan-out event broadcaster with replay and error-propagation support.

    Multiple consumers can subscribe to the same source and each receives an
    independent copy of every event.  Subscriptions are managed via an async
    context manager so queues are always cleaned up, even on cancellation.

    Design guarantees:

    - The boundary between replayed history and live delivery is handled
      atomically inside the internal lock, preventing event loss or duplication.
    - Subscribing to an already-closed source exits immediately rather than
      blocking forever.
    - `publish` calls `put_nowait` on each queue *outside* the lock, so the
      lock is never held across a suspension point and other `subscribe` /
      `aclose` calls are never blocked.
    - `aclose(exception=...)` propagates an error to every active subscriber;
      each subscriber re-raises it at the end of its stream.
    - A class-specific sentinel object (not `None`) is used so that
      `EventT = None` works correctly.

    Args:
        max_buffer: Maximum number of events to buffer per subscriber queue and
            in the replay history.  Defaults to ``1000``.
        slow_subscriber_policy: Action taken when a subscriber queue is full.
            Defaults to `SlowSubscriberPolicy.DROP`.

    Examples:
        >>> import asyncio
        >>> from formed.common.streaming import EventSource
        >>>
        >>> async def demo() -> None:
        ...     source: EventSource[int] = EventSource()
        ...     async with source.subscribe() as stream:
        ...         await source.publish(1)
        ...         await source.publish(2)
        ...         await source.aclose()
        ...         print([e async for e in stream])  # [1, 2]
        >>>
        >>> asyncio.run(demo())
    """

    _CLOSED: object = object()

    def __init__(
        self,
        max_buffer: int = 1000,
        slow_subscriber_policy: SlowSubscriberPolicy = SlowSubscriberPolicy.DROP,
    ) -> None:
        self._max_buffer = max_buffer
        self._slow_subscriber_policy = slow_subscriber_policy
        self._queues: set[asyncio.Queue[EventT | object]] = set()
        self._history: deque[EventT] = deque(maxlen=max_buffer)
        self._lock = asyncio.Lock()
        self._closed = False
        self._exception: BaseException | None = None

    async def publish(self, event: EventT) -> None:
        """Broadcast an event to all active subscribers.

        Args:
            event: The event to deliver.

        Raises:
            RuntimeError: If the source has already been closed.
            SlowSubscriberError: If a queue is full and the policy is
                `SlowSubscriberPolicy.ERROR`.
        """
        async with self._lock:
            if self._closed:
                raise RuntimeError("EventSource is already closed")
            self._history.append(event)
            queues = list(self._queues)
        for q in queues:
            if q.full():
                if self._slow_subscriber_policy is SlowSubscriberPolicy.ERROR:
                    raise SlowSubscriberError(f"Subscriber queue is full (max_buffer={self._max_buffer})")
                # DROP: skip delivery to this subscriber
            else:
                q.put_nowait(event)

    @asynccontextmanager
    async def subscribe(self, replay: int = 0) -> AsyncIterator[AsyncIterator[EventT]]:
        """Async context manager that yields an event stream.

        Args:
            replay: Number of historical events to replay before live events.
                Must not exceed ``max_buffer``.

        Yields:
            An `AsyncIterator` that streams replayed and then live events.

        Raises:
            ValueError: If ``replay`` exceeds ``max_buffer``.
        """
        if replay > self._max_buffer:
            raise ValueError(f"replay={replay} exceeds max_buffer={self._max_buffer}")

        queue: asyncio.Queue[EventT | object] = asyncio.Queue(maxsize=self._max_buffer)

        async with self._lock:
            snapshot = list(self._history)[-replay:] if replay > 0 else []
            if self._closed:
                await queue.put(self._CLOSED)
            else:
                self._queues.add(queue)

        async def _iter() -> AsyncIterator[EventT]:
            for e in snapshot:
                yield e
            while True:
                item = await queue.get()
                if item is self._CLOSED:
                    break
                yield item  # type: ignore[misc]
            if self._exception is not None:
                raise self._exception

        try:
            yield _iter()
        finally:
            async with self._lock:
                self._queues.discard(queue)

    async def aclose(self, exception: BaseException | None = None) -> None:
        """Close the source and notify all subscribers.

        Args:
            exception: Optional exception to propagate to every active
                subscriber.  Subscribers will re-raise it at the end of their
                stream.
        """
        async with self._lock:
            self._closed = True
            self._exception = exception
            for q in self._queues:
                await q.put(self._CLOSED)


class StreamingResponse(Generic[EventT, StateT_co, ResultT_co]):
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
            ``(final_state, terminal_result)``.
    """

    def __init__(
        self,
        source: EventSource[EventT],
        coro: Callable[[], Coroutine[Any, Any, tuple[StateT_co, ResultT_co]]],
    ) -> None:
        self._source = source
        self._coro = coro
        self._task: asyncio.Task[tuple[StateT_co, ResultT_co]] | None = None

    def _ensure_started(self) -> asyncio.Task[tuple[StateT_co, ResultT_co]]:
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

    async def collect(self) -> tuple[StateT_co, ResultT_co]:
        """Await the agent to completion and return ``(final_state, result)``.

        Starts the agent if it has not started yet.

        Returns:
            A tuple of ``(final_state, result)`` where ``final_state`` is the
            agent state after the last ``Handler`` invocation and ``result`` is
            the value wrapped in ``Stop`` (post-processed by ``response_format``
            if one was supplied).

        Raises:
            Exception: Any exception raised inside the agent coroutine.
        """
        return await self._ensure_started()
