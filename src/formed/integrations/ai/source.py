from __future__ import annotations

import asyncio
import enum
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Generic, TypeVar

from .exceptions import SlowSubscriberError

EventT = TypeVar("EventT")


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
        >>> from formed.integrations.ai.source import EventSource
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
