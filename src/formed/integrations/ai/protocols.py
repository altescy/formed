from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol, TypeVar, runtime_checkable

from .control import Continue, Stop

EventT_co = TypeVar("EventT_co", covariant=True)
EventT_contra = TypeVar("EventT_contra", contravariant=True)
StateT = TypeVar("StateT")
StateT_contra = TypeVar("StateT_contra", contravariant=True)
SignalT_co = TypeVar("SignalT_co", covariant=True)
SignalT_contra = TypeVar("SignalT_contra", contravariant=True)
ResultT_co = TypeVar("ResultT_co", covariant=True)
RequestT_contra = TypeVar("RequestT_contra", contravariant=True)
QueryT = TypeVar("QueryT")
QueryT_co = TypeVar("QueryT_co", covariant=True)
QueryT_contra = TypeVar("QueryT_contra", contravariant=True)
TerminalT_co = TypeVar("TerminalT_co", covariant=True)


@runtime_checkable
class Engine(Protocol[QueryT_contra, EventT_co]):
    """Protocol for the first stage of the agent loop.

    An `Engine` accepts a query and returns an async context manager that
    yields an event stream.  Using a context manager lets the engine own the
    full lifecycle of the underlying LLM connection, including cleanup on
    cancellation.

    Examples:
        >>> async with engine(query) as stream:
        ...     async for event in stream:
        ...         process(event)
    """

    def __call__(self, query: QueryT_contra) -> AbstractAsyncContextManager[AsyncIterator[EventT_co]]: ...


@runtime_checkable
class Reducer(Protocol[EventT_contra, StateT, SignalT_co]):
    """Protocol for folding events into signals.

    A `Reducer` is a pure function ``(state, event) → (new_state, signals)``
    that accumulates streaming events into internal state and emits completed
    semantic units (*signals*) once they are ready.  While still accumulating,
    it returns an empty sequence.

    ```
    fold(reducer, state₀, events) : [Event] → (State, [Signal])
    ```

    A `Reducer` must have no side effects.
    """

    def __call__(self, state: StateT, event: EventT_contra) -> tuple[StateT, Sequence[SignalT_co]]: ...


@runtime_checkable
class Handler(Protocol[QueryT, StateT, SignalT_contra, TerminalT_co]):
    """Protocol for folding signals into control decisions.

    A `Handler` is an async function
    ``(state, query, signal) → (new_state, new_query, control)`` that acts as
    the second fold layer, symmetric to the `Reducer`.

    ```
    fold(handler, ctx₀, signals) : [Signal] → (State, Query, Control)
        ctx₀ = (State, Query)
    ```

    The handler is responsible for updating state and query, executing side
    effects such as tool calls, and deciding whether to continue or stop:

    - Returning `Continue` resumes the loop with `new_query`.
    - Returning `Stop(result)` terminates the agent and surfaces `result`.
    """

    async def __call__(
        self,
        state: StateT,
        query: QueryT,
        signal: SignalT_contra,
    ) -> tuple[StateT, QueryT, Continue | Stop[TerminalT_co]]: ...


@runtime_checkable
class Contextualizer(Protocol[RequestT_contra, QueryT_co, StateT_contra]):
    """Protocol for converting a request into the initial query.

    A `Contextualizer` is an async function
    ``(state, request) → query`` that translates the caller's request into
    the first query sent to the `Engine`.  State is read-only.  The async
    signature allows I/O-bound initialization such as RAG retrieval.
    """

    async def __call__(self, state: StateT_contra, request: RequestT_contra) -> QueryT_co: ...
