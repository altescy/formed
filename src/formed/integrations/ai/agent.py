from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar, overload

from .control import Stop
from .exceptions import AgentExhausted
from .protocols import Contextualizer, Engine, Handler, Reducer
from .response import Response
from .source import EventSource

RequestT = TypeVar("RequestT")
QueryT = TypeVar("QueryT")
EventT = TypeVar("EventT")
StateT = TypeVar("StateT")
SignalT = TypeVar("SignalT")
TerminalT = TypeVar("TerminalT")
ResultT = TypeVar("ResultT")


class Agent(Generic[RequestT, QueryT, EventT, StateT, SignalT, TerminalT]):
    """Agent execution engine built on a two-stage fold.

    The loop is expressed as two sequential folds over the event and signal
    sequences:

    ```
    fold(reducer, state₀, engine(query)) : [Event]  → (State, [Signal])
    fold(handler, ctx₀,   signals)       : [Signal] → (State, Query, Control)
    ```

    Execution flow:

    1. `contextualizer(state, request)` builds the initial query.
    2. `engine(query)` opens a streaming context; the `reducer` folds every
       event into state and accumulates signals.
    3. The `handler` folds all signals emitted in the turn:
        - `Stop(result)` — terminates the loop and returns ``(final_state, result)``.
        - `Continue` — goes back to step 2 with the updated query.
    4. If the engine stream ends without producing any signals,
       `AgentExhausted` is raised.

    Exceptions propagate to the `EventSource` (notifying all subscribers)
    and are re-raised from `Response.collect`.

    Responsibilities:

    - `Contextualizer` — `(State, Request) → Query`.  Read-only on state.
    - `Reducer` — `(State, Event) → (State, [Signal])`.  Pure function.
    - `Handler` — `(State, Query, Signal) → (State, Query, Control)`.
      Owns query updates, side effects (e.g. tool calls), and termination.

    Args:
        engine: Streaming event source.
        reducer: Pure fold function over events.
        handler: Async fold function over signals.
        contextualizer: Converts the initial request into the first query.

    Examples:
        >>> response = agent(state=my_state, request=my_request)
        >>> async for event in response.events():
        ...     print(event)
        >>> final_state, result = await response.collect()
    """

    def __init__(
        self,
        engine: Engine[QueryT, EventT],
        reducer: Reducer[EventT, StateT, SignalT],
        handler: Handler[QueryT, StateT, SignalT, TerminalT],
        contextualizer: Contextualizer[RequestT, QueryT, StateT],
    ) -> None:
        self._engine = engine
        self._reducer = reducer
        self._handler = handler
        self._contextualizer = contextualizer

    @overload
    def __call__(
        self,
        state: StateT,
        request: RequestT,
        response_format: Callable[[TerminalT], ResultT],
    ) -> Response[EventT, StateT, ResultT]: ...

    @overload
    def __call__(
        self,
        state: StateT,
        request: RequestT,
        response_format: None = None,
    ) -> Response[EventT, StateT, TerminalT]: ...

    def __call__(
        self,
        state: StateT,
        request: RequestT,
        response_format: Callable[[TerminalT], ResultT] | None = None,
    ) -> Response[EventT, StateT, ResultT | TerminalT]:
        """Start (lazily) and return a handle to the agent execution.

        The agent does not actually start until the returned `Response` is
        consumed.

        Args:
            state: Initial agent state.
            request: Caller's request, converted to the first query by the
                `Contextualizer`.
            response_format: Optional function to post-process the terminal
                result before it is surfaced by ``Response.collect``.

        Returns:
            A `Response` handle for streaming events and collecting
            ``(final_state, result)``.
        """
        source: EventSource[EventT] = EventSource()
        initial_state = state

        async def _run() -> tuple[StateT, ResultT | TerminalT]:
            state = initial_state
            exc: BaseException | None = None
            try:
                query = await self._contextualizer(state, request)

                while True:
                    signals: list[SignalT] = []

                    async with self._engine(query) as stream:
                        async for event in stream:
                            await source.publish(event)
                            state, new_signals = self._reducer(state, event)
                            signals.extend(new_signals)

                    if not signals:
                        raise AgentExhausted("Engine stream ended without producing any signals.")

                    state, query, control = await self._handler(state, query, signals)
                    if isinstance(control, Stop):
                        if response_format is not None:
                            return state, response_format(control.result)
                        return state, control.result

            except BaseException as e:
                exc = e
                raise
            finally:
                await source.aclose(exception=exc)

        return Response(source=source, coro=_run)
