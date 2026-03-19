"""Multi-agent orchestration via a router–worker pattern.

This module provides :class:`Orchestrator`, which coordinates a set of
specialised agents behind a single *router* agent.  The router inspects the
incoming request and returns a ``(agent_name, sub_request)`` tuple that tells
the orchestrator which worker agent to invoke and with what request.

The :class:`Orchestrator` exposes the same ``(state, request) → Response``
call interface as a plain :class:`~formed.integrations.ai.agent.Agent`, so it
can be nested transparently — used as a node in a
:class:`~formed.integrations.ai.extensions.graph.graph.Graph`, as a sub-agent
of another :class:`Orchestrator`, or anywhere an :class:`~formed.integrations.ai.agent.Agent`
is expected.

Examples:
    >>> from formed.integrations.ai.extensions.multi.orchestrator import Orchestrator
    >>> orchestrator = Orchestrator(
    ...     router=router_agent,
    ...     agents={
    ...         "search": search_agent,
    ...         "summarize": summarize_agent,
    ...     },
    ... )
    >>> response = orchestrator(state=state, request=user_query)
    >>> result = await response.collect()
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from ...agent import Agent
from ...response import Response
from ...source import EventSource

RequestT = TypeVar("RequestT")
ResultT_co = TypeVar("ResultT_co", covariant=True)
StateT = TypeVar("StateT")


class Orchestrator(Generic[RequestT, ResultT_co]):
    """Router–worker orchestrator that delegates to one of several agents.

    On each call the *router* agent analyses the request and returns a
    ``(agent_name, sub_request)`` pair.  The orchestrator then dispatches the
    *sub_request* to the named worker agent and forwards its result to the
    caller.

    Both the router's events and the selected worker's events are published to
    a shared :class:`~formed.integrations.ai.source.EventSource` so callers
    receive a single unified stream via
    :meth:`~formed.integrations.ai.response.Response.events`.

    Because :meth:`__call__` returns a standard
    :class:`~formed.integrations.ai.response.Response`, an
    :class:`Orchestrator` is interchangeable with a plain
    :class:`~formed.integrations.ai.agent.Agent` and can be nested inside
    other orchestrators or graph nodes.

    Args:
        agents: Dictionary of worker agents keyed by the name the router uses
            to identify them.
        router: Agent that receives the original request and returns a
            ``(agent_name, sub_request)`` tuple as its
            :class:`~formed.integrations.ai.control.Stop` result.

    Examples:
        >>> orchestrator = Orchestrator(
        ...     router=router_agent,
        ...     agents={
        ...         "search": search_agent,
        ...         "summarize": summarize_agent,
        ...     },
        ... )
        >>> response = orchestrator(state=state, request=user_query)
        >>> async for event in response.events():
        ...     print(event)
        >>> result = await response.collect()
    """

    def __init__(
        self,
        agents: dict[str, Agent[Any, Any, Any, Any, Any, Any]],
        router: Agent[RequestT, Any, Any, Any, Any, tuple[str, Any]],
    ) -> None:
        self._agents = agents
        self._router = router

    def __call__(self, state: Any, request: RequestT) -> Response[Any, Any, ResultT_co]:
        """Route *request* through the router and execute the selected worker.

        Creates a background coroutine that:

        1. Calls the router agent with *state* and *request*.
        2. Collects the router's ``(agent_name, sub_request)`` result.
        3. Calls the corresponding worker agent with *state* and *sub_request*.
        4. Returns the worker's result as the final response result.

        All events from both the router and the worker are forwarded to the
        shared :class:`~formed.integrations.ai.source.EventSource` so they
        appear in the stream returned by
        :meth:`~formed.integrations.ai.response.Response.events`.

        Args:
            state: Shared mutable state passed to both the router and the
                selected worker agent.
            request: The original request forwarded to the router.

        Returns:
            A :class:`~formed.integrations.ai.response.Response` whose event
            stream contains events from the router followed by events from the
            selected worker, and whose collected result is the worker's final
            output.
        """
        source: EventSource[Any] = EventSource()

        async def _run() -> tuple[Any, ResultT_co]:
            exc: BaseException | None = None
            try:
                router_response = self._router(state, request)
                async for event in router_response.events():
                    await source.publish(event)
                _, routing = await router_response.collect()
                agent_name, sub_request = routing

                agent = self._agents[agent_name]
                sub_response = agent(state, sub_request)
                async for event in sub_response.events():
                    await source.publish(event)
                sub_state, result = await sub_response.collect()
                return sub_state, result
            except BaseException as e:
                exc = e
                raise
            finally:
                await source.aclose(exception=exc)

        return Response(source=source, coro=_run)
