from __future__ import annotations

import dataclasses

import pytest

from formed.integrations.ai.providers.base.contextualizer import DefaultContextualizer
from formed.integrations.ai.providers.base.messages import TextContent, UserMessage
from formed.integrations.ai.providers.base.tools import Toolset


@dataclasses.dataclass(frozen=True)
class _State:
    pass


@pytest.mark.anyio
async def test_default_contextualizer_builds_query_with_tools_and_system() -> None:
    def ping() -> str:
        return "pong"

    toolset = Toolset.from_tools({"ping": ping})
    contextualizer = DefaultContextualizer[_State](
        system="system",
        toolset=toolset,
    )

    query = await contextualizer(_State(), "hello")

    assert query.system == "system"
    assert tuple(t.name for t in query.tools) == ("ping",)
    assert isinstance(query.history[0], UserMessage)
    assert query.history[0].parts == (TextContent(text="hello"),)


@pytest.mark.anyio
async def test_default_contextualizer_accepts_user_message() -> None:
    contextualizer = DefaultContextualizer[_State]()
    user = UserMessage(parts=(TextContent(text="x"),))

    query = await contextualizer(_State(), user)
    assert query.history[0] is user


@pytest.mark.anyio
async def test_default_contextualizer_accepts_toolset() -> None:
    def ping() -> str:
        return "pong"

    toolset = Toolset.from_tools({"ping": ping})
    contextualizer = DefaultContextualizer[_State](toolset=toolset)

    query = await contextualizer(_State(), "hi")
    assert tuple(t.name for t in query.tools) == ("ping",)
