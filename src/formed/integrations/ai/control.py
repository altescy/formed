from __future__ import annotations

import dataclasses
from typing import Generic, TypeVar

TerminalT_co = TypeVar("TerminalT_co", covariant=True)


@dataclasses.dataclass(frozen=True)
class Continue:
    """Signal that the agent loop should continue.

    Returned by a `Handler` to indicate that the loop should proceed to the
    next `Engine` call.  The handler is responsible for updating the query
    before returning `Continue`.
    """


@dataclasses.dataclass(frozen=True)
class Stop(Generic[TerminalT_co]):
    """Signal that the agent loop should terminate.

    Returned by a `Handler` to end the run and surface a final result.

    Attributes:
        result: The terminal value produced by the agent.
    """

    result: TerminalT_co
