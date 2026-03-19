from __future__ import annotations

import dataclasses
from typing import Union

from .messages import ToolCallRecord


@dataclasses.dataclass(frozen=True)
class TextOutput:
    """Completed text turn produced by the assistant.

    Emitted by :class:`AgentReducer` when a :class:`~events.TextPartDone`
    event arrives.

    Attributes:
        text: Full assembled text of the assistant turn.
    """

    text: str


# ToolCallRecord is defined in messages.py because AssistantMessage also
# references it.  Re-export it here so callers can import from one place.
Signal = Union[TextOutput, ToolCallRecord]

__all__ = [
    "TextOutput",
    "Signal",
]
