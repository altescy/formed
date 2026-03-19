from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class ToolDefinition:
    """Definition of a callable tool exposed to the model.

    Attributes:
        name: Tool name as the model will call it.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema object describing the arguments.
        strict: If ``True``, request strict schema adherence (e.g. OpenAI
            structured outputs). ``None`` means provider default.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool | None = None
