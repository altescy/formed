"""AI agent framework based on a two-stage fold architecture.

Implements the design established in the prototype (work/aiv6.py) as the
production codebase.

The execution pipeline is expressed as two sequential folds:

```
Engine  →  [Event]  →  Reducer  →  [Signal]  →  Handler  →  Control
```

Module layout:

- **Core abstract layer** — `Agent`, `Response`, `EventSource`, protocols, control
  values, and exceptions.
- **providers/base** — Provider-agnostic event types, message types, and `AgentReducer`.
- **providers/openai** — `OpenAIEngine` for the OpenAI Chat Completions Streaming API.
- **providers/litellm** — `LiteLLMEngine` wrapping the LiteLLM unified interface.
- **extensions/agui** — Adapter that converts a `Response` stream to AG-UI protocol events.
- **extensions/graph** — Graph execution engine that composes `Agent` instances as nodes.
- **extensions/multi** — `Orchestrator` for multi-agent coordination.

Examples:
    >>> from formed.integrations.ai import Agent, Response, EventSource
    >>> from formed.integrations.ai import Engine, Reducer, Handler, Contextualizer
    >>> from formed.integrations.ai import Continue, Stop, AgentExhausted

"""

from .agent import Agent
from .control import Continue, Stop
from .exceptions import AgentExhausted, SlowSubscriberError
from .protocols import Contextualizer, Engine, Handler, Reducer
from .response import Response
from .source import EventSource, SlowSubscriberPolicy

__all__ = [
    # core
    "Agent",
    "Continue",
    "Stop",
    "AgentExhausted",
    "SlowSubscriberError",
    "Contextualizer",
    "Engine",
    "Handler",
    "Reducer",
    "Response",
    "EventSource",
    "SlowSubscriberPolicy",
]
