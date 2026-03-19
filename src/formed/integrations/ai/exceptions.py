from __future__ import annotations


class AgentExhausted(Exception):
    """Raised when the agent loop ends without emitting any signals.

    This happens when the `Reducer` consistently returns an empty sequence for
    every event produced by the `Engine`, meaning no `Signal` was ever
    generated to drive the `Handler`.
    """


class SlowSubscriberError(Exception):
    """Raised by `EventSource.publish` when a subscriber queue is full.

    Only raised when `SlowSubscriberPolicy.ERROR` is active.  Use
    `SlowSubscriberPolicy.DROP` if you want to silently skip slow subscribers
    instead.
    """
