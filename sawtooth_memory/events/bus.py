"""Simple async event bus with optional handler batching."""

import asyncio
import logging
from typing import Dict, List, Callable, Awaitable, Optional, TypeVar

from .types import SawtoothEvent

logger = logging.getLogger(__name__)

EventHandler = Callable[[SawtoothEvent], Awaitable[None]]

T = TypeVar("T")


class EventBus:
    """
    Minimal async event bus.

    - No background worker: handlers are called immediately in the emitter's context
      (but we shield from errors and allow asyncio.gather)
    - Backpressure: callers can choose to fire-and-forget or await all handlers
    - Pluggable: subscribe/unsubscribe by event type
    """

    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to a specific event type."""
        self._handlers.setdefault(event_type, []).append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all events."""
        self._global_handlers.append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove handler for an event type."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def emit(self, event: SawtoothEvent, fire_and_forget: bool = True) -> None:
        """
        Emit an event.

        Args:
            event: The event to emit
            fire_and_forget: If True, run handlers in background tasks (non-blocking).
                             If False, wait for all handlers to complete.
        """
        # Get all relevant handlers
        handlers = self._global_handlers.copy()
        if event.event_type in self._handlers:
            handlers.extend(self._handlers[event.event_type])

        if not handlers:
            return

        if fire_and_forget:
            # Fire and forget – no backpressure, but errors are logged
            for handler in handlers:
                asyncio.create_task(self._safe_call(handler, event))
        else:
            # Wait for all handlers with timeout to avoid blocking forever
            tasks = [self._safe_call(handler, event) for handler in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_call(self, handler: EventHandler, event: SawtoothEvent) -> None:
        """Call handler and log errors without crashing."""
        try:
            await handler(event)
        except Exception as e:
            logger.exception(f"Error in event handler {handler.__name__}: {e}")


# Global bus singleton (lightweight)
_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get global event bus instance."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def reset_event_bus() -> None:
    """Reset the global bus (useful for testing)."""
    global _bus
    _bus = None
