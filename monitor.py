"""
monitor.py — Token counting and threshold detection.

Uses tiktoken to count tokens locally before deciding whether to
trigger background compression.
"""

from __future__ import annotations

import logging

import tiktoken

from .state import MemoryState, Message

logger = logging.getLogger(__name__)

_MESSAGE_OVERHEAD = 4


class TokenMonitor:
    """
    Counts tokens using tiktoken and detects when Working Memory (L1)
    has crossed the soft compression threshold.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        soft_limit: int = 3000,
        hard_limit: int = 6000,
    ) -> None:
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        self._enc = tiktoken.encoding_for_model(model)

    def count_text(self, text: str) -> int:
        """Return the token count of a raw string."""
        if not text.strip():
            return 0
        return len(self._enc.encode(text))

    def count_message(self, message: Message) -> int:
        """Return the token count of a Message including role overhead."""
        return self.count_text(message.content) + _MESSAGE_OVERHEAD

    def exceeds_soft_limit(self, state: MemoryState) -> bool:
        return state.l1_working.token_count >= self.soft_limit

    def exceeds_hard_limit(self, state: MemoryState) -> bool:
        return state.l1_working.token_count >= self.hard_limit

    def recount_working_memory(self, state: MemoryState) -> None:
        """Recompute token counts for all L1 messages and update the total."""
        total = 0
        for msg in state.l1_working.messages:
            msg.token_count = self.count_message(msg)
            total += msg.token_count
        state.l1_working.token_count = total
