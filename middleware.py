"""
middleware.py — ContextManager: MVP implementation.

Maintains a two-tier memory (L0 system prompt + L1 working memory).
Token counting uses a simple word-based approximation.
No background compression — exceeding the soft limit drops the oldest
chunk immediately on the main thread.

Usage:
    async with ContextManager("You are a helpful agent.") as cm:
        await cm.add_message("user", "Hello")
        messages = cm.build_prompt()
"""

from __future__ import annotations

import logging

from .config import ContextManagerConfig
from .state import MemoryState, Message, MessageRole, SystemPrompt, WorkingMemory

logger = logging.getLogger(__name__)

# Approximate tokens per word for the word-count fallback (~1.3 is typical).
_WORDS_PER_TOKEN = 1.3
# Per-message role/separator overhead in tokens.
_MESSAGE_OVERHEAD = 4


def _count_tokens(text: str) -> int:
    """Approximate token count via word count."""
    return max(1, int(len(text.split()) * _WORDS_PER_TOKEN)) if text.strip() else 0


class ContextManager:
    """
    Drop-in middleware between your agent loop and your LLM API call.

    Maintains L0 (system prompt) and L1 (working memory). When L1 exceeds
    the soft limit, the oldest chunk is synchronously discarded to keep
    prompt length in check.
    """

    def __init__(
        self,
        system_prompt: str,
        config: ContextManagerConfig | None = None,
    ) -> None:
        self._config = config or ContextManagerConfig()

        sp_tokens = _count_tokens(system_prompt)
        self._state = MemoryState(
            l0_system=SystemPrompt(content=system_prompt, token_count=sp_tokens),
            l1_working=WorkingMemory(),
        )

        logger.debug(
            f"ContextManager initialised. "
            f"soft_limit={self._config.soft_limit_tokens}, "
            f"hard_limit={self._config.hard_limit_tokens}"
        )

    # ------------------------------------------------------------------
    # Lifecycle (no-ops in MVP; reserved for future async worker)
    # ------------------------------------------------------------------

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def __aenter__(self) -> "ContextManager":
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def add_message(self, role: MessageRole, content: str) -> None:
        """
        Add a message to Working Memory (L1).

        If the soft token limit is crossed, the oldest chunk_size messages
        are discarded immediately to keep memory within bounds.
        """
        token_count = _count_tokens(content) + _MESSAGE_OVERHEAD
        msg = Message(role=role, content=content, token_count=token_count)
        self._state.l1_working.append(msg)

        logger.debug(
            f"add_message: role={role}, tokens={token_count}, "
            f"l1_total={self._state.l1_working.token_count}"
        )

        if self._state.l1_working.token_count >= self._config.soft_limit_tokens:
            self._truncate()

    def build_prompt(self) -> list[dict[str, str]]:
        """
        Compile memory tiers into an OpenAI-compatible messages list.

        Returns a list of {"role": "...", "content": "..."} dicts ready
        to pass directly to your LLM SDK.
        """
        system_content = f"[SYSTEM_L0]\n{self._state.l0_system.content}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        for msg in self._state.l1_working.messages:
            messages.append(msg.to_openai_dict())

        return messages

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _truncate(self) -> None:
        """Drop the oldest chunk to bring L1 back under the soft limit."""
        chunk = self._state.l1_working.slice_oldest(self._config.chunk_size)
        logger.warning(f"Soft limit reached: dropped {len(chunk)} oldest messages.")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def state(self) -> MemoryState:
        return self._state

    def get_stats(self) -> dict:
        return {
            "l0_tokens": self._state.l0_system.token_count,
            "l1_tokens": self._state.l1_working.token_count,
            "l1_message_count": len(self._state.l1_working.messages),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"<ContextManager "
            f"l1={stats['l1_tokens']}/{self._config.soft_limit_tokens} tokens, "
            f"l1_msgs={stats['l1_message_count']}>"
        )
