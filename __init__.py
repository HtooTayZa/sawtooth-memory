"""
Sawtooth-Memory — Async context manager middleware for LLM agents.

MVP: two-tier memory (L0 system prompt + L1 working memory) with
naive truncation when the token soft limit is exceeded.

Public API:
    ContextManager       — Main middleware class
    ContextManagerConfig — Configuration (token limits)

Example:
    from sawtooth_memory import ContextManager, ContextManagerConfig

    config = ContextManagerConfig(soft_limit_tokens=3000)

    async with ContextManager("You are a helpful agent.", config) as cm:
        await cm.add_message("user", "What is 2 + 2?")
        messages = cm.build_prompt()
        # Pass `messages` to your LLM SDK
"""

from .config import ContextManagerConfig
from .middleware import ContextManager
from .state import MemoryState

__all__ = [
    "ContextManager",
    "ContextManagerConfig",
    "MemoryState",
]

__version__ = "0.1.0"
