"""
config.py — Configuration models for Sawtooth-Memory.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, SecretStr


# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------


class Provider(str, Enum):
    """Supported cloud LLM providers for the compression backend."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


# ---------------------------------------------------------------------------
# Backend configs
# ---------------------------------------------------------------------------


class OllamaConfig(BaseModel):
    """Connection settings for the local Ollama compression backend."""

    base_url: str = "http://localhost:11434"
    model: str = "phi4-mini:latest"
    timeout_seconds: int = 90


class CloudConfig(BaseModel):
    """
    Connection settings for a cloud LLM compression backend.

    Supports OpenAI, Anthropic, and Gemini via their respective APIs.
    Use ``base_url`` to route traffic through proxies like Helicone,
    LiteLLM, or Azure OpenAI without changing provider-specific payload
    construction.

    Example::

        from sawtooth_memory.config import CloudConfig, Provider

        cfg = CloudConfig(
            provider=Provider.ANTHROPIC,
            model="claude-3-5-haiku-latest",
            api_key="sk-ant-...",
        )
    """

    provider: Provider
    model: str
    api_key: SecretStr
    base_url: Optional[str] = None
    timeout_seconds: int = 60


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class ContextManagerConfig(BaseModel):
    soft_limit_tokens: int = Field(
        default=1000,
        description="Trigger compression when L1 tokens exceed this soft limit.",
    )
    hard_limit_tokens: int = Field(
        default=2500,
        description="Failsafe limit to hard-truncate older L1 messages if compression is too slow.",
    )
    chunk_size: int = Field(
        default=4,
        description="Number of oldest L1 messages to summarize and evict per compression cycle.",
    )

    fallback_truncate: bool = Field(
        default=True,
        description="Whether to aggressively discard messages if hard_limit is reached.",
    )

    # NEW: Turn-based batching
    max_unsummarized_turns: Optional[int] = Field(
        default=None,
        description="Trigger compression if unsummarized L1 messages reach this count, acting as a turn-based batching threshold.",
    )

    tokenizer_model: str = Field(
        default="gpt-4o",
        description="Tokenizer encoding to use for precise context monitoring.",
    )
    journal_path: str = Field(
        default=".sawtooth_journal.jsonl",
        description="Path to the JSONL auditing journal.",
    )

    ollama: Optional[OllamaConfig] = None
    cloud: Optional[CloudConfig] = None
