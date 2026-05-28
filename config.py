"""
config.py — Configuration models for Sawtooth-Memory.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    """Connection settings for the local Ollama compression backend."""

    base_url: str = "http://localhost:11434"
    model: str = "phi4"
    timeout_seconds: int = 90


class ContextManagerConfig(BaseModel):
    """Top-level configuration for the ContextManager middleware."""

    soft_limit_tokens: int = 3000
    hard_limit_tokens: int = 6000
    chunk_size: int = 10
    tokenizer_model: str = "gpt-4o"
    fallback_truncate: bool = True

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
