"""
config.py — Configuration for Sawtooth-Memory (MVP).
"""

from __future__ import annotations

from pydantic import BaseModel


class ContextManagerConfig(BaseModel):
    """Top-level configuration for the ContextManager middleware."""

    # When L1 Working Memory exceeds this, slice the oldest chunk.
    soft_limit_tokens: int = 3000

    # Hard cap: force immediate truncation if exceeded.
    hard_limit_tokens: int = 6000

    # Number of messages to drop per truncation pass.
    chunk_size: int = 10
