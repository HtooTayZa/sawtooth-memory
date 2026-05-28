"""
state.py — Core data models for Sawtooth-Memory (MVP).

Tiers:
  L0  — SystemPrompt  : Immutable agent persona.
  L1  — WorkingMemory : Sliding window of raw messages.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """A single turn in Working Memory (L1)."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    token_count: int = 0

    def to_openai_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class SystemPrompt(BaseModel):
    """L0 — Immutable system prompt. Set once, never modified."""

    content: str
    token_count: int = 0


class WorkingMemory(BaseModel):
    """L1 — Sliding window of recent raw messages."""

    messages: list[Message] = Field(default_factory=list)
    token_count: int = 0

    def append(self, message: Message) -> None:
        self.messages.append(message)
        self.token_count += message.token_count

    def slice_oldest(self, n: int) -> list[Message]:
        """Remove and return the oldest n messages, updating token count."""
        chunk = self.messages[:n]
        self.messages = self.messages[n:]
        self.token_count = sum(m.token_count for m in self.messages)
        return chunk


class MemoryState(BaseModel):
    """Root state object."""

    l0_system: SystemPrompt
    l1_working: WorkingMemory = Field(default_factory=WorkingMemory)
