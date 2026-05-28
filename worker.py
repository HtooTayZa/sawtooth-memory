"""
worker.py — Async background compression worker.

Runs as an asyncio Task. Pulls CompressionTasks from a queue, calls the
Ollama compressor, then merges results into MemoryState without blocking
the main agent thread.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .compressor import OllamaCompressor
from .exceptions import CompressionError, OllamaConnectionError
from .state import MemoryState, Message

logger = logging.getLogger(__name__)

_SENTINEL = None


@dataclass
class CompressionTask:
    """A chunk of messages queued for background compression."""

    messages: list[Message]
    state: MemoryState


def _messages_to_text(messages: list[Message]) -> str:
    parts = []
    for msg in messages:
        parts.append(f"{msg.role.upper()}: {msg.content}")
    return "\n\n".join(parts)


class CompressionWorker:
    """
    Background asyncio worker that processes compression tasks off the
    critical path.
    """

    def __init__(
        self,
        compressor: OllamaCompressor,
        fallback_truncate: bool = True,
    ) -> None:
        self._compressor = compressor
        self._fallback_truncate = fallback_truncate
        self._queue: asyncio.Queue[CompressionTask | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running: bool = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._loop(), name="sawtooth-compression-worker"
        )
        logger.info("CompressionWorker: started.")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        await self._queue.put(_SENTINEL)
        if self._task:
            await self._task
        await self._compressor.close()
        logger.info("CompressionWorker: stopped.")

    def enqueue(self, task: CompressionTask) -> None:
        self._queue.put_nowait(task)
        logger.debug(
            f"CompressionWorker: enqueued {len(task.messages)} messages. "
            f"Queue depth: {self._queue.qsize()}"
        )

    async def _loop(self) -> None:
        while True:
            task = await self._queue.get()
            if task is _SENTINEL:
                self._queue.task_done()
                break
            try:
                await self._process(task)
            except Exception as exc:
                logger.error(
                    f"CompressionWorker: unhandled error: {exc}", exc_info=True
                )
            finally:
                self._queue.task_done()

    async def _process(self, task: CompressionTask) -> None:
        state = task.state
        messages_text = _messages_to_text(task.messages)

        try:
            result = await self._compressor.compress(messages_text)
            self._merge(state, result)
            logger.info(
                f"CompressionWorker: compressed {len(task.messages)} messages."
            )
        except (OllamaConnectionError, CompressionError) as exc:
            logger.warning(f"CompressionWorker: compression failed ({exc}).")
            if self._fallback_truncate:
                self._fallback_merge(state, task.messages)
            else:
                raise

    def _merge(self, state: MemoryState, result: dict) -> None:
        narrative = result.get("narrative_summary", "").strip()
        entities = result.get("extracted_entities", {})

        if narrative:
            state.l2_archival.append_narrative(narrative)

        if entities:
            state.l1_5_entities.upsert(entities)

    def _fallback_merge(self, state: MemoryState, messages: list[Message]) -> None:
        note = (
            f"[COMPRESSION UNAVAILABLE: {len(messages)} messages were truncated. "
            f"First message role: {messages[0].role if messages else 'unknown'}]"
        )
        state.l2_archival.append_narrative(note)

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        return {
            "queue_depth": self.queue_depth,
            "running": self._running,
        }
