"""
worker.py — Async background compression worker.

Runs as an asyncio Task. Pulls CompressionTasks from a queue, calls the
Ollama compressor, then merges results into the MemoryState — all without
blocking the main agent thread.

Graceful degradation: if Ollama is unavailable, appends a truncation note
to L2 instead of crashing.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from typing import Union
from .compressor import OllamaCompressor, CloudCompressor
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
    """Flatten a message list to a readable string for the compressor."""
    parts = []
    for msg in messages:
        parts.append(f"{msg.role.upper()}: {msg.content}")
    return "\n\n".join(parts)


class CompressionWorker:
    """
    Background asyncio worker that processes compression tasks off the
    critical path.

    Lifecycle:
        worker = CompressionWorker(compressor, fallback_truncate=True)
        await worker.start()
        worker.enqueue(task)          # non-blocking
        await worker.stop()           # drains queue then exits
    """

    def __init__(
        self,
        compressor: Union[OllamaCompressor, CloudCompressor],
        fallback_truncate: bool = True,
    ) -> None:
        self._compressor = compressor
        self._fallback_truncate = fallback_truncate

        self._queue: asyncio.Queue[CompressionTask | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._running: bool = False
        self._processed: int = 0
        self._failed: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._loop(), name="sawtooth-compression-worker"
        )
        logger.info("CompressionWorker: started.")

    async def stop(self) -> None:
        """
        Signal the worker to stop after draining the queue.
        Waits for in-flight compression to finish before returning.
        """
        if not self._running:
            return
        self._running = False
        await self._queue.put(_SENTINEL)
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=120)
            except asyncio.TimeoutError:
                logger.warning(
                    "CompressionWorker: shutdown timed out, cancelling task."
                )
                self._task.cancel()
        await self._compressor.close()
        logger.info(
            f"CompressionWorker: stopped. "
            f"Processed={self._processed}, Failed={self._failed}"
        )

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(self, task: CompressionTask) -> None:
        """Put a task on the queue. Returns immediately; does not block."""
        self._queue.put_nowait(task)
        logger.debug(
            f"CompressionWorker: enqueued {len(task.messages)} messages. "
            f"Queue depth: {self._queue.qsize()}"
        )

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while True:
            task = await self._queue.get()
            if task is _SENTINEL:
                self._queue.task_done()
                break
            try:
                await self._process(task)
                self._processed += 1
            except Exception as exc:
                self._failed += 1
                logger.error(
                    f"CompressionWorker: unhandled error processing task: {exc}",
                    exc_info=True,
                )
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------
    # Task processing
    # ------------------------------------------------------------------

    async def _process(self, task: CompressionTask) -> None:
        state = task.state
        messages_text = _messages_to_text(task.messages)

        try:
            result = await self._compressor.compress(messages_text)
            self._merge(state, result)
            logger.info(
                f"CompressionWorker: compressed {len(task.messages)} messages → "
                f"narrative ({len(result['narrative_summary'])} chars), "
                f"{len(result['extracted_entities'])} entities extracted."
            )
        except (OllamaConnectionError, CompressionError) as exc:
            logger.warning(
                f"CompressionWorker: compression failed ({exc}). "
                f"fallback_truncate={self._fallback_truncate}"
            )
            if self._fallback_truncate:
                self._fallback_merge(state, task.messages)
            else:
                raise

    # ------------------------------------------------------------------
    # State merging
    # ------------------------------------------------------------------

    def _merge(self, state: MemoryState, result: dict) -> None:
        narrative = result.get("narrative_summary", "").strip()
        entities = result.get("extracted_entities", {})

        if narrative:
            state.l2_archival.append_narrative(narrative)
            logger.debug("CompressionWorker: appended narrative to L2.")

        if entities:
            state.l1_5_entities.upsert(entities)
            logger.debug(
                f"CompressionWorker: upserted {len(entities)} entities into L1.5."
            )

    def _fallback_merge(self, state: MemoryState, messages: list[Message]) -> None:
        note = (
            f"[COMPRESSION UNAVAILABLE: {len(messages)} messages were truncated. "
            f"First message role: {messages[0].role if messages else 'unknown'}]"
        )
        state.l2_archival.append_narrative(note)
        logger.warning("CompressionWorker: fallback truncation note written to L2.")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        return {
            "processed": self._processed,
            "failed": self._failed,
            "queue_depth": self.queue_depth,
            "running": self._running,
        }
