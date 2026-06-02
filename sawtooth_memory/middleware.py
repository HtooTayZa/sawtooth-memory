"""
middleware.py — ContextManager: the primary public API for Sawtooth-Memory.

Drop this between your agent loop and your LLM API call.

Quick start:
    from sawtooth_memory import ContextManager, ContextManagerConfig

    config = ContextManagerConfig(soft_limit_tokens=3000)

    async with ContextManager("You are a data analysis agent.", config) as cm:
        await cm.add_message("user", "Analyse Q3 revenue.")
        await cm.add_message("assistant", "Connecting to the database...")

        messages = cm.build_prompt()
        # response = await openai_client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=messages,
        # )
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional, Literal
from .config import ContextManagerConfig
from .compressor import CloudCompressor, OllamaCompressor
from .exceptions import TokenLimitExceededError
from .monitor import TokenMonitor
from .state import (
    ArchivalMemory,
    EntityLedger,
    MemoryState,
    Message,
    MessageRole,
    SystemPrompt,
    WorkingMemory,
)
from .worker import CompressionWorker, CompressionTask
from .events.bus import EventBus, get_event_bus
from .events.types import (
    CompressionCycleStartEvent,
    EntityAnchoredEvent,
    L1EvictionEvent,
)
from .journal import AsyncCompressionJournal

logger = logging.getLogger(__name__)


class ContextManager:
    def __init__(
        self,
        system_prompt: str,
        config: ContextManagerConfig | None = None,
        *,
        enable_events: bool = True,
        journal_path: Optional[Path] = None,
    ) -> None:
        self._config = config or ContextManagerConfig()
        self._enable_events = enable_events

        # 1. Initialize Event Bus and Journal FIRST
        self._event_bus: Optional[EventBus] = None
        self._journal: Optional[AsyncCompressionJournal] = None

        if self._enable_events:
            self._event_bus = get_event_bus()

            # Localize the journal instance to this ContextManager
            j_path = journal_path or Path("./sawtooth_compression_journal.jsonl")
            self._journal = AsyncCompressionJournal(j_path)

            from .events.handlers import make_journal_handler

            # Assign to a variable first so Ruff doesn't wrap the line
            handler = make_journal_handler(self._journal)
            self._event_bus.subscribe("compression.cycle_complete", handler)  # type: ignore[arg-type]

        # 2. Token monitor now receives the initialized event_bus
        self._monitor = TokenMonitor(
            model=self._config.tokenizer_model,
            soft_limit=self._config.soft_limit_tokens,
            hard_limit=self._config.hard_limit_tokens,
            event_bus=self._event_bus,
        )

        sp_tokens = self._monitor.count_text(system_prompt)
        self._state = MemoryState(
            l0_system=SystemPrompt(content=system_prompt, token_count=sp_tokens),
            l1_working=WorkingMemory(),
            l1_5_entities=EntityLedger(),
            l2_archival=ArchivalMemory(),
        )

        # 3. Bind the Entity Ledger telemetry callback
        if self._enable_events and self._event_bus:

            def handle_ledger_mutation(key: str, value: str, op: str):
                # 3. Bind the Entity Ledger telemetry callback
                if self._enable_events and self._event_bus:

                    def handle_ledger_mutation(
                        key: str, value: str, op: Literal["insert", "update", "delete"]
                    ):
                        if self._event_bus:
                            event = EntityAnchoredEvent(
                                entity_key=key, entity_value=value, operation=op
                            )
                            asyncio.create_task(self._event_bus.emit(event))

            self._state.l1_5_entities.set_event_callback(handle_ledger_mutation)

        # 4. Compression backend & Worker
        self._compressor: CloudCompressor | OllamaCompressor
        if self._config.cloud:
            self._compressor = CloudCompressor(self._config.cloud)
        else:
            self._compressor = OllamaCompressor(self._config.ollama)

        self._worker = CompressionWorker(
            compressor=self._compressor,
            fallback_truncate=self._config.fallback_truncate,
            event_bus=self._event_bus,
            journal=self._journal,
        )

        logger.debug(
            f"ContextManager initialised. "
            f"soft_limit={self._config.soft_limit_tokens}, "
            f"hard_limit={self._config.hard_limit_tokens}, "
            f"chunk_size={self._config.chunk_size}, "
            f"events_enabled={self._enable_events}"
        )

    async def start(self) -> None:
        if self._enable_events and self._journal:
            await self._journal.start()
        await self._worker.start()

    async def stop(self) -> None:
        await self._worker.stop()
        if self._enable_events and self._journal:
            await self._journal.stop()  # Stops just this agent's journal instance

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

        If the soft token limit is crossed after adding this message,
        the oldest chunk_size messages are sliced off and enqueued for
        background compression — without blocking this call.

        If the hard limit is crossed and fallback_truncate is False,
        raises TokenLimitExceededError.

        When events are enabled, this method emits L1 eviction events
        and compression start events.
        """
        msg = Message(role=role, content=content)
        msg.token_count = self._monitor.count_message(msg)
        self._state.l1_working.append(msg)

        logger.debug(
            f"add_message: role={role}, tokens={msg.token_count}, "
            f"l1_total={self._state.l1_working.token_count}"
        )

        # Hard limit check (immediate action, no background)
        if self._monitor.exceeds_hard_limit(self._state):
            if not self._config.fallback_truncate:
                raise TokenLimitExceededError(
                    f"Working Memory exceeded hard limit of "
                    f"{self._config.hard_limit_tokens} tokens and "
                    f"fallback_truncate is disabled."
                )
            logger.warning(
                "Hard token limit reached before compression finished. "
                "Forcing immediate truncation of oldest messages."
            )
            self._force_truncate()

        # Soft limit check → trigger async compression
        elif self._monitor.exceeds_soft_limit(self._state):
            await self._trigger_compression()

    def build_prompt(self) -> list[dict[str, str]]:
        """
        Compile all memory tiers into an OpenAI-compatible messages list.

        Returns a list of {"role": "...", "content": "..."} dicts, ready
        to pass directly to openai.chat.completions.create() or equivalent.

        Structure of the injected system message:
            [SYSTEM_L0]
            <system prompt>

            [ARCHIVE_L2]          (omitted if empty)
            <compressed history narrative>

            [ENTITY_LEDGER_L1_5]  (omitted if empty)
            <json key-value pairs>

        Followed by raw Working Memory (L1) messages.
        """
        state = self._state
        system_parts: list[str] = []

        system_parts.append(f"[SYSTEM_L0]\n{state.l0_system.content}")

        if state.l2_archival.narrative.strip():
            system_parts.append(f"[ARCHIVE_L2]\n{state.l2_archival.narrative.strip()}")

        if state.l1_5_entities.entities:
            system_parts.append(
                f"[ENTITY_LEDGER_L1_5]\n{state.l1_5_entities.to_json_str()}"
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": "\n\n".join(system_parts)}
        ]

        for msg in state.l1_working.messages:
            messages.append(msg.to_openai_dict())

        return messages

    # ------------------------------------------------------------------
    # Internal compression triggers
    # ------------------------------------------------------------------

    async def _trigger_compression(self) -> None:
        """
        Non-blocking: slice the oldest chunk and hand it off to the worker.
        The main thread continues running immediately.

        If events are enabled, emits a CompressionCycleStartEvent and
        (later in the worker) the completion/failure events.
        """
        chunk = self._state.l1_working.slice_oldest(self._config.chunk_size)
        if not chunk:
            return

        # Generate a unique cycle ID for this compression run
        import uuid

        cycle_id = str(uuid.uuid4())

        # Emit start event if bus exists
        if self._event_bus:
            await self._event_bus.emit(
                CompressionCycleStartEvent(
                    cycle_id=cycle_id,
                    current_l1_tokens=self._state.l1_working.token_count,
                    chunk_size=self._config.chunk_size,
                    session_id=None,  # can be extended later
                )
            )

        # Emit L1 eviction event (this compression will evict these messages)
        if self._event_bus:
            evicted_tokens = sum(m.token_count for m in chunk)
            await self._event_bus.emit(
                L1EvictionEvent(
                    tokens_evicted=evicted_tokens,
                    messages_evicted=len(chunk),
                    tokens_remaining_l1=self._state.l1_working.token_count,
                    evicted_message_ids=[m.id for m in chunk],
                    trigger="soft_limit_exceeded",
                    session_id=None,
                    cycle_id=cycle_id,  # link to compression cycle
                )
            )

        # Create task with cycle_id so worker can correlate events
        task = CompressionTask(
            messages=chunk,
            state=self._state,
            cycle_id=cycle_id,
        )
        self._worker.enqueue(task)

        logger.info(
            f"Compression triggered: offloaded {len(chunk)} messages to worker. "
            f"L1 remaining: {self._state.l1_working.token_count} tokens, "
            f"cycle_id={cycle_id}"
        )

    def _force_truncate(self) -> None:
        """
        Hard-limit fallback: discard the oldest messages immediately on
        the main thread without waiting for Ollama/Cloud.

        Note: This does NOT emit events because it's a fallback path
        and not a full compression cycle. The journal remains
        unaffected (only complete cycles are logged).
        """
        chunk = self._state.l1_working.slice_oldest(self._config.chunk_size)
        note = (
            f"[HARD TRUNCATION: {len(chunk)} messages dropped because the "
            f"compression worker has not yet caught up.]"
        )
        self._state.l2_archival.append_narrative(note)
        logger.warning(f"Hard truncation: dropped {len(chunk)} messages from L1.")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def state(self) -> MemoryState:
        """Read-only access to the current MemoryState."""
        return self._state

    def get_stats(self) -> dict:
        """
        Return a snapshot of current token usage and worker health.

        Returns:
            {
                "l0_tokens": int,
                "l1_tokens": int,
                "l1_message_count": int,
                "l1_5_entity_count": int,
                "l2_tokens": int,
                "worker": {"processed": int, "failed": int, "queue_depth": int, ...}
            }
        """
        return {
            "l0_tokens": self._state.l0_system.token_count,
            "l1_tokens": self._state.l1_working.token_count,
            "l1_message_count": len(self._state.l1_working.messages),
            "l1_5_entity_count": len(self._state.l1_5_entities.entities),
            "l2_tokens": self._monitor.count_text(self._state.l2_archival.narrative),
            "worker": self._worker.stats,
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"<ContextManager "
            f"l1={stats['l1_tokens']}/{self._config.soft_limit_tokens} tokens, "
            f"l1_msgs={stats['l1_message_count']}, "
            f"l2_tokens={stats['l2_tokens']}, "
            f"entities={stats['l1_5_entity_count']}, "
            f"queue={stats['worker']['queue_depth']}>"
        )

    async def health_check(self) -> dict[str, Any]:
        """
        Verifies runtime configurations and basic initialization readiness.
        Returns a diagnostic report dictionary. Raises ValueError on broken configurations.
        """
        report: dict[str, Any] = {"status": "healthy", "checks": {}}

        # 1. Validate Token Configurations
        if self._config.soft_limit_tokens >= self._config.hard_limit_tokens:
            report["status"] = "unhealthy"
            raise ValueError(
                f"Configuration Error: soft_limit_tokens ({self._config.soft_limit_tokens}) "
                f"must be strictly less than hard_limit_tokens ({self._config.hard_limit_tokens})."
            )
        report["checks"]["configuration"] = "OK"

        # 2. Verify Background Worker State
        if getattr(self, "_worker", None) and self._worker._running:
            report["checks"]["worker_status"] = "RUNNING"
        else:
            report["checks"]["worker_status"] = "STOPPED"

        # 3. (Optional) Check event bus and journal health
        if self._enable_events:
            report["checks"]["events"] = "ENABLED"
            if self._journal:
                report["checks"]["journal_path"] = str(self._journal.path)
            else:
                report["checks"]["journal"] = "NOT_INITIALIZED"
        else:
            report["checks"]["events"] = "DISABLED"

        return report
