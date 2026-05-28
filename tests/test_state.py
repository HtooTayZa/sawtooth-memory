"""tests/test_state.py — Unit tests for Pydantic state schemas."""

import pytest
from pydantic import ValidationError

from sawtooth_memory.state import (
    ArchivalMemory,
    EntityLedger,
    MemoryState,
    Message,
    SystemPrompt,
    WorkingMemory,
)


class TestMessage:
    def test_valid_roles(self):
        for role in ["user", "assistant", "system", "tool"]:
            msg = Message(role=role, content="hello")
            assert msg.role == role

    def test_invalid_role_raises(self):
        with pytest.raises(ValidationError):
            Message(role="bot", content="hello")

    def test_to_openai_dict(self):
        msg = Message(role="user", content="Hi")
        d = msg.to_openai_dict()
        assert d == {"role": "user", "content": "Hi"}

    def test_default_token_count(self):
        msg = Message(role="user", content="test")
        assert msg.token_count == 0


class TestWorkingMemory:
    def test_append_updates_token_count(self):
        wm = WorkingMemory()
        msg = Message(role="user", content="hello", token_count=10)
        wm.append(msg)
        assert wm.token_count == 10
        assert len(wm.messages) == 1

    def test_slice_oldest_removes_and_recounts(self):
        wm = WorkingMemory()
        msgs = [Message(role="user", content=f"msg{i}", token_count=5) for i in range(5)]
        for m in msgs:
            wm.append(m)

        assert wm.token_count == 25

        chunk = wm.slice_oldest(3)
        assert len(chunk) == 3
        assert len(wm.messages) == 2
        assert wm.token_count == 10

    def test_slice_more_than_available(self):
        wm = WorkingMemory()
        wm.append(Message(role="user", content="only one", token_count=5))
        chunk = wm.slice_oldest(10)
        assert len(chunk) == 1
        assert wm.token_count == 0


class TestEntityLedger:
    def test_upsert_merges(self):
        ledger = EntityLedger()
        ledger.upsert({"db_id": "abc123"})
        ledger.upsert({"file": "/tmp/out.txt"})
        assert ledger.entities == {"db_id": "abc123", "file": "/tmp/out.txt"}

    def test_upsert_overwrites(self):
        ledger = EntityLedger()
        ledger.upsert({"key": "old"})
        ledger.upsert({"key": "new"})
        assert ledger.entities["key"] == "new"

    def test_to_json_str(self):
        import json
        ledger = EntityLedger(entities={"x": "1"})
        parsed = json.loads(ledger.to_json_str())
        assert parsed == {"x": "1"}


class TestArchivalMemory:
    def test_append_narrative_first(self):
        arch = ArchivalMemory()
        arch.append_narrative("First note.")
        assert arch.narrative == "First note."

    def test_append_narrative_subsequent(self):
        arch = ArchivalMemory()
        arch.append_narrative("First.")
        arch.append_narrative("Second.")
        assert "First." in arch.narrative
        assert "Second." in arch.narrative

    def test_append_empty_string_noop(self):
        arch = ArchivalMemory()
        arch.append_narrative("   ")
        assert arch.narrative == ""


class TestMemoryState:
    def test_default_empty_tiers(self):
        state = MemoryState(l0_system=SystemPrompt(content="You are helpful."))
        assert state.l1_working.messages == []
        assert state.l1_5_entities.entities == {}
        assert state.l2_archival.narrative == ""

    def test_l0_content(self):
        state = MemoryState(l0_system=SystemPrompt(content="Agent persona."))
        assert state.l0_system.content == "Agent persona."
