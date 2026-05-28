"""tests/test_monitor.py — Unit tests for TokenMonitor."""

import pytest

from sawtooth_memory.monitor import TokenMonitor
from sawtooth_memory.state import MemoryState, Message, SystemPrompt, WorkingMemory


@pytest.fixture
def monitor():
    return TokenMonitor(model="gpt-4o", soft_limit=100, hard_limit=200)


class TestTokenCounting:
    def test_count_text_nonempty(self, monitor):
        count = monitor.count_text("Hello, world!")
        assert count > 0

    def test_count_text_empty(self, monitor):
        assert monitor.count_text("") == 0

    def test_count_message_includes_overhead(self, monitor):
        msg = Message(role="user", content="Hi")
        count = monitor.count_message(msg)
        text_only = monitor.count_text("Hi")
        assert count == text_only + 4

    def test_longer_content_more_tokens(self, monitor):
        short = monitor.count_text("Hi")
        long = monitor.count_text("Hi " * 50)
        assert long > short


class TestThresholds:
    def test_exceeds_soft_limit_false(self, monitor):
        state = MemoryState(l0_system=SystemPrompt(content="test"))
        state.l1_working.token_count = 50
        assert not monitor.exceeds_soft_limit(state)

    def test_exceeds_soft_limit_true(self, monitor):
        state = MemoryState(l0_system=SystemPrompt(content="test"))
        state.l1_working.token_count = 100
        assert monitor.exceeds_soft_limit(state)

    def test_exceeds_hard_limit_false(self, monitor):
        state = MemoryState(l0_system=SystemPrompt(content="test"))
        state.l1_working.token_count = 150
        assert not monitor.exceeds_hard_limit(state)

    def test_exceeds_hard_limit_true(self, monitor):
        state = MemoryState(l0_system=SystemPrompt(content="test"))
        state.l1_working.token_count = 200
        assert monitor.exceeds_hard_limit(state)


class TestRecount:
    def test_recount_working_memory(self, monitor):
        state = MemoryState(l0_system=SystemPrompt(content="test"))
        state.l1_working.messages = [
            Message(role="user", content="Hello", token_count=999),
            Message(role="assistant", content="Hi there", token_count=999),
        ]
        state.l1_working.token_count = 999

        monitor.recount_working_memory(state)

        expected = sum(
            monitor.count_message(m) for m in state.l1_working.messages
        )
        assert state.l1_working.token_count == expected
        assert state.l1_working.token_count != 999
