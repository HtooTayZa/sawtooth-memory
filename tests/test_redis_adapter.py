"""
tests/test_redis_adapter.py

Validates the RedisStorageAdapter serialization, deserialization, and
state management using asynchronous mocks.
"""

import pytest
from unittest.mock import AsyncMock, patch

from sawtooth_memory.state import MemoryState, SystemPrompt
from sawtooth_memory.storage.redis_adapter import RedisStorageAdapter


@pytest.mark.asyncio
@patch("redis.asyncio.from_url")
async def test_redis_save_load_delete_lifecycle(mock_from_url):
    """Verify that the MemoryState accurately converts to JSON and back."""
    # 1. Setup the mocked Redis client
    mock_client = AsyncMock()
    mock_from_url.return_value = mock_client

    adapter = RedisStorageAdapter(redis_url="redis://fake:6379", ttl_seconds=3600)
    session_id = "user_alpha_99"
    expected_key = "sawtooth:session:user_alpha_99"

    # 2. Create a fresh memory state
    original_state = MemoryState(
        l0_system=SystemPrompt(content="You are a distributed agent.")
    )
    # --- TEST SAVE ---
    await adapter.save_state(session_id, original_state)

    # Verify Redis SETEX was called with the correct key and TTL
    mock_client.setex.assert_called_once()
    called_args = mock_client.setex.call_args[0]
    assert called_args[0] == expected_key
    assert called_args[1] == 3600

    # Extract the JSON payload sent to Redis
    saved_json_payload = called_args[2]
    assert "You are a distributed agent." in saved_json_payload

    # --- TEST LOAD ---
    # Simulate Redis returning the JSON string when GET is called
    mock_client.get.return_value = saved_json_payload

    loaded_state = await adapter.load_state(session_id)

    # Verify the Pydantic model hydrated perfectly
    assert loaded_state is not None
    assert loaded_state.l0_system.content == "You are a distributed agent."
    assert loaded_state.l1_working.messages == []

    # --- TEST DELETE ---
    await adapter.delete_state(session_id)
    mock_client.delete.assert_called_once_with(expected_key)
