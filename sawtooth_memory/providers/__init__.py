"""
providers — Cloud provider adapters for the Sawtooth-Memory compression layer.

Each adapter implements the ProviderAdapter Protocol and knows how to:
  1. Build the correct API endpoint URL.
  2. Construct provider-specific auth headers.
  3. Construct the exact JSON payload, enforcing structured/JSON-mode output.
  4. Parse the raw response back into (dict, total_tokens).

Supported providers
-------------------
- OpenAI   → OpenAIAdapter
- Anthropic → AnthropicAdapter
- Gemini   → GeminiAdapter

Usage
-----
    from sawtooth_memory.providers import get_adapter
    from sawtooth_memory.config import CloudConfig, Provider

    cfg = CloudConfig(provider=Provider.OPENAI, model="gpt-4o-mini", api_key="sk-...")
    adapter = get_adapter(cfg)
"""

from .adapters import AnthropicAdapter, GeminiAdapter, OpenAIAdapter, ProviderAdapter
from .factory import get_adapter

__all__ = [
    "ProviderAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "get_adapter",
]
