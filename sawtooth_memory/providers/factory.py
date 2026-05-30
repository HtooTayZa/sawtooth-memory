"""
providers/factory.py — Instantiates the correct ProviderAdapter for a CloudConfig.
"""

from __future__ import annotations

from sawtooth_memory.config import CloudConfig, Provider

from .adapters import AnthropicAdapter, GeminiAdapter, OpenAIAdapter, ProviderAdapter


def get_adapter(config: CloudConfig) -> ProviderAdapter:
    """
    Return the correct ProviderAdapter instance for the given CloudConfig.

    Raises:
        ValueError: if ``config.provider`` is an unsupported value.
    """
    base_url = config.base_url

    if config.provider == Provider.OPENAI:
        return OpenAIAdapter(base_url=base_url)

    if config.provider == Provider.ANTHROPIC:
        return AnthropicAdapter(base_url=base_url)

    if config.provider == Provider.GEMINI:
        return GeminiAdapter(model=config.model, base_url=base_url)

    raise ValueError(
        f"Unsupported provider: {config.provider!r}. "
        f"Valid choices: {[p.value for p in Provider]}"
    )
