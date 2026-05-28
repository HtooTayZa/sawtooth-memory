"""
exceptions.py — Sawtooth-Memory exception hierarchy.

SawtoothError
├── CompressionError          # Generic compression pipeline failure
│   ├── OllamaConnectionError # Ollama is unreachable
│   └── MalformedOutputError  # Local model returned unparseable JSON
└── TokenLimitExceededError   # Hard cap reached, cannot proceed safely
"""


class SawtoothError(Exception):
    """Base class for all Sawtooth-Memory errors."""


class CompressionError(SawtoothError):
    """Raised when the background compression pipeline fails."""


class OllamaConnectionError(CompressionError):
    """Raised when the Ollama backend cannot be reached."""


class MalformedOutputError(CompressionError):
    """Raised when the local model returns output that cannot be parsed."""


class TokenLimitExceededError(SawtoothError):
    """
    Raised when Working Memory exceeds the hard cap and graceful
    degradation is disabled (fallback_truncate=False).
    """
