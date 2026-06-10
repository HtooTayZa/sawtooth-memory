"""
sawtooth_memory/ner.py
Deterministic, local-first Named Entity Recognition pipeline.
Runs entirely in-process with pre-compiled patterns for zero-latency extraction.
"""

from __future__ import annotations

import contextvars
import re
from typing import Dict, Protocol

# Ambient context channel for telemetry tracking across async boundaries
active_strategy_context: contextvars.ContextVar[Dict[str, str]] = (
    contextvars.ContextVar("active_strategy_context", default={})
)


class DeterministicExtractor(Protocol):
    """Protocol for any deterministic value extractor."""

    def extract(self, text: str) -> Dict[str, str]: ...


class RegexEntityExtractor:
    """High-performance regex matcher with configurable patterns."""

    _DEFAULT_PATTERNS: Dict[str, str] = {
        "uuid": r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
        "file_path": r"(?:/[a-zA-Z0-9_\-.]+)+",
        "uri": r"[a-zA-Z][a-zA-Z0-9+\-.]*://[^\s]+",
    }

    def __init__(self, extra_patterns: Dict[str, str] | None = None) -> None:
        self._patterns: Dict[str, re.Pattern[str]] = {}
        all_patterns = {**self._DEFAULT_PATTERNS, **(extra_patterns or {})}
        for entity_name, regex_str in all_patterns.items():
            self._patterns[entity_name] = re.compile(regex_str)

    def extract(self, text: str) -> Dict[str, str]:
        """Return the first match for each pattern in the text."""
        result: Dict[str, str] = {}
        for name, pattern in self._patterns.items():
            m = pattern.search(text)
            if m:
                result[name] = m.group(0)
        return result


class NERPipeline:
    """Orchestrates deterministic extraction layers."""

    def __init__(self, *extractors: DeterministicExtractor) -> None:
        self._extractors = list(extractors)

    def extract(self, text: str) -> Dict[str, str]:
        combined: Dict[str, str] = {}
        for ext in self._extractors:
            combined.update(ext.extract(text))
        return combined

    @classmethod
    def from_config(
        cls, enable: bool = True, custom_patterns: Dict[str, str] | None = None
    ) -> "NERPipeline":
        if not enable:
            return cls()
        return cls(RegexEntityExtractor(custom_patterns))
