"""
compressor.py — Ollama-backed async compression engine.

Handles:
  1. Pre-processing: strips base64 blobs, stack traces, verbose JSON.
  2. Dual-extraction inference: sends pruned chunk to a local Ollama model.
  3. Output parsing: returns {"narrative_summary": ..., "extracted_entities": {...}}.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from .config import OllamaConfig
from .exceptions import CompressionError, OllamaConnectionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compression system prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a memory compression engine for an AI agent system.

Your job:
1. Read the conversational logs provided.
2. Write a dense, chronological NARRATIVE of what the agent decided, discovered, \
and accomplished. Be specific. Preserve causality (why things happened).
3. Extract all EXACT DETERMINISTIC VALUES into a flat key-value dictionary. \
This includes: UUIDs, database IDs, file paths, connection strings, precise \
numeric results, API endpoints, and any other value that must be reproduced \
exactly in future tool calls.

Rules:
- IGNORE errors/exceptions if they were subsequently resolved.
- Do NOT include verbose JSON payloads or base64 strings.
- Use snake_case keys in extracted_entities.
- Respond ONLY with valid JSON. No preamble, no markdown fences, no extra text.

Required output schema:
{
  "narrative_summary": "<dense chronological narrative as a single string>",
  "extracted_entities": {
    "<key>": "<exact_value>"
  }
}
"""

# ---------------------------------------------------------------------------
# Pre-processing regexes
# ---------------------------------------------------------------------------

# Base64-like strings over 80 chars (avoids mangling normal text)
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{80,}={0,2}")

# Python / JS stack traces
_STACKTRACE_RE = re.compile(
    r"Traceback \(most recent call last\):.*?(?=\n\n|\Z)",
    re.DOTALL,
)

# Long runs of whitespace-separated hex (e.g. binary output)
_HEX_RE = re.compile(r"(?:[0-9a-fA-F]{2}\s){16,}")


def _prune(raw: str) -> str:
    """Strip noise that wastes compressor tokens without adding meaning."""
    text = _BASE64_RE.sub("[BASE64_REMOVED]", raw)
    text = _STACKTRACE_RE.sub("[STACKTRACE_REMOVED]", text)
    text = _HEX_RE.sub("[HEX_REMOVED]", text)
    return text


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


class OllamaCompressor:
    """
    Async client for the local Ollama inference backend.

    Sends pruned message chunks to a small local model and returns a
    structured dict with 'narrative_summary' and 'extracted_entities'.
    """

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=self._config.timeout_seconds,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("OllamaCompressor: HTTP client closed.")

    async def compress(self, messages_text: str) -> dict:
        """
        Prune and compress a raw message chunk.

        Returns:
            {
                "narrative_summary": str,
                "extracted_entities": dict[str, str],
            }

        Raises:
            OllamaConnectionError: if Ollama is unreachable.
            CompressionError: if the HTTP response indicates an error or
                              the request times out.
        """
        pruned = _prune(messages_text)
        logger.debug(
            f"OllamaCompressor: pruned {len(messages_text)} → {len(pruned)} chars"
        )

        payload = {
            "model": self._config.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Compress the following context logs:\n\n{pruned}",
                },
            ],
        }

        client = await self._get_client()

        try:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self._config.base_url}. "
                "Is Ollama running? (`ollama serve`)"
            ) from exc
        except httpx.TimeoutException as exc:
            raise CompressionError(
                f"Ollama timed out after {self._config.timeout_seconds}s. "
                "Try a smaller model or increase timeout_seconds."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise CompressionError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc

        raw_content = resp.json().get("message", {}).get("content", "")
        return self._parse_output(raw_content)

    def _parse_output(self, content: str) -> dict:
        """
        Parse the model's JSON output. Applies light cleanup to handle
        common model quirks (markdown fences, leading text).
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", content).strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning(
                        "OllamaCompressor: could not parse model output as JSON; "
                        "storing raw text as narrative."
                    )
                    return {"narrative_summary": content, "extracted_entities": {}}
            else:
                return {"narrative_summary": content, "extracted_entities": {}}

        narrative = result.get("narrative_summary", "")
        entities = result.get("extracted_entities", {})

        if not isinstance(entities, dict):
            entities = {}
        entities = {str(k): str(v) for k, v in entities.items()}

        return {"narrative_summary": narrative, "extracted_entities": entities}
