# Sawtooth Memory

[![Automated Test Suite](https://github.com/HtooTayZa/sawtooth-memory/actions/workflows/test.yaml/badge.svg)](https://github.com/HtooTayZa/sawtooth-memory/actions/workflows/test.yaml)
[![PyPI version](https://badge.fury.io/py/sawtooth-memory.svg)](https://badge.fury.io/py/sawtooth-memory)
[![Python Support](https://img.shields.io/pypi/pyversions/sawtooth-memory.svg)](https://pypi.org/project/sawtooth-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A high-performance, asynchronous non-blocking hierarchical memory framework for LLM Agents.**

## The Problem

Standard LLM memory systems (like LangChain's `ConversationSummaryMemory`) process conversation history sequentially on the main application thread. Every time a user sends a message, the entire application freezes while the system waits for an LLM to generate a new historical summary. Furthermore, these summaries suffer from the "Lost in the Middle" hallucination effect, frequently deleting specific UUIDs, names, or rules to save tokens.

## The Solution

**Sawtooth Memory** eliminates this latency and data loss. It immediately stores the user's message and returns control to the application in milliseconds, offloading the heavy summarization to an asynchronous background worker. To prevent hallucinations, it extracts critical facts into an immutable ledger before summarizing.

---

## Architecture & Data Flow

### 1. The Non-Blocking Execution Model

```text
  Standard Memory (Blocking)            Sawtooth Memory (Async)
  ──────────────────────────            ───────────────────────

  [ Application ]                       [ Application ]
         │                                     │
         ▼                                     ▼
  [ Save Context ]                      [ ContextManager ]
         │                                     │
         ▼                                     ├───────────────────┐ (Instant Return)
  [ LLM Summarizes ]                           ▼                   ▼
  (App freezes for 5-10s)               [ Next User Turn ]  [ Background Worker ]
         │                                                         │
         ▼                                                         ▼
  [ Next User Turn ]                                        [ LLM Summarizes ]

```

### 2. The Hierarchical Memory Stack

When your agent is ready to respond, Sawtooth stitches together an optimized context payload from distinct layers, ensuring critical facts are never summarized away.

```text
    Agent Loop
        │
        ▼
┌─────────────────────┐
│   ContextManager    │
│  ┌───────────────┐  │
│  │ L0 System     │  │  immutable persona + tool schemas
│  │ L2 Archive    │  │  compressed narrative memory
│  │ L1.5 Entities │  │  exact IDs, rolling conflict history
│  │ L1 Working    │  │  recent raw conversation
│  └───────────────┘  │
└──────────┬──────────┘
           │
           ▼
     build_prompt() / get_compiled_prompt()
           │
           ▼
        LLM API

```

* **Phase 2 Update to L1.5:** The Entity Ledger now utilizes a rolling window history. Instead of overwriting older values, it preserves conflicts and automatically injects a `<key>__history` variable into the prompt so the LLM can see the chronological provenance of changing variables.

---

## Key Features

* **Zero-Latency Ingestion:** Messages are appended to L1 instantly. A local `tiktoken` monitor checks thresholds without making API calls.
* **Dual LLM Compression Backends:** Run compression locally via `OllamaCompressor` or in the cloud using `CloudCompressor` (with modular adapters for OpenAI, Anthropic, and Gemini).
* **Deterministic NER Engine:** A zero-latency local regex pipeline extracts UUIDs, file paths, and URIs *before* the LLM sees the text, securely populating the Entity Ledger (L1.5) and overriding potential LLM hallucinations.
* **Turn-Based Batching & Debouncing:** Prevent background queue flooding using `max_unsummarized_turns` to trigger compression safely by turn count, alongside token limits.
* **Graceful Degradation:** If the system hits the `hard_limit_tokens` before the asynchronous background worker finishes a cycle, a fallback protocol forcefully truncates the oldest L1 messages on the main thread to prevent API crashes.

---

## Performance Benchmarks

By moving compression to the background, Sawtooth achieves massive latency reductions on the main thread while maintaining 100% recall accuracy.

**Local GPU Benchmark (NVIDIA RTX 5060 | Model: phi4-mini | 20-Message Conversation)**

| Performance Metric | Standard Summary Memory | Sawtooth Hierarchical | Architectural Advantage |
| --- | --- | --- | --- |
| **Main Thread Latency** | 64.15 seconds | **5.70 seconds** | **11.3x Faster Execution** |
| **Final Prompt Payload** | 506 tokens | **454 tokens** | **10% Lower Token Cost** |
| **UUID / Fact Recall** | Variable / Hallucinates | **100% Retained** | **Guaranteed via L1.5 Ledger** |

For full methodology, cloud comparisons, and reproducibility steps, view our [Performance Benchmarks](BENCHMARKS.md).

---

## Installation

```bash
pip install sawtooth-memory

```

*Optional dependencies for cloud providers:*

```bash
pip install langchain-openai langchain-anthropic langchain-google-genai

```

---

## Quickstart (V2 Configuration)

The V2 configuration introduces dynamic validation, allowing you to set a single `background_model` parameter that automatically routes to the respective local or cloud backend.

```python
import asyncio
from sawtooth_memory import ContextManager, ContextManagerConfig

async def main():
    # V2 Simplified Configuration
    config = ContextManagerConfig(
        background_model="gpt-4o-mini",   # Auto-routes to CloudCompressor (or "phi4" for local Ollama)
        soft_limit_tokens=1000,           # Token threshold to trigger background compression
        hard_limit_tokens=2000,           # Emergency truncation limit
        max_unsummarized_turns=10,        # Turn-based batching threshold
        enable_deterministic_ner=True     # Enable local regex extraction for the Entity Ledger
    )

    async with ContextManager(system_prompt="You are a helpful assistant.", config=config) as cm:

        # Optional: Run a health check to verify backend routing and worker status
        await cm.health_check()

        # 1. Instantly ingest messages (Zero-latency on the main thread)
        await cm.add_message("user", "My transaction ID is txn_998877_alpha")
        await cm.add_message("assistant", "I have noted your transaction ID.")

        # 2. Build the optimized prompt to send to your main LLM
        prompt = await cm.get_compiled_prompt()
        print(prompt)

if __name__ == "__main__":
    asyncio.run(main())

```

---

## Advanced Features

### 1. Deterministic NER (Named Entity Recognition)

By setting `enable_deterministic_ner=True`, Sawtooth intercepts incoming text and uses a fast regex engine to extract critical string identifiers directly into the Entity Ledger. You can also inject custom patterns:

```python
config = ContextManagerConfig(
    enable_deterministic_ner=True,
    custom_ner_patterns={
        "aws_arn": r"arn:aws:[a-z0-9\-]+:[a-z0-9\-]+:\d{12}:[a-zA-Z0-9\-\_/]+"
    }
)

```

### 2. LangGraph Integration & ToolMessage Sanitization

Sawtooth provides a native `SawtoothLangGraphAdapter` to sync state seamlessly.

**V2 Safety Feature:** Strict cloud APIs (like Anthropic/OpenAI) will crash if a `ToolMessage` is sent without its parent `AIMessage` (the tool call request). The LangGraph adapter includes an advanced **3-pass sanitization logic** that automatically detects and drops orphaned `ToolMessage`s when their parent `AIMessage` has been compressed and evicted to L2 Archival Memory.

```python
from langgraph.graph import StateGraph
from sawtooth_memory.integrations.langgraph import SawtoothLangGraphAdapter

# Initialize the adapter with your Sawtooth ContextManager
adapter = SawtoothLangGraphAdapter(cm)

# Automatically syncs state, deduplicates message IDs, and sanitizes orphaned tools
sanitized_messages = await adapter.sync_and_sanitize(langgraph_state_messages)

```

### 3. Recall Explainability Traces

Sawtooth eliminates the "black-box" of agent memory by providing deterministic audit trails. You can query the memory system to see exactly why a fact was retained in the prompt.

```python
trace = await cm.explain_prompt()

import json
print(json.dumps(trace, indent=2))

```

**Output:**

```json
{
  "system_prompt": "You are a helpful assistant.",
  "l2_summary_lineage": [
    "User initiated troubleshooting for router.",
    "User provided MAC address."
  ],
  "l1_5_entities": [
    {
      "key": "user_transaction_id",
      "value": "txn_998877_alpha",
      "origin": "Anchored via L1.5 deterministic NER extraction"
    }
  ],
  "l1_active_messages": 4,
  "total_tokens": 342
}

```

---

## Roadmap

* [x] **Phase 1: Core Architecture**
* [x] L1/L2 Hierarchical Buffer
* [x] Asynchronous Background Worker
* [x] Local (Ollama) & Cloud compatibility


* [x] **Phase 2: Observability & Stability**
* [x] EventBus Subsystem & JSONL Auditing Journal
* [x] Explainability Traces & Performance Benchmarking Harness
* [x] Deterministic NER Engine
* [x] LangGraph ToolMessage Sanitization
* [x] Turn-Based Batching & Debouncing


* [ ] **Phase 3: Advanced Architectures (Up Next)**
* [ ] Multi-Agent Memory Pooling (Shared contextual state)
* [ ] Semantic Vector L3 Archival Memory (RAG integration)
* [ ] Redis/Postgres Adapter for Distributed Deployments



---

## Contributing

We welcome pull requests. See our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to run the test suite and ensure code quality.

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
