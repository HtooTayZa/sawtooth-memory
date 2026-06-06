# Sawtooth Memory: Performance Benchmarks

This document provides empirical performance metrics comparing `sawtooth-memory` against industry-standard memory patterns (such as standard sequential summary memory architectures).

---

## Benchmark Methodology

Our benchmark harness (`scripts/benchmark_memory.py`) simulates an intense conversation of **20 highly detailed messages** containing technical data. To evaluate memory recall accuracy, a unique tracking identifier ("Golden Needle") is injected early in the session (`txn_998877_alpha_omega`).

We track three core variables:
1. **Main Thread Latency:** The amount of time the main execution loop is blocked during memory operations.
2. **Final Prompt Token Footprint:** The size of the payload sent back to the primary LLM agent loop.
3. **Information Retention (Recall):** Whether critical unique identifiers survived compression.

---

## Test Environment & Configuration

* **Host Hardware:** Local Machine (NVIDIA GeForce RTX 5060 Laptop GPU, 8GB VRAM)
* **Local LLM Model Server:** Ollama (`phi4-mini`, 2.5 GB)
* **Sawtooth Target Constraints:** `soft_limit_tokens=250`, `hard_limit_tokens=600`, `chunk_size=4`

---

## Performance Matrix (Local Inference Results)

| Performance Metric | Standard Sequential Summary Memory | Sawtooth Hierarchical Memory | Architectural Advantage |
| :--- | :--- | :--- | :--- |
| **Main Thread Latency** | 64.15 seconds | **5.70 seconds** |  **11.3x Faster Execution** |
| **Final Prompt Payload** | 506 tokens | **454 tokens** |  **10% Lower Token Footprint** |
| **UUID / Deterministic Recall** | Variable / Subject to Hallucination | **100% Retained** |  **Guaranteed via L1.5 Ledger** |

---

## Key Architectural Insights

### Why Sawtooth is 11x Faster on the Main Thread
Standard implementations process conversation histories sequentially on the primary application thread. When using local LLM models, this forces your application to completely freeze for several seconds on every turn while the GPU processes context updates.

Sawtooth employs a multi-tiered queue structure that handles message absorption asynchronously. It immediately yields control back to the main user loop, pushing heavier compression sub-routines into a background worker pipeline. The only minor latency occurs at the conclusion of a session (`cm.stop()`), when the buffer safely flushes pending modifications to the disk.

### Token Management vs. Context Integrity
While alternative summaries compress histories into highly lossy paragraphs, they rapidly decay over long sessions and discard exact numbers. Sawtooth optimizes prompt structures through dynamic windowing, balancing active context with its structured L1.5 Entity Ledger to maintain complete retrieval accuracy at a reduced prompt cost.

---

## Re-Running the Benchmarks Locally

Ensure your local Ollama instance is serving the target model, then activate your virtual environment and execute the harness:

```bash
# Ensure model availability
ollama pull phi4-mini

# Run the comparative test
python scripts/benchmark_memory.py
