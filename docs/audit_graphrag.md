# GraphRAG LLMOps Audit — P01

**Repository:** `qa-multisaut` — agricultural corpus, French documents, KG + LLM answers.
**Objective:** Add LLMOps layer (evaluation, CI, observability, cost control) on top of the existing pipeline **without changing its behavior.**
**Status:** Pipeline models are fixed in `checkpoints/`. No retraining.

---

## 1. Architecture — Full Query Flow

```mermaid
flowchart TD
    %% Entry points
    API[FastAPI /ask (src/api/main.py:122)] -->|question| Pipeline[LangGraph Pipeline (src/orchestrator/graph.py:91)]
    API -->|health check| Health[/health (src/api/main.py:110)]

    %% Node 1: Entity Detection
    Entity[EntityDetectionNode (src/orchestrator/nodes.py:42)]:::entity
    Pipeline --> Entity
    Entity -->|question_entities| Neo4j[Neo4jRetrievalNode (src/orchestrator/nodes.py:111)]

    %% Node 2: Neo4j Retrieval
    Neo4j -->|retrieved_triplets| GNN[GNNScoringNode (src/orchestrator/nodes.py:228)]
    Neo4j -->| fallback | PyG[PyGFallbackRetrievalNode (src/orchestrator/nodes.py:197)]

    %% Node 3: GNN Scoring
    GNN -->|scored_triplets| TopK[TopKSelectionNode (src/orchestrator/nodes.py:396)]

    %% Node 4: Top-K Selection
    TopK -->|topk_triplets| Abstain[AbstentionNode (src/orchestrator/nodes.py:424)]

    %% Node 5: Abstention
    Abstain -->|abstained / answer| Gen[GenerationNode (src/orchestrator/nodes.py:464)]

    %% Node 6: Generation
    Gen -->|answer + sources| API

    %% Style notes
    classDef entity fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef other fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
```

**Detailed per-node mapping (LLMOps hook points):**

| Step | File | Function | Hook type |
|------|------|----------|-----------|
| 1 | `src/orchestrator/nodes.py:60` | `EntityDetectionNode.__call__()` | entity extraction trace |
| 2 | `src/orchestrator/nodes.py:131` | `Neo4jRetrievalNode.__call__()` | retrieval latency / error trace |
| 3 | `src/orchestrator/nodes.py:260` | `GNNScoringNode.__call__()` | GNN scoring metadata (logits, BM25 mix) |
| 4 | `src/orchestrator/nodes.py:401` | `TopKSelectionNode.__call__()` | top-k selection log |
| 5 | `src/orchestrator/nodes.py:430` | `AbstentionNode.__call__()` | abstention decision & reason |
| 6 | `src/orchestrator/nodes.py:481` | `GenerationNode.__call__()` | prompt building + LLM call trace |
| 7 | `src/llm/prompt.py:6` | `build_prompt_from_triplets()` | prompt format & token count |
| 8 | `src/llm/load_llm.py:63` | `generate_answer()` | token count, latency, cost model |

**Lifespan / Entry point:**
- `src/api/main.py:27-30` — `lifespan` loads `PipelineConfig()` and calls `create_pipeline()` once.
- `src/api/main.py:134` — `/ask` calls `run_pipeline(question, pipeline_config)` which invokes the compiled LangGraph.

**Mistral model location:** The main pipeline does **not** call Mistral. Mistral appears only in the eval pathway:
- `eval/run_ragas.py:273-312` — `MistralLitellmEmbeddings` for RAGAS judge embeddings
- `eval/run_ragas.py:330-336` — `MISTRAL_API_KEY` required for judge step
- `.env.example:6` — `MISTRAL_API_KEY=` (empty by design; `--no-judge` avoids this requirement)

**Base model called in pipeline:** `Qwen/Qwen2.5-1.5B-Instruct` with 4-bit QLoRA (`src/llm/load_llm.py:9`), configured with `use_4bit=True`, `llm_max_new_tokens=64`.

---

## 2. Files / Functions Where LLMOps Hooks Will Be Added

These are the **concrete surfaces** in the existing repo where new LLMOps code (in `llmops/`, `eval/`, `ops/`, `prompts/`) will integrate, **without modifying** legacy pipeline files:

| Directory | Purpose | Key existing files |
|-----------|---------|-------------------|
| `llmops/` | Adapter, router, cache interfaces for the pipeline | `llmops/adapter.py:ExistingGraphRAGAdapter.query()` (read-only wrap), `llmops/router.py:`, `llmops/cache.py:` |
| `eval/` | Evaluation datasets, runners, judges | `eval/run_ragas.py:`, `eval/datasets/golden_v1.jsonl:` |
| `ops/` | Observability / Langfuse / cost tracking wiring | empty (to be filled) |
| `prompts/` | Prompt template management | `prompts/.gitkeep:` (template directory) |
| `reports/` | Evaluation report storage | `reports/*.md: J1-J16 legacy reports` |

**Specific hook insertion points (read-only, no pipeline modification):**

1. **Tracing:** `NodeBase.log()` in `src/orchestrator/nodes.py:34` — already provides per-node logging dicts; LLMOps can observe these without restructuring.
2. **Router:** `llmops/router.py` — expose `route_question(question)` that returns a mode string; consumed by a potential wrapper around `run_pipeline()`.
3. **Cache:** `llmops/cache.py` — keyed cache of `question → answer` (or `question → prompt + tokens`) LRU; hooked into `GenerationNode` output without altering the node logic.
4. **Cost tracking:** `src/llm/load_llm.py:63` `generate_answer()` — wrap to count output tokens and accumulate cost in `PipelineState.generation_log["cost_usd"]`.
5. **Prompt variation:** `src/llm/prompt.py:6` `build_prompt_from_triplets()` — LLMOps can pre/post-process the prompt string; the function signature is the stable surface.

---

## 3. Gap List (P01 — what's missing today)

| Category | Gap | Reference |
|---|---|---|
| **Evaluation** | No systematic eval set integrated with CI; golden dataset exists (`eval/datasets/golden_v1.jsonl`) but not auto-tested | `tests/test_dataset_frozen.py` |
| **CI / Tests** | No CI pipeline that runs the full GraphRAG query flow on every commit (tests may 503 if GPU/Neo4j unavailable) | `tests/` directory |
| **Tracing** | No Langfuse/tracing integration hooked into the 7 node logs; `NodeBase.log()` exists but is not wired to any observability backend | `src/orchestrator/nodes.py:34` |
| **Routing** | No question routing (e.g., detect off-topic or route to different retrieval mode) | `llmops/router.py` (empty) |
| **Cache** | No answer/prompt cache; every query recomputes from scratch | `llmops/cache.py` (empty) |
| **Cost tracking** | No token counting / cost accumulation for LLM calls; `generate_answer()` has no cost metadata | `src/llm/load_llm.py:63` |
| **Hardcoded prompts** | `build_prompt_from_triplets()` in `src/llm/prompt.py:6` has fixed format; no template parameterization | `src/llm/prompt.py:6` |
| **Abstention configurability** | Thresholds `max_score_threshold=0.0` and `min_triplets_for_generation=1` are hardcoded in `PipelineConfig` | `src/orchestrator/state.py:77-78` |
| **Model provenance** | No metadata tracking which checkpoint version produced which answer (GNN version, LLM version) | `src/orchestrator/state.py:61-74` |
| **Secret management** | `MISTRAL_API_KEY` only needed for RAGAS judge; main pipeline has no API key dependency | `.env.example:6` |

---

## 4. Command that Runs One Query Today

```bash
# Via the API:
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'

# Or via the Python API:
from src.api.main import app
client = client.TestClient(app)
client.post("/ask", json={"question": "What is the capital of France?"})
```

Internally this triggers: `/ask` → `run_pipeline()` (`src/orchestrator/graph.py:91`) → LangGraph 7-node execution → `generate_answer()` (`src/llm/load_llm.py:63`) with `Qwen2.5-1.5B-Instruct` 4-bit.

---

## 5. LLMOps Layer Scaffold (P01) — New Files Created

| Path | Status |
|------|--------|
| `llmops/__init__.py` | exists (imports) |
| `llmops/adapter.py` | exists + `ExistingGraphRAGAdapter` protocol + `ExistingGraphRAGAdapter.query()` |
| `llmops/router.py` | exists (empty, to be filled) |
| `llmops/cache.py` | exists (empty, to be filled) |
| `eval/.gitkeep` | exists |
| `ops/.gitkeep` | exists |
| `prompts/.gitkeep` | exists |
| `reports/.gitkeep` | exists |
| `docs/audit_graphrag.md` | exists (this file) |
| `Makefile` | updated with `eval-legacy`, `eval`, `eval-smoke` targets |
| `.gitignore` | updated with eval/ops/prompts/reports patterns |
| `.env.example` | has `MISTRAL_API_KEY=`, `LANGFUSE_PUBLIC_KEY=`, `LANGFUSE_SECRET_KEY=`, `LANGFUSE_HOST=` |
| `tests/test_llmops_scaffold.py` | scaffolds P01 import/gitignore checks |

---

**Commit:** `chore(llmops): audit graphrag codebase, scaffold llmops layer`