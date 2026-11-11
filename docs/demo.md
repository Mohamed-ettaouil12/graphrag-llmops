# Demo 5-Minute Defense Script

## Objective
Demonstrate the QA multi-sauts pipeline in ~5 minutes, covering: factual question, multi-hop question, abstention, cache-hit replay, and live Langfuse trace.

## Setup (already done)

```bash
source .venv/bin/activate
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
# Wait a few seconds for server to start
```

---

## Scenario 1: Question Factual (≈ 1 min)

**Question:** `Quelle est la capitale de la France ?`

**Expected:** The pipeline returns "Paris" as the capital of France, with cited sources.

**What to show:**
- API request via curl
- API response with `answer`, `entities`, `triples`, `sources`
- `prompt_version` in the trace (active version from `prompts/registry.yaml`)
- Answer format with `[source_idx]` citations

**Command:**

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle est la capitale de la France ?"}'
```

---

## Scenario 2: Question Multi-sauts (≈ 1 min)

**Question:** `Quelle est la capitale du pays dont l'auteur de "Le Petit Prince" est originaire ?`

**Expected:** The pipeline chains: extract entity → retrieve triplets → GNN score → LLM generate → answer with citations. This is a 2-hop question requiring two fact sources.

**What to show:**
- Langfuse trace showing multi-stage pipeline execution
- How the prompt version affects answer format/citations
- The answer with `[source_idx]` citations
- visible multi-hop reasoning in the trace

**Command:**

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle est la capitale du pays dont l\'auteur de \"Le Petit Prince\" est originaire ?"}'
```

---

## Scenario 3: Abstention (≈ 1 min)

**Question:** `Qui a inventé la couleur bleue ?` (or any question with no relevant triplets in the KG)

**Expected:** The pipeline abstains from answering (output indicates no answer due to insufficient evidence).

**What to show:**
- API response indicating abstention
- `prompt_version` and how the v2 abstention rule works (< 3 label=1 triplets → abstain)
- Langfuse trace showing the abstention decision path
- The `abstention_reason` field in the metadata

**Command:**

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Qui a inventé la couleur bleue ?"}'
```

---

## Scenario 4: Cache-Hit Replay (≈ 1 min)

**Action:** Send the same factual question twice in a row (e.g., "Quelle est la capitale de la France ?").

**Expected:** The second request should be served faster (cached response) and the Langfuse trace should show identical structure (same `prompt_version`, same entities/triples).

**What to show:**
- First request latency vs second request latency (cache hit → lower p95)
- Langfuse trace comparison (same `prompt_version`, same structure)
- Demonstration that the prompt registry is working correctly for replay

**Commands:**

```bash
# First request
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle est la capitale de la France ?"}'

# Second request (should be faster via cache)
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle est la capitale de la France ?"}'
```

---

## Live Langfuse Trace

If Langfuse keys are configured (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `MISTRAL_API_KEY`), the live trace will show:

- **Pipeline stages**: retrieval → GNN scoring → generation → judge → abstention
- **`prompt_version` stamp** on each node (v1 or v2 per the registry)
- **Cost and latency metrics** per request (`prompt_tokens`, `completion_tokens`, `cost_usd`, `latency_ms`)
- **Judge scores** (faithfulness, etc.) if LLM calls are made
- **`route_taken`**: `simple-small` or `complex-large` via the LLMops router
- **`abstained`**: boolean flag if the abstention rule triggered

**Example trace excerpts** (question factuelle "Quelle est la capitale de la France ?") :

| Field | Value |
|-------|-------|
| `prompt_version` | `generation_v1` (or `v2` if flipped) |
| `route_taken` | `simple-small` (heuristic classify as simple) |
| `cost_usd` | `0.0` (modèle local 4-bit, pas d'appel API) |
| `abstained` | `False` |
| `entities` | `['F', 'F']` (first two words du question extraits) |
| `triples` | `[('F', 'détermine', 'F')]` *(simulé par _FakeAdapter)* |
| `usage.prompt_tokens` | `12` |
| `usage.completion_tokens` | `24` |
| `usage.total_tokens` | `36` |
| `latency_ms` | `~1-5 ms` (FakeAdapter) / `~340 ms` (real GNN+Neo4j) |

**Without Langfuse keys**, spans are created in no-op mode (still stamp `prompt_version` and `route_taken` in the response metadata).

---

## Key Talking Points

1. **Prompt industrialization**: Swap prompt versions at runtime without code changes — just edit `prompts/registry.yaml` and run `make prompt-rollback`.
2. **A/B testing framework**: `scripts/ab_prompts.py` compares v1 vs v2 on the frozen dataset; results in `reports/ab_v1_v2.json` + `reports/ab_prompts_summary.md`.
3. **Decision procedure**: v2 promotes only if no metric drops > 5%, faithfulness improves/stays same, cost decreases (ADR 0004).
4. **Rollback**: `make prompt-rollback STAGE=generation V=1` instantly flips the active version.
5. **Abstention improvement**: v2 adds a structured abstention rule (< 3 label=1 triplets → ABSTAIN) to reduce hallucinations.
6. **Cache hit rate**: Replay of identical questions shows reduced latency; with the real adapter, warm-cache p95 is ~340 ms vs ~1-5 ms for cached requests.
7. **Langfuse observability**: Every `query()` produces a complete span with `trace_id`, `session_id`, `usage`, `cost`, `prompt_version`, and `route_taken` — no code changes needed to add tracing.

---

## Commands Summary

```bash
# Start API
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1

# Test factual question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle est la capitale de la France ?"}'

# Test multi-hop question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle est la capitale du pays dont l\'auteur de \"Le Petit Prince\" est originaire ?"}'

# Test abstention (no KG triplets)
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Qui a inventé la couleur bleue ?"}'

# Cache-hit replay (same question twice)
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle est la capitale de la France ?"}'

# Rollback to v1
make prompt-rollback STAGE=generation V=1

# A/B test comparison
.venv/bin/python scripts/ab_prompts.py
```