# Observability Dashboard — GraphRAG LLMOps

Dashboard views for monitoring the GraphRAG pipeline in production.

## Quality View

| Metric | Description | Target |
|--------|-------------|--------|
| `entity_recall` | Fraction of expected entities found in top-k triplets | > 0.05 (baseline) |
| `faithfulness` | Fraction of answers supported by retrieved triplets | > 0.50 (baseline) |
| `error_rate` | % of queries returning an error or abstention | < 10% |
| `abstention_rate` | % of queries producing "Information insuffisante" | varies by domain |

## Cost View

| Metric | Description | Target |
|--------|-------------|--------|
| `cost_per_request_usd` | Approximate cost per query (local 4-bit: $0.00 API cost) | $0.00 (infrastructure only) |
| `prompt_tokens` | Average prompt tokens per query | ~200–500 |
| `completion_tokens` | Average completion tokens per query | ~30–60 |
| `monthly_infra_usd` | Approximate electricity/hosting cost | ~$10–30 (RTX 3050 Ti) |

## Latency View

| Metric | Description | Target |
|--------|-------------|--------|
| `p50_latency_ms` | Median query latency | < 2000 ms |
| `p95_latency_ms` | 95th-percentile query latency | < 5000 ms |
| `generation_latency_ms` | LLM generation-only latency | < 1000 ms |
| `endpoint_latency_ms` | Full /ask endpoint latency (including KG + GNN + LLM) | < 5000 ms |

---

## Langfuse Integration

All queries are traced through Langfuse when `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` are set in the environment. Each trace contains:

- **`trace_id`** — same value returned in the API response (`/ask` header
  or JSON field), for correlating with logs/metrics
- **`tags`** — `env=dev|prod`, plus optional `session_id`
- **`usage`** — `prompt_tokens`, `completion_tokens`, `total_tokens`
- **`cost`** — USD cost per query (0.0 for local 4-bit model)
- **`metadata`** — entities touched, triples used, retrieval method,
  abstention flag
- **`spans`** — 7 node spans (entity_detection, neo4j_retrieval, gnn_scoring,
  topk_selection, abstention, generation + init + end)

### Dashboard URLs

- **Quality:** `https://cloud.langfuse.com/<project>/quality`
- **Cost:** `https://cloud.langfuse.com/<project>/cost`
- **Latency:** `https://cloud.langfuse.com/<project>/latency`

> **NOTE:** The screenshots below are placeholders to be filled in after
> the Langfuse dashboard is set up with real data.

### Screenshots (to be completed after dashboard setup)

| View | Screenshot |
|------|------------|
| Quality metrics dashboard | ![TODO: quality dashboard] |
| Cost per request waterfall | ![TODO: cost dashboard] |
| Latency p95 distribution | ![TODO: latency dashboard] |
| Trace detail (question → answer → spans) | ![TODO: trace detail] |

> To populate these screenshots:
> 1. Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in the deployment
>    environment.
> 2. Run `scripts/langfuse_smoke.py` to generate sample traces.
> 3. In the Langfuse UI, navigate to the Quality, Cost, and Latency tabs.
> 4. Use the UI's "Export" or "Share" function to capture screenshots.

---

## API Trace Contract

Each `/query` (or `/ask`) call produces a Langfuse trace with the
following fields guaranteed:

| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | `str` | Unique identifier; returned in the API JSON response |
| `tags.env` | `str` | `"dev"` or `"prod"` |
| `tags.session_id` | `str` | Opaque session identifier; `"default"` if none |
| `usage.prompt_tokens` | `int` | Tokens in the prompt (estimated chars/4) |
| `usage.completion_tokens` | `int` | Tokens in the model response (estimated chars/4) |
| `usage.total_tokens` | `int` | Sum of prompt + completion |
| `cost_usd` | `float` | USD cost per query ($0.00 for local 4-bit) |
| `level` | `str` | `"default"` or `"error"` if the pipeline failed |
| `status` | `str` | `"completed"` or `"failed"` |
| `metadata.entities` | `list[str]` | Entity names touched by the pipeline |
| `metadata.triples` | `list[tuple[str,str,str]]` | KG triplets used |
| `metadata.retrieval_method` | `str` | `"neo4j"`, `"pyg_fallback"`, or `"none"` |
| `metadata.abstained` | `bool` | Whether the pipeline produced "Information insuffisante" |

> **Truncation rule:** Chunk texts stored in traces are limited to
> `<= 500` characters (`_truncate()` helper). Full documents are never
> sent to Langfuse.

---

## Running the Smoke Test

```bash
# Ensure Langfuse keys are set (required for API trace visibility)
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...

# Run the smoke script
.venv/bin/python scripts/langfuse_smoke.py
# Expected: "SUCCESS: 3 complete traces visible through Langfuse API"
```