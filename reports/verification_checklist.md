# Verification Checklist — LLMOps Deliverables Audit (P01-P15)

Global score: **18/22 PASS** (82%)

---

## Legend
- **PASS** — file exists with non-trivial content (populated classes, metrics, templates, or YAML)
- **FAIL** — file missing, empty, or stubbed

---

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 1 | `docs/audit_graphrag.md` | PASS | 8252 bytes; contains pipeline audit, G11-G15 references, langfuse tracing specs |
| 2 | `llmops/adapter.py` | PASS | 13396 bytes; has `GraphAnswerMetadata`, `GraphRAGAdapter`, `_get_langfuse()`, `_classify_complexity` |
| 3 | `llmops/router.py` | PASS | 10296 bytes; has `_classify_complexity`, `_LARGE_MODEL`, `_SMALL_MODEL`, `route_generation` |
| 4 | `llmops/cache.py` | **FAIL** | 0 bytes; empty stub — no cache instrumentation present |
| 5 | `eval/datasets/golden_v1.jsonl` | PASS | 25196 bytes; 56-row HotpotQA distractor dataset with `id`, `question`, `expected_answer`, `expected_entities`, `hops`, `tags` |
| 6 | `eval/datasets/golden_v1.jsonl.sha256` | **FAIL** | Missing — no sha256 file alongside the dataset (P03 dataset freeze not completed) |
| 7 | `eval/run_ragas.py` | PASS | 25331 bytes; `cmd_collect`, `cmd_judge`, `_compute_ragas`, `_FakeAdapter`, `_CostTracker` |
| 8 | `eval/graph_metrics.py` | PASS | 12383 bytes; `compute_entity_recall`, `_grade_triple_groundedness`, `_grade_hop_correctness`, aggregates |
| 9 | `reports/baseline.json` | PASS | 321 bytes; `metrics`: p50/p95 latency, entity_recall, avg_cost_per_request, error_rate |
| 10 | `reports/BASELINE.md` | PASS | 343 bytes; aggregate metrics section, generated_at timestamp |
| 11 | `reports/eval_latest.json` | PASS | 4155 bytes; per-row + aggregates + _meta from RAGAS judge |
| 12 | `reports/final.json` | PASS | 327 bytes; adapter + metrics snapshot |
| 13 | `reports/SUMMARY.md` | PASS | 3791 bytes; before/after vs baseline table, target analysis, MET/NOT MET for all 6 targets |
| 14 | `.github/workflows/eval-gate.yml` | PASS | 8277 bytes; eval gate pipeline with RAGAS judge, fail-on-faithlessness < 0.80 |
| 15 | `.github/workflows/daily-alert.yml` | PASS | 2703 bytes; daily alert checks for pipeline health |
| 16 | `.github/workflows/nightly` | **FAIL** | Missing — no nightly workflow present (only `daily-alert.yml` and `eval-gate.yml` exist) |
| 17 | `promptfooconfig.yaml` | PASS | 3976 bytes; litellm model registry, cost per token, JUDGE_MODEL env |
| 18 | `prompts/registry.yaml` | PASS | 40 bytes; `generation: v1`, `judge: v1`, `extraction: v1` |
| 19 | `prompts/generation_v1.j2` | PASS | 603 bytes; Jinja2 template for generation v1 (prompt orchestration) |
| 20 | `prompts/generation_v2.j2` | PASS | 998 bytes; Jinja2 template for generation v2 (stricter citation contract + abstention rule) |
| 21 | `prompts/judge_v1.j2` | PASS | 1155 bytes; Jinja2 template for judge v1 (strict JSON output, no recursion) |
| 22 | `prompts/extraction_v1.j2` | PASS | 714 bytes; Jinja2 template for extraction v1 |
| 23 | `docs/adr/0001-init.md` | PASS | 1162 bytes; ADR with status, context, decision, current state, related ADRs |
| 24 | `docs/adr/0002-prompt-registry.md` | PASS | 2224 bytes; ADR about prompt registry system, rollback mechanism, A/B testing |
| 25 | `docs/adr/0003-prompt-v2.md` | PASS | 2247 bytes; ADR about prompt v2 candidate, structured abstention rule (< 3 label=1 triplets) |
| 26 | `docs/adr/0004-prompt-v2.md` | PASS | 3658 bytes; ADR about promotion decision procedure, criteria, rollback command |
| 27 | `docs/observability.md` | PASS | 4688 bytes; Langfuse span spec, cost tracking, `prompt_version`, `route_taken` stamping |
| 28 | `docs/demo.md` | PASS | 6879 bytes; 5-minute defense script (factual, multi-hop, abstention, cache-hit replay) |
| 29 | `Makefile: install` | PASS | `$(PY) -m pip install -q -r requirements.txt` |
| 30 | `Makefile: test` | PASS | `$(PY) -m pytest tests/ -v --cov=... --cov-fail-under=70` |
| 31 | `Makefile: lint` | PASS | `$(PY) -m ruff check ...` |
| 32 | `Makefile: eval` | PASS | `PYTHONPATH=. $(PY) -m src.eval.run_evaluation --config C5 --split val` |
| 33 | `Makefile: bench` | PASS | `@bash eval/bootstrap.sh; PYTHONPATH=. $(PY) scripts/bench.py` |
| 34 | `Makefile: compare` | PASS | Side-by-side baseline vs `bench_latest.json` display |
| 35 | `Makefile: update-baseline` | PASS | Executes `make bench`, prompts for baseline promotion |
| 36 | `Makefile: prompt-rollback` | **FAIL** | Not found in Makefile — rollback target missing (P15 prompt versioning not wired) |
| 37 | `.env.example` | PASS | 747 bytes; `MISTRAL_API_KEY=`, `LANGFUSE_PUBLIC_KEY=`, `LANGFUSE_SECRET_KEY=`, `LANGFUSE_HOST=` placeholders |

---

## Gap Analysis — Mapped to Prompts

| Gap | File | Prompt Owner | Description |
|-----|------|--------------|-------------|
| llmops/cache.py empty | `llmops/cache.py` | P01 / P14 | Cache instrumentation not yet implemented; `# noqa` directive present but no-op |
| golden_v1.jsonl.sha256 missing | `eval/datasets/` | P03 | Dataset freeze not completed — sha256 hash not generated/committed |
| .github/workflows/nightly missing | `.github/workflows/` | P14 / P15 | CI nightly pipeline not yet added; only daily-alert + eval-gate present |
| Makefile prompt-rollback missing | `Makefile` | P15 | Prompt version rollback target not added to Makefile; ADR 0004 rollback command has no corresponding Make target |

---

## Summary

- **18 deliverables PASS**, **4 deliverables FAIL** (out of 22 listed items; 37 individual sub-checks)
- Key gaps: cache.py stub, missing sha256, missing nightly CI, missing Makefile rollback target
- All ADRs (0001–0004) present with full non-trivial content
- All prompt templates (v1/v2) present and non-trivial
- Benchmark summary (SUMMARY.md) fully populated with MET/NOT MET assessments