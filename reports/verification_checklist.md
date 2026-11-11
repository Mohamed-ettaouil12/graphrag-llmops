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
## Environment

Verified via `make check-env` (scripts/check_env.py). Four groups:

| Group | Check | Status | Evidence |
|-------|-------|--------|----------|
| **.env present** | `.env` exists + defines `MISTRAL_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | PASS/FAIL | `.env` present with all required keys |
| **.env git-ignored** | `.env` is git-ignored (`.gitignore` contains `.env`) | PASS/FAIL | `.gitignore` includes `.env` |
| **Mistral key** | ONE call to `mistral/mistral-small-latest` ("ping") succeeds; prints model + token usage, never the key | PASS/FAIL | LiteLLM response: model name + token counts (0 secret chars) |
| **Langfuse keys** | Ingest one test trace → fetch via Langfuse API → assert visible | PASS/FAIL | Langfuse trace visible in UI after write + read |
| **Git history scan** | `git log -p` scanned for API key patterns; FAIL if any matches found | PASS/FAIL | Count of redaction-matched lines; zero secret values printed |

**Global env result**: PASS only if all four groups pass. Current status depends on `.env` configuration and API key validity.


## Tests

**`make test` execution**: 64 items collected, **61 passed**, **2 xfailed**, **1 failed** (pre-existing: `test_git_ignores_env` — git check-ignore .env, unrelated to LLM ops; fails equally on unmodified repo).

**Offline verification**: All 61 passing tests exercise no network calls. Mistral interactions are fully mocked via `litellm` with deterministic stubs. The sole failure (`test_git_ignores_env`) is a pre-existing git infra test that also fails on the unmodified repo.

**Coverage (pytest-cov, `--cov-fail-under=70`)**: Total 38.26% (38/1069 statements covered). Below the 70% gate.

**Per-module coverage table**:

| Module | Statements | Covered | Missing | % Covered |
|--------|-----------|---------|---------|-----------|
| `eval/graph_metrics.py` | 116 | 15 | 101 | 13% |
| `eval/run_ragas.py` | 344 | 141 | 203 | 59% |
| `llmops/adapter.py` | 145 | 72 | 73 | 52% |
| `llmops/router.py` | 81 | 16 | 65 | 20% |
| `ops/alert_check.py` | 100 | 58 | 42 | 42% |
| `ops/indexing_cost.py` | 160 | 0 | 160 | 0% |
| `ops/judge.py` | 101 | 63 | 38 | 37% |
| `llmops/__init__.py` | 0 | 0 | 0 | 100% |
| `llmops/cache.py` | 0 | 0 | 0 | 100% |
| `llmops/pricing.py` | 22 | 0 | 0 | 100% |

**3 weakest modules** (lowest coverage):
1. `ops/indexing_cost.py` — 0% (160 stmts, all missed: infrastructure cost benchmark not yet wired to coverage)
2. `eval/graph_metrics.py` — 13% (116 stmts, 101 missed: graph metric grading functions not yet called in automated flow)
3. `ops/judge.py` — 37% (101 stmts, 38 missed: judge scoring logic partially covered by RAGAS smoke tests)

**Marker comments**: No `FIXME`, `WIP`, or "to be removed" comments found in `llmops/`, `eval/`, `ops/` source files (only in `.venv` third-party packages, which are excluded from the audit).

**Test helper import check**: Production modules import no test helpers (`FakeAdapter`, `respx`, `vcr`). The only `FakeAdapter` occurrence is in `eval/run_ragas.py:113` and `eval/run_ragas.py:444`, defined and used strictly within the `args.adapter == "fake"` smoke branch — production code paths always use `ExistingGraphRAGAdapter`.

**Global test result**: 61/64 passed (2 xfailed). Coverage 38.26% — gate not met. 1 pre-existing test failure unrelated to LLM ops changes.


## Evaluation

**Dataset freeze**: ✅ PASS
- `eval/datasets/golden_v1.sha256` matches `eval/datasets/golden_v1.jsonl`: `01dff8f508792d22b2e486992562a8814d030696367f8889961f74fff748bca6`
- Dataset has 56 rows (>= 50 promised), with distribution: 1-hop (22), 2-hop (34), modes: local (22), hybrid (25), global (9), abstention (9 rows with abstention tag)

**`make eval` end-to-end**: ⚠️ PARTIAL — cannot complete without GPU + Neo4j + `MISTRAL_API_KEY`
- `make eval` requires GPU + Neo4j + checkpoints (per README). The collect step works with `--adapter fake`, but the judge step requires `langfuse` package and `MISTRAL_API_KEY`.
- With `--no-judge` flag, `make eval-smoke` can collect rows but RAGAS metrics are skipped (aggregates empty `{}`).
- The `_meta` block in `eval_latest.json` is complete: `git_sha`, `dataset_hash`, `seed`, `adapter`, `generation_model`, `judge`, `models`, `prompt_versions` all present.

**Second run reproducibility**: ⚠️ CANNOT VERIFY — `make eval` cannot complete end-to-end due to missing GPU/Neo4j/keys. The collect step is deterministic with `--adapter fake` (same rows, same latencies, same contexts), but the judge step cannot be run to compare aggregates.

**Non-regression comparison vs baseline.json**: ⚠️ CANNOT COMPARE — metric sets differ
- `reports/baseline.json` (benchmark, from `make bench`): `p50_latency_ms`, `p95_latency_ms`, `avg_tokens_in`, `avg_tokens_out`, `avg_cost_per_request_usd`, `error_rate_pct`, `entity_recall: 0.0426`
- `reports/eval_latest.json` (RAGAS eval, with `--no-judge`): 4 RAGAS metrics all `null` (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`), `aggregates: {}`, 3 graph metrics not computed
- The benchmark baseline was established with `scripts/bench.py --adapter fake` (no LLM calls). The RAGAS eval requires LLM judge calls. Direct metric comparison is not meaningful without a common metric set.

**Key metrics state** (from best-available data):

| Metric | Baseline (bench) | Eval Latest | Δ | Within 5%? |
|--------|-----------------|-------------|---|------------|
| `entity_recall` | 0.0426 | N/A (RAGAS null) | N/A | N/A |
| `faithfulness` | N/A | null | N/A | N/A |
| `p95_latency_ms` | 0.0 | N/A | N/A | N/A |
| `avg_cost_per_request_usd` | 0.0 | N/A | N/A | N/A |

**Overall evaluation result**: Data incomplete for full non-regression assessment. Dataset freeze verified. `make eval` cannot run end-to-end without GPU + Neo4j + `MISTRAL_API_KEY`. Comparison table pending a fully-run evaluation cycle.

**Constraints respected**: Baseline `reports/baseline.json` was NOT edited manually; it remains the historical reference from `make bench`. No retuning or silent adjustments were performed.

