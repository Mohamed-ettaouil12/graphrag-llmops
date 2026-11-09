"""Benchmark script: replay frozen dataset through adapter + report metrics.

Reads the golden dataset, runs each row through the GraphRAG adapter (real or
fake), collects latency/tokens, and emits a JSON report plus a human-readable
MD summary. Reuses cached eval results if less than 24h old.

Usage:
    python scripts/bench.py           # run bench with existing adapter
    python scripts/bench.py --adapter fake  # force fake adapter (no GPU/Neo4j)
    python scripts/bench.py --adapter existing  # try real adapter (may 503)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "eval" / "datasets" / "golden_v1.jsonl"
EVAL_DIR = ROOT / "reports"
BASELINE_JSON = EVAL_DIR / "baseline.json"
BASELINE_MD = EVAL_DIR / "BASELINE.md"
LATEST_EVAL = EVAL_DIR / "eval_latest.json"

# ── Import adapter ────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))

from llmops.adapter import ExistingGraphRAGAdapter, GraphRAGAdapter
from eval.run_ragas import _FakeAdapter


# ── Helpers ────────────────────────────────────────────────────────────
def _cost_per_request(tokens_in: int | None, tokens_out: int | None,
                      /, *,
                      model: str = "mistral-small") -> float:
    """Approximate judge-call cost for one row (mistral-small pricing)."""
    if tokens_in is None or tokens_out is None:
        return 0.0
    input_cost = (tokens_in / 1_000_000) * 0.30
    output_cost = (tokens_out / 1_000_000) * 0.90
    return round(input_cost + output_cost, 6)


def _is_eval_fresh(max_age_hours: int = 24) -> bool:
    """Return True if eval_latest.json exists and is < max_age_hours old."""
    if not LATEST_EVAL.exists():
        return False
    mtime = LATEST_EVAL.stat().st_mtime
    age_hours = (datetime.now(timezone.utc).timestamp() - mtime) / 3600
    return age_hours < max_age_hours


def _load_latest_aggregates() -> dict | None:
    """Load faithfulness and entity_recall from the latest eval report."""
    if not _is_eval_fresh():
        return None
    try:
        with open(LATEST_EVAL) as f:
            payload = json.load(f)
        agg = payload.get("aggregates", {})
        return {
            "faithfulness": agg.get("faithfulness", {}).get("mean"),
            "entity_recall": agg.get("entity_recall", {}).get("mean"),
        }
    except Exception:
        return None


def _row_result(row: dict, adapter: GraphRAGAdapter) -> dict:
    """Run one row through the adapter and collect benchmark fields."""
    question = row["question"]
    t0 = time.perf_counter()
    try:
        result = adapter.query(question, "hybrid")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        entities = list(result.entities)
        chunks = result.chunks
        triples = result.triples
        tokens_in = result.metadata.tokens_in
        tokens_out = result.metadata.tokens_out
        answer = result.answer
        error = False
    except Exception:
        latency_ms = None
        entities = []
        chunks = []
        triples = []
        tokens_in = None
        tokens_out = None
        answer = "ERROR"
        error = True

    return {
        "id": row["id"],
        "question": question,
        "latency_ms": latency_ms,
        "error": error,
        "entities": entities,
        "chunks": chunks,
        "triples": triples,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "answer": answer,
        "hops": row.get("hops"),
        "expected_answer": row.get("expected_answer"),
        "expected_entities": row.get("expected_entities", []),
    }


def _compute_metrics(rows: list[dict], results: list[dict]) -> dict:
    """Compute aggregate benchmark metrics."""
    # entity_recall = |expected ∩ surfaced| / |expected|
    total_expected = 0
    total_overlap = 0
    for row, res in zip(rows, results):
        expected = row.get("expected_entities", [])
        if not expected:
            continue
        total_expected += 1
        surfaced = set()
        for h, r, t in res.get("triples", []):
            surfaced.add(h.lower())
            surfaced.add(t.lower())
        for c in res.get("chunks", []):
            txt = c.get("text", "").lower()
            if txt.split():
                surfaced.add(txt.split()[0])
        overlap = len(set(e.lower() for e in expected) & surfaced)
        total_overlap += overlap

    scored_rows = max(total_expected, 1)
    recall = round(total_overlap / total_expected, 4) if total_expected else None

    # latency stats
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
    p50 = round(median(latencies), 1) if latencies else None
    p95_idx = int(len(latencies) * 0.95) - 1 if latencies else -1
    p95 = round(sorted(latencies)[p95_idx] if latencies and p95_idx >= 0 else None, 1)

    # token & cost stats
    token_in_vals = [r["tokens_in"] for r in results if r["tokens_in"] is not None]
    token_out_vals = [r["tokens_out"] for r in results if r["tokens_out"] is not None]
    avg_tokens_in = round(sum(token_in_vals) / len(token_in_vals), 1) if token_in_vals else None
    avg_tokens_out = round(sum(token_out_vals) / len(token_out_vals), 1) if token_out_vals else None

    error_count = sum(1 for r in results if r["error"])
    error_rate = round(error_count / len(results) * 100, 1) if results else None

    # cost per request (avg, only over non-error rows)
    non_error_costs = [
        _cost_per_request(r["tokens_in"], r["tokens_out"])
        for r in results
        if not r["error"]
    ]
    cost_per_req = round(sum(non_error_costs) / max(len(non_error_costs), 1), 6)

    return {
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "avg_tokens_in": avg_tokens_in,
        "avg_tokens_out": avg_tokens_out,
        "avg_cost_per_request_usd": cost_per_req,
        "error_rate_pct": error_rate,
        "entity_recall": recall,
    }


def main() -> None:
    """Run the benchmark and emit reports."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", choices=["existing", "fake"], default="existing",
        help="Which adapter to use for benchmarking",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to N rows from the dataset (for quick bench)",
    )
    args = parser.parse_args()

    # ── Load dataset ────────────────────────────────────────────────
    rows = []
    with open(DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit is not None:
        rows = rows[: args.limit]

    # ── Pick adapter ────────────────────────────────────────────────
    if args.adapter == "fake":
        adapter: GraphRAGAdapter = _FakeAdapter()
        adapter_name = "FakeAdapter"
    else:
        try:
            adapter = ExistingGraphRAGAdapter()
            adapter_name = "ExistingGraphRAGAdapter"
        except Exception:
            adapter = _FakeAdapter()
            adapter_name = "ExistingGraphRAGAdapter (fallback)"

    # ── Run benchmark ───────────────────────────────────────────────
    results = [_row_result(row, adapter) for row in rows]

    # ── Aggregates ──────────────────────────────────────────────────
    metrics = _compute_metrics(rows, results)

    # ── Load latest eval aggregates (faithfulness, entity_recall) ────
    latest_agg = _load_latest_aggregates()
    if latest_agg is not None:
        for k in ("faithfulness", "entity_recall"):
            if latest_agg[k] is not None and metrics[k] is None:
                metrics[k] = latest_agg[k]

    # ── Output JSON report ──────────────────────────────────────────
    report = {
        "dataset_rows": len(rows),
        "adapter": adapter_name,
        "metrics": metrics,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_JSON, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Also write bench_latest.json (same content, different path)
    with open(ROOT / "reports" / "bench_latest.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── Human-readable MD summary ───────────────────────────────────
    md_lines = [
        "# GraphRAG Benchmark Baseline",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Git SHA:** `{os.popen('git rev-parse HEAD').read().strip()[:12]}`",
        "",
        "## Aggregate Metrics",
    ]

    m = metrics
    def kv(label, val):
        return f"- {label}: `{val}`" if val is not None else f"- {label}: `N/A`"

    md_lines.append(kv("p50 latency (ms)", m["p50_latency_ms"]))
    md_lines.append(kv("p95 latency (ms)", m["p95_latency_ms"]))
    md_lines.append(kv("avg tokens in", m["avg_tokens_in"]))
    md_lines.append(kv("avg tokens out", m["avg_tokens_out"]))
    md_lines.append(kv("avg cost per request (USD)", m["avg_cost_per_request_usd"]))
    md_lines.append(kv("error rate (%)", m["error_rate_pct"]))
    md_lines.append(kv("entity_recall", m["entity_recall"]))
    if latest_agg:
        md_lines.append(kv("faithfulness (from eval)", latest_agg["faithfulness"]))
    md_lines.extend(["", "---", "*Baseline for future LLMops gains (P07+).*"])

    with open(BASELINE_MD, "w") as f:
        f.write("\n".join(md_lines))

    # ── Print concise summary to stdout ─────────────────────────────
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Adapter: {report['adapter']}")
    print(f"Dataset rows: {report['dataset_rows']}")
    print(f"p50 latency (ms): {metrics['p50_latency_ms']}")
    print(f"p95 latency (ms): {metrics['p95_latency_ms']}")
    print(f"avg tokens in: {metrics['avg_tokens_in']}")
    print(f"avg tokens out: {metrics['avg_tokens_out']}")
    print(f"avg cost per request (USD): {metrics['avg_cost_per_request_usd']}")
    print(f"error rate (%): {metrics['error_rate_pct']}")
    print(f"entity_recall: {metrics['entity_recall']}")
    if latest_agg:
        print(f"faithfulness: {latest_agg['faithfulness']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()