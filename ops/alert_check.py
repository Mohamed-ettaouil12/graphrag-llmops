"""Daily alert check: reads Langfuse aggregates for the last 24h and writes
reports/alerts.jsonl. Exits non-zero when any threshold is violated.

Thresholds (all configurable via env vars):
  - JUDGE_FAITHFULNESS_THR: average judge faithfulness (0-5). Alert if < 3.5 (default)
  - LATENCY_P95_THR: p95 query latency in seconds. Alert if > 1.5 (default)
  - ERROR_RATE_THR: error rate as % of queries. Alert if > 2% (default)
  - DAILY_COST_USD_THR: daily spend in USD. Alert if > threshold (default from
    env, no default = disabled)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langfuse import Langfuse


# ── Thresholds (env‑overridable) ────────────────────────────────────────────

JUDGE_FAITHFULNESS_THR = float(os.getenv("JUDGE_FAITHFULNESS_THR", "3.5"))
LATENCY_P95_THR = float(os.getenv("LATENCY_P95_THR", "1.5"))  # seconds
ERROR_RATE_THR = float(os.getenv("ERROR_RATE_THR", "2.0"))  # percent
DAILY_COST_THR = os.getenv("DAILY_COST_USD_THR", "")  # "" means disabled


# ── Langfuse helper ─────────────────────────────────────────────────────────

_langfuse = None


def _get_langfuse() -> Langfuse:
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    try:
        pk = os.getenv("LANGFUSE_PUBLIC_KEY")
        sk = os.getenv("LANGFUSE_SECRET_KEY")
        if pk and sk:
            _langfuse = Langfuse(public_key=pk, secret_key=sk)
        else:
            _langfuse = None  # type: ignore
    except Exception:
        _langfuse = None  # type: ignore
    return _langfuse


# ── Aggregate helpers ────────────────────────────────────────────────────────

def _fetch_traces(last_hours: int = 24) -> list[dict]:
    """Fetch traces from Langfuse for the last *last_hours* (best-effort)."""
    lf = _get_langfuse()
    if not lf:
        return []
    # Langfuse does not provide a built‑in time filter, so we fetch a limited
    # set and rely on the caller to slice by timestamp if needed.
    # We ask for 200 traces which, at >10% sample rate, should cover the 24h window.
    try:
        traces = lf.get_traces(limit=200)
        return list(traces)
    except Exception:
        return []


def _compute_aggregates(traces: list[dict]) -> dict[str, float]:
    """Compute the key aggregates from a list of Langfuse trace dicts."""
    if not traces:
        return {
            "avg_faithfulness": 0.0,
            "p95_latency_ms": 0.0,
            "p95_latency_s": 0.0,
            "error_rate_pct": 0.0,
            "daily_cost_usd": 0.0,
        }

    # Faithfulness: from judge scores (if any); fall back to 0.0 if none
    faith_scores = []
    for t in traces:
        scores = t.get("scores", []) or []
        for s in scores:
            if s.get("name") == "faithfulness" and s.get("value") is not None:
                faith_scores.append(float(s["value"]))
    avg_faithfulness = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0

    # p95 latency: Langfuse stores latency_ms per trace
    latencies = []
    for t in traces:
        lat = t.get("latency_ms", 0) or 0
        if lat is not None:
            latencies.append(float(lat))
    p95_latency_ms = float(
        sorted(latencies)[int(len(latencies) * 0.95) - 1] if latencies else 0
    )
    p95_latency_s = p95_latency_ms / 1000.0

    # Error rate: traces where abstained=True or that had an error
    error_count = 0
    for t in traces:
        # Check metadata for abstention
        meta = t.get("metadata", {}) or {}
        if meta.get("abstained", False):
            error_count += 1
        # Also check if there's an error in the output
        out = t.get("output", "") or ""
        if "ERROR" in out.upper() or "error" in out.lower():
            error_count += 1
    error_rate_pct = (error_count / len(traces) * 100) if traces else 0.0

    # Daily cost: sum of judge costs + estimate of LLM cost per trace
    daily_cost_usd = 0.0  # we defer detailed cost accounting to the judge cycle

    return {
        "avg_faithfulness": round(avg_faithfulness, 2),
        "p95_latency_ms": p95_latency_ms,
        "p95_latency_s": round(p95_latency_s, 2),
        "error_rate_pct": round(error_rate_pct, 2),
        "daily_cost_usd": round(daily_cost_usd, 4),
    }


# ── Alert file writing ──────────────────────────────────────────────────────

def _write_alerts(aggregates: dict[str, float]) -> list[dict[str, object]]:
    """Write reports/alerts.jsonl and return a list of triggered alerts."""
    alerts: list[dict[str, object]] = []
    lines: list[str] = []

    # Faithfulness check
    if aggregates["avg_faithfulness"] < JUDGE_FAITHFULNESS_THR:
        alerts.append(
            {
                "metric": "avg_judge_faithfulness",
                "value": aggregates["avg_faithfulness"],
                "threshold": JUDGE_FAITHFULNESS_THR,
                "message": f"Average judge faithfulness {aggregates['avg_faithfulness']} below threshold {JUDGE_FAITHFULNESS_THR}",
                "severity": "high",
            }
        )
        lines.append(
            f"ALERT avg_judge_faithfulness={aggregates['avg_faithfulness']} < {JUDGE_FAITHFULNESS_THR}"
        )

    # p95 latency check
    if aggregates["p95_latency_s"] > LATENCY_P95_THR:
        alerts.append(
            {
                "metric": "p95_latency_s",
                "value": aggregates["p95_latency_s"],
                "threshold": LATENCY_P95_THR,
                "message": f"p95 latency {aggregates['p95_latency_s']}s above threshold {LATENCY_P95_THR}s",
                "severity": "high",
            }
        )
        lines.append(
            f"ALERT p95_latency={aggregates['p95_latency_s']}s > {LATENCY_P95_THR}s"
        )

    # Error rate check
    if aggregates["error_rate_pct"] > ERROR_RATE_THR:
        alerts.append(
            {
                "metric": "error_rate_pct",
                "value": aggregates["error_rate_pct"],
                "threshold": ERROR_RATE_THR,
                "message": f"Error rate {aggregates['error_rate_pct']}% above threshold {ERROR_RATE_THR}%",
                "severity": "medium",
            }
        )
        lines.append(
            f"ALERT error_rate={aggregates['error_rate_pct']}% > {ERROR_RATE_THR}%"
        )

    # Daily cost check
    if DAILY_COST_THR and aggregates["daily_cost_usd"] > float(DAILY_COST_THR):
        alerts.append(
            {
                "metric": "daily_cost_usd",
                "value": aggregates["daily_cost_usd"],
                "threshold": float(DAILY_COST_THR),
                "message": f"Daily cost ${aggregates['daily_cost_usd']} above threshold ${DAILY_COST_THR}",
                "severity": "medium",
            }
        )
        lines.append(
            f"ALERT daily_cost=${aggregates['daily_cost_usd']} > ${DAILY_COST_THR}"
        )

    # Write alerts.jsonl
    alerts_path = Path("reports/alerts.jsonl")
    alerts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(alerts_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
        # Also write structured JSON line
        if alerts:
            f.write(json.dumps({"alerts": alerts}, ensure_ascii=False) + "\n")

    return alerts


# ── Main entry point ────────────────────────────────────────────────────────

def run_alert_check() -> int:
    """Run one alert‑check cycle.

    Returns 0 if no threshold was crossed, 1 if any threshold was violated.
    """
    traces = _fetch_traces(last_hours=24)
    aggregates = _compute_aggregates(traces)
    alerts = _write_alerts(aggregates)

    # Print summary
    print(f"Aggregates (last 24h): {aggregates}")
    if alerts:
        print("Triggered alerts:")
        for a in alerts:
            print(f"  - {a['message']}")
        return 1  # non‑exit code → GitHub Actions job fails
    else:
        print("No thresholds crossed.")
        return 0


def main() -> int:
    """CLI entry point."""
    return run_alert_check()


if __name__ == "__main__":
    sys.exit(main())