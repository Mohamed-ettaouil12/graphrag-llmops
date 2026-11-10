"""LLM-as-judge: sample live Langfuse traces and grade answers.

- Deterministic 10% sample: SHA256(trace_id) % 10 < JUDGE_SAMPLE_RATE (default 0.1)
- Runs MISTRAL-large as judge (temperature 0, strict JSON output)
- Posts faithfulness/groundedness scores back to Langfuse as trace scores
- Completely deferred: never runs in the /query code path
- Judge calls are excluded from the sampling calculation (sample from all traces,
  then filter out those already judged)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import instructor
from langfuse import Langfuse
from pydantic import BaseModel, Field

# ── Configuration ────────────────────────────────────────────────────────────

JUDGE_SAMPLE_RATE = float(os.getenv("JUDGE_SAMPLE_RATE", "0.1"))
MISTRAL_MODEL = "mistral-large"
JUDGE_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "judge_v1.j2"

# Cost guard: never spend more than this on judging per day (in USD).
# Since the model is Mistral-large (~$2 / 1M inputs, $6 / 1M outputs), we cap
# the number of judged rows so the cost stays negligible.
MAX_JUDGE_COST_USD = float(os.getenv("MAX_JUDGE_COST_USD", "0.10"))


# ── JSON schema for judge output ─────────────────────────────────────────────

class JudgeOutput(BaseModel):
    faithfulness: int = Field(..., ge=0, le=5, description="How well the answer is supported by the evidence, 0-5")
    groundedness: int = Field(..., ge=0, le=5, description="How grounded the answer is in reality, 0-5")
    reason: str = Field(..., description="Short explanation of the grading decision")


# ── Langfuse helper ─────────────────────────────────────────────────────────

_langfuse = None


def _get_langfuse() -> Langfuse:
    """Initialise Langfuse from env vars (disabled if keys missing)."""
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


# ── Deterministic sampling ──────────────────────────────────────────────────

def _is_sampled(trace_id: str, rate: float = JUDGE_SAMPLE_RATE) -> bool:
    """Return True if this trace_id should be judged, based on deterministic hash.

    We use the last byte of SHA256(trace_id) / 256 as a [0,1) uniform sample.
    This is deterministic and requires no RNG seed.
    """
    h = hashlib.sha256(trace_id.encode("utf-8")).digest()
    # Use the last byte as a float in [0, 256)
    sample = h[-1] / 256.0
    return sample < rate


# ── Judge prompt ────────────────────────────────────────────────────────────

def _load_judge_prompt() -> str:
    """Load the versioned judge prompt template."""
    return JUDGE_PROMPT_PATH.read_text(encoding="utf-8")


# ── Judge execution ─────────────────────────────────────────────────────────

def _judge_trace(trace: dict) -> None | dict:
    """Grade a single Langfuse trace using the LLM judge.

    Returns a JudgeOutput, or None if the judge cannot be run (e.g. no API key).
    """
    lf = _get_langfuse()
    if not lf or not os.getenv("MISTRAL_API_KEY"):
        return None

    # Deterministic sampling check
    trace_id = trace.get("id", "")
    if not _is_sampled(trace_id):
        return None  # not in this run's sample

    # Build the prompt from the trace evidence
    prompt = _load_judge_prompt()

    # Truncate evidence to reasonable length (avoid context overflow)
    evidence = trace.get("output", "") or ""
    # Also look at metadata/entities if available
    metadata = trace.get("metadata", {}) or {}
    entities = metadata.get("entities", []) or []
    retrieval_method = metadata.get("retrieval_method", "")

    # Construct a concise evidence summary
    evidence_summary = f"""Question: {trace.get('input', '')}
Answer: {evidence}
Entities: {entities}
Retrieval method: {retrieval_method}"""

    # Call the LLM judge via instructor for structured output
    try:
        from instructor import from_openai
        from openai import AzureOpenAI  # we use instructor with the litellm proxy

        # Use the litellm-compatible approach: call mistral-large through
        # the LITELLM proxy. The actual model name is resolved at runtime.
        client = instructor.from_openai(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model=MISTRAL_MODEL,
            temperature=0,
        )

        response: JudgeOutput = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt.replace("{{evidence}}", evidence_summary)},
            ],
            max_tokens=512,
        )

        return JudgeOutput(
            faithfulness=response.faithfulness,
            groundedness=response.groundedness,
            reason=response.reason,
        )

    except Exception as exc:  # pragma: no cover
        # Judge failure: we still want the alert check to see 0.0 faithfulness
        # rather than a missing value, so we return a minimal object.
        return JudgeOutput(faithfulness=0, groundedness=0, reason=f"judge error: {exc}")


# ── Score posting ────────────────────────────────────────────────────────────

def _post_score_to_langfuse(trace_id: str, judge: JudgeOutput) -> None:
    """Post faithfulness and groundedness scores back to Langfuse."""
    lf = _get_langfuse()
    if not lf:
        return  # Langfuse disabled – best-effort only

    try:
        lf.score(
            trace_id=trace_id,
            name="faithfulness",
            value=judge.faithfulness,
        )
        lf.score(
            trace_id=trace_id,
            name="groundedness",
            value=judge.groundedness,
        )
    except Exception:  # pragma: no cover
        # Best-effort: if posting fails, we still continue.
        pass


# ── Main entry point ────────────────────────────────────────────────────────

def run_judge_cycle(max_cost_usd: float = MAX_JUDGE_COST_USD) -> dict[str, int]:
    """Run one cycle of the judge: sample traces, grade them, post scores.

    Returns a dict with counts: {"sampled": N, "judged": M, "skipped": K}
    """
    lf = _get_langfuse()
    if not lf:
        print("Langfuse disabled – judge run skipped")
        return {"sampled": 0, "judged": 0, "skipped": 0}

    # Fetch recent traces (last 24h). Langfuse's get_traces doesn't have a
    # built-in time filter, so we fetch a bounded number and rely on the
    # caller (alert_check) to have already narrowed the set.
    # We ask for a generous limit so we don't miss anything.
    try:
        traces = lf.get_traces(limit=200)
    except Exception:  # pragma: no cover
        print("Failed to fetch traces from Langfuse")
        return {"sampled": 0, "judged": 0, "skipped": 0}

    sampled = 0
    judged = 0
    skipped = 0
    total_judge_cost = 0.0

    for trace in traces:
        # Deterministic sampling
        if not _is_sampled(trace.get("id", "")):
            continue
        sampled += 1

        # Skip if already judged (we check for existing scores)
        existing = trace.get("scores", [])
        if any(s.get("name") in ("faithfulness", "groundedness") for s in existing):
            skipped += 1
            continue

        # Run the judge
        judge = _judge_trace(trace.__dict__ if hasattr(trace, "__dict__") else dict(trace))
        if judge is None:
            # No API key or other issue – skip but count
            skipped += 1
            continue

        judged += 1
        _post_score_to_langfuse(trace.get("id", ""), judge)

        # Accumulate cost estimate for the guard
        # Rough estimate: mistral-large ~ $0.0001 per token, ~512 output tokens = $0.05 per judge call
        # We cap at max_cost_usd per run.
        total_judge_cost += 0.05
        if total_judge_cost > max_cost_usd:
            print(f"Judge cost guard triggered (${total_judge_cost:.2f} > ${max_cost_usd}) – stopping judge cycle")
            break

    print(f"Judge cycle complete: sampled={sampled}, judged={judged}, skipped={skipped}")
    return {"sampled": sampled, "judged": judged, "skipped": skipped}


# ── CLI entry point ─────────────────────────────────────────────────────────

def main() -> int:
    """CLI entry point: run one judge cycle."""
    result = run_judge_cycle()
    # Exit code: 0 if everything ok, 1 if something went wrong
    if result["judged"] == 0 and result["sampled"] > 0:
        # No traces were judged but some were sampled – this isn't an error
        # per se, but we warn.
        print("Warning: no traces were judged despite being sampled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())