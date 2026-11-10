"""Promptfoo provider that calls the local GraphRAG adapter.

This provider is deterministic: it always uses ExistingGraphRAGAdapter
(with lazy graph loading) and a fixed 10-row smoke subset of the golden
dataset. No randomness, no GPU dependency beyond what the adapter needs.

The output format follows promptfoo's expected shape: each item has
`question`, `answer`, `metadata` fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on the path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llmops.adapter import ExistingGraphRAGAdapter, GraphRAGAdapter, GraphAnswerMetadata

# ── Deterministic 10-row smoke subset (first 10 ids from golden_v1) ─────────
# These are selected by id order, no randomness.
SMOKE_QUESTIONS: list[dict] = []


def _load_smoke_questions() -> list[dict]:
    """Load the deterministic 10-row smoke subset from the golden dataset."""
    global SMOKE_QUESTIONS
    if SMOKE_QUESTIONS:
        return SMOKE_QUESTIONS

    dataset_path = REPO_ROOT / "eval" / "datasets" / "golden_v1.jsonl"
    rows: list[dict] = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # Deterministic: first 10 rows, sorted by id (they already are)
    smoke = rows[:10]
    SMOKE_QUESTIONS = [
        {"id": r["id"], "question": r["question"], "expected_answer": r["expected_answer"], "expected_entities": r["expected_entities"]}
        for r in smoke
    ]
    return SMOKE_QUESTIONS


# ── Provider interface ────────────────────────────────────────────────────────

def provider() -> list[dict]:
    """Run the smoke subset through the real adapter and return results.

    promptfoo calls this function to get model answers for evaluation.
    The results must include at minimum: `question`, `answer`,
    `expected_answer` (if promptfoo uses it for grading), and any
    `metadata` the caller needs.

    Returns
    -------
    list[dict]
        One dict per smoke-row with at least `question` and `answer`.
    """
    questions = _load_smoke_questions()
    adapter: GraphRAGAdapter = ExistingGraphRAGAdapter()
    results: list[dict] = []

    for qinfo in questions:
        try:
            result: GraphAnswer = adapter.query(qinfo["question"], "hybrid")
            answer = result.answer
            entities = list(result.entities)
            triples = result.triples
        except Exception as exc:
            answer = f"ERROR: {exc}"
            entities = []
            triples = []

        results.append(
            {
                "question": qinfo["question"],
                "answer": answer,
                "id": qinfo["id"],
                "expected_answer": qinfo["expected_answer"],
                "expected_entities": qinfo["expected_entities"],
                "entities": entities,
                "triples": triples,
                "metadata": {
                    "latency_ms": result.metadata.latency_ms if hasattr(result, 'metadata') and result.metadata else None,
                    "tokens_in": result.metadata.tokens_in if hasattr(result.metadata, 'tokens_in') else None,
                    "tokens_out": result.metadata.tokens_out if hasattr(result.metadata, 'tokens_out') else None,
                    "model": result.metadata.model if hasattr(result.metadata, 'model') else "",
                },
            }
        )

    return results


def main() -> None:
    """Entry point when called directly (for testing/debugging)."""
    import json
    results = provider()
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()