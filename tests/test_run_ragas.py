"""Tests P04 : runner RAGAS (eval/run_ragas.py) en mode offline.

- collect  : _FakeAdapter déterministe (aucun GPU, aucun poids) — STAGES_DIR est
  monkeypatché vers tmp_path pour ne JAMAIS écrire dans reports/ du repo ;
- judge    : --no-judge (pas de ragas/litellm — le venv du repo n'a ni l'un ni
  l'autre). Le chemin judge réel (Mistral) est exécuté via `make eval` sur la
  box GPU + clé d'API.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval.run_ragas as rr


@pytest.fixture
def isolated_stages(monkeypatch, tmp_path) -> Path:
    """Redirige CONTEXTS_FILE/STAGES_DIR du runner vers tmp_path."""
    stages = tmp_path / "stages"
    contexts = stages / "contexts_latest.json"
    monkeypatch.setattr(rr, "STAGES_DIR", stages)
    monkeypatch.setattr(rr, "CONTEXTS_FILE", contexts)
    return stages


@pytest.fixture
def fake_rows() -> list[dict]:
    """Journal 'contexts' minimal (2 rows : answered + abstained)."""
    return [
        {
            "id": "gold-001",
            "question": "Which engine did Ftirzin use?",
            "mode": "local",
            "hops": 1,
            "ground_truth": "Magellant",
            "expected_entities": ["Ftirzin"],
            "tags": ["local"],
            "answer": "[local] Ftirzin and Magellant are linked",
            "contexts": ["[faux_doc] simulated context", "(Ftirzin, determine, Magellant)"],
            "abstained": False,
        },
        {
            "id": "gold-002",
            "question": "Which movie stars Sabliers?",
            "mode": "abstention",
            "hops": 2,
            "ground_truth": "Sansabelum",
            "expected_entities": [],
            "tags": ["abstention"],
            "answer": "(information absente du graphe — abstention)",
            "contexts": [rr.ABSTENTION_FALLBACK_CONTEXT],
            "abstained": True,
        },
    ]


def _write_contexts_report(tmp_path: Path, rows: list[dict]) -> Path:
    """Écrit un journal contexts valide + renvoie son chemin (--report-in)."""
    payload = {
        "_meta": rr.build_meta(
            dataset_path=rr.DEFAULT_DATASET,
            adapter_name="ExistingGraphRAGAdapter",
            generation_model="Qwen/Qwen2.5-1.5B-Instruct (local, 4-bit NF4)",
            judge=None,
        ),
        "rows": rows,
    }
    path = tmp_path / "in" / "contexts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


# ---------- collect (:: _FakeAdapter) ----------


def test_collect_fake_writes_contexts(isolated_stages):
    rc = rr.main(
        [
            "collect",
            "--dataset",
            str(rr.DEFAULT_DATASET),
            "--limit",
            "4",
            "--adapter",
            "fake",
        ]
    )
    assert rc == 0
    contexts_file = rr.STAGES_DIR / "contexts_latest.json"
    assert contexts_file.exists()
    payload = json.loads(contexts_file.read_text())
    assert payload["_meta"]["adapter"] == "_FakeAdapter"
    assert payload["_meta"]["generation_model"] == "fake"
    assert len(payload["rows"]) == 4
    for row in payload["rows"]:
        # chaque row porte des contextes (chunks ou fallback abstention)
        assert row["contexts"], f"row {row['id']} sans contextes"
        # collect ne calcule PAS de métriques (c'est le rôle du judge)
        assert "metrics" not in row
        assert row["ground_truth"] == row["ground_truth"].strip()


def test_collect_fake_abstention_rows(isolated_stages):
    """Rows taguées 'abstention' (mode réel hybrid) : pas de preuve -> fallback + flag."""
    rc = rr.main(["collect", "--dataset", str(rr.DEFAULT_DATASET), "--adapter", "fake"])
    assert rc == 0
    payload = json.loads((rr.STAGES_DIR / "contexts_latest.json").read_text())
    gold = {r["id"]: r for r in rr.load_golden_dataset(rr.DEFAULT_DATASET)}
    abstain_ids = {rid for rid, r in gold.items() if "abstention" in r.get("tags", [])}
    assert abstain_ids, "dataset sans rows abstention (inattendu)"
    for row in payload["rows"]:
        if row["id"] in abstain_ids:
            assert row["abstained"], f"row {row['id']} (tag abstention) non abstained"
            assert row["contexts"] == [rr.ABSTENTION_FALLBACK_CONTEXT]
        else:
            assert not row["abstained"], f"row {row['id']} recevable -> ne doit pas abstained"


def test_collect_meta_dataset_hash(isolated_stages):
    rr.main(["collect", "--dataset", str(rr.DEFAULT_DATASET), "--limit", "1", "--adapter", "fake"])
    payload = json.loads((rr.STAGES_DIR / "contexts_latest.json").read_text())
    assert payload["_meta"]["dataset_hash"] == rr.hash_file(rr.DEFAULT_DATASET)
    assert payload["_meta"]["seed"] == rr.SEED
    assert payload["_meta"]["models"]["generation"] == "fake"


def test_collect_bad_dataset_exits(tmp_path, isolated_stages):
    """Dataset introuvable -> message + code 1 (pas de traceback nu)."""
    rc = rr.main(["collect", "--dataset", str(tmp_path / "missing.jsonl"), "--adapter", "fake"])
    assert rc == 1


# ---------- meta ----------


def test_hash_file_matches_hashlib(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"bonjour pre\240n\255uy")
    assert rr.hash_file(f) == hashlib.sha256(f.read_bytes()).hexdigest()


def test_build_meta_fields(isolated_stages):
    meta = rr.build_meta(
        dataset_path=rr.DEFAULT_DATASET,
        adapter_name="_FakeAdapter",
        generation_model="fake",
        judge=None,
    )
    assert meta["dataset_hash"] == rr.hash_file(rr.DEFAULT_DATASET)
    assert meta["seed"] == rr.SEED
    assert meta["models"]["generation"] == "fake"
    assert meta["models"]["embeddings"] is None
    assert "generation" in meta["prompt_versions"]
    assert meta["git_sha"] == rr.git_sha()


def test_git_sha_nonempty():
    sha = rr.git_sha()
    assert sha != "unknown"  # on est dans un repo git
    assert len(sha) == 40


# ---------- judge (:: --no-judge) ----------


def test_judge_no_judge_roundtrip(tmp_path, fake_rows):
    out_dir = tmp_path / "eval-outputs"
    rc = rr.main(
        [
            "judge",
            "--report-in",
            str(_write_contexts_report(tmp_path, fake_rows)),
            "--reports-dir",
            str(out_dir),
            "--no-judge",
        ]
    )
    assert rc == 0
    assert (out_dir / "eval_latest.json").exists()
    stamped = [p for p in out_dir.glob("eval_*.json") if p.name != "eval_latest.json"]
    assert stamped, "aucun rapport horodaté (eval_*.json)"
    payload = json.loads((out_dir / "eval_latest.json").read_text())
    assert payload["_meta"]["judge"].startswith("skipped (--no-judge smoke :")
    assert payload["aggregates"] == {}
    assert len(payload["rows"]) == 2
    for row in payload["rows"]:
        assert row["metrics"] == {m: None for m in rr.METRICS}


def test_judge_no_judge_numeric_csv(tmp_path, fake_rows):
    """Les métriques nulles restent sérialisables (JSON null, pas NaN JSON invalide)."""
    out_dir = tmp_path / "eval-outputs"
    rr.main(
        [
            "judge",
            "--report-in",
            str(_write_contexts_report(tmp_path, fake_rows)),
            "--reports-dir",
            str(out_dir),
            "--no-judge",
        ]
    )
    text = (out_dir / "eval_latest.json").read_text()
    assert "NaN" not in text
    # JSON strict refuserait NaN/Infinity
    json.loads(text, parse_constant=lambda x: pytest.fail(f"JSON non strict : {x}"))


# ---------- plancher faithfulness ----------


@pytest.mark.parametrize(
    "mean,expected",
    [(0.0, 1), (0.79, 1), (None, 1), (0.80, 0), (0.85, 0)],
)
def test_exit_code_threshold(mean, expected):
    agg = {"faithfulness": {"mean": mean, "scored_rows": 6}}
    assert rr._exit_code(agg, no_judge=False) == expected


def test_exit_code_no_judge_never_fails():
    assert rr._exit_code(None, no_judge=True) == 0
    assert rr._exit_code({"faithfulness": {"mean": 0.0}}, no_judge=True) == 0


# ---------- misc ----------


def test_normalize_judge_model():
    assert rr.normalize_judge_model("mistral-large") == "mistral/mistral-large-latest"
    assert rr.normalize_judge_model("mistral-small") == "mistral/mistral-small"
    assert rr.normalize_judge_model("openai/gpt-4o") == "openai/gpt-4o"
    assert rr.normalize_judge_model("  nova ") == "mistral/nova"


def test_metrics_contained():
    assert set(rr.METRICS) == {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
