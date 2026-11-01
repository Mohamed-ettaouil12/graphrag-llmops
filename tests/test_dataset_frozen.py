"""P03 — gel du dataset doré (golden_v1) : tout édit doit passer par le dégel."""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval" / "datasets" / "golden_v1.jsonl"
HASH_FILE = ROOT / "eval" / "datasets" / "golden_v1.sha256"
UNFREEZE_MSG = "Dégel conscient requis : make accept-dataset-change"


def test_frozen_hash_matches():
    assert DATASET.exists() and HASH_FILE.exists()
    digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    stored = HASH_FILE.read_text().split()
    assert stored, "fichier de hash vide"
    assert stored[0] == digest, f"golden_v1.jsonl modifié après gel ! {UNFREEZE_MSG}"


def test_dataset_has_50_plus_rows():
    rows = [line for line in DATASET.read_text().splitlines() if line.strip()]
    assert len(rows) >= 50, f"seulement {len(rows)} lignes"
    assert rows[0].startswith('{"id":'), "format JSONL attendu"


def test_validator_passes_and_prints_distribution():
    proc = subprocess.run(
        [sys.executable, "scripts/validate_dataset.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "OK :" in proc.stdout
    for category in ("local", "hybrid", "global", "abstention"):
        assert category in proc.stdout, f"distribution {category} manquante"


def test_tampering_detected():
    tampered = DATASET.read_text() + '\n{"id": "g1-999"}\n'
    digest = hashlib.sha256(tampered.encode()).hexdigest()
    stored = HASH_FILE.read_text().split()
    assert digest != stored[0]
