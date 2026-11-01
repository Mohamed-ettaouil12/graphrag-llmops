"""Validateur du jeu de données doré (RF-06).

Vérifie le schéma, l'unicité des ids, la déduplication (question normalisée),
les contraintes par catégorie et la distribution minimale. Options :
- --dataset PATH : autre fichier à valider (défaut eval/datasets/golden_v1.jsonl)
- --update-hash  : recalcule le sha256 et l'écrit dans eval/datasets/golden_v1.sha256
                   (cf. `make accept-dataset-change` : chemin de dégel conscient)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "eval" / "datasets" / "golden_v1.jsonl"
HASH_PATH = ROOT / "eval" / "datasets" / "golden_v1.sha256"
REQUIRED_KEYS = {
    "id",
    "question",
    "expected_answer",
    "expected_entities",
    "expected_triples",
    "hops",
    "mode",
    "tags",
}
MIN_DISTRIBUTION = {"local": 20, "hybrid": 15, "global": 8, "abstention": 7}
MIN_TOTAL = 50


class ValidationError(Exception):
    pass


def normalize_question(q: str) -> str:
    s = q.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_rows(path: Path) -> list[dict]:
    rows = []
    try:
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValidationError(f"ligne {lineno}: JSON invalide ({exc})") from exc
    except FileNotFoundError as exc:
        raise ValidationError(f"fichier absent : {path}") from exc
    if not rows:
        raise ValidationError("dataset vide")
    return rows


def validate_row(row: dict, lineno: int) -> None:
    if not isinstance(row, dict) or set(row.keys()) != REQUIRED_KEYS:
        got = ", ".join(sorted(row.keys())) if isinstance(row, dict) else type(row).__name__
        raise ValidationError(f"ligne {lineno}: clés attendues exactement {sorted(REQUIRED_KEYS)} ; reçues [{got}]")
    if not isinstance(row["id"], str) or not row["id"]:
        raise ValidationError(f"ligne {lineno}: id manquant")
    for field in ("question", "expected_answer"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValidationError(f"ligne {lineno}: {field} doit être une chaîne non vide")
    if not isinstance(row["expected_entities"], list) or not all(
        isinstance(e, str) for e in row["expected_entities"]
    ):
        raise ValidationError(f"ligne {lineno}: expected_entities doit être une liste de chaînes")
    triples = row["expected_triples"]
    if not isinstance(triples, list) or not all(
        isinstance(tr, list) and len(tr) == 3 and all(isinstance(x, str) for x in tr)
        for tr in triples
    ):
        raise ValidationError(f"ligne {lineno}: expected_triples doit être une liste de triplets [str, str, str]")
    if not isinstance(row["hops"], int) or isinstance(row["hops"], bool) or row["hops"] not in (1, 2, 3):
        raise ValidationError(f"ligne {lineno}: hops doit être 1|2|3")
    if row["mode"] not in ("local", "global", "hybrid"):
        raise ValidationError(f"ligne {lineno}: mode invalide ({row['mode']!r})")
    if not isinstance(row["tags"], list) or not all(isinstance(t, str) for t in row["tags"]):
        raise ValidationError(f"ligne {lineno}: tags doit être une liste de chaînes")
    abstention = "abstention" in row["tags"]
    if abstention:
        if row["expected_entities"] or row["expected_triples"]:
            raise ValidationError(f"ligne {lineno}: ligne d'abstention mais entités/triples attendus non vides")
    elif not row["expected_entities"]:
        raise ValidationError(f"ligne {lineno}: entités attendues vides pour une ligne non-abstention")


def distribution(rows: list[dict]) -> Counter:
    return Counter(
        "abstention" if "abstention" in row["tags"] else row["mode"] for row in rows
    )


def check_dataset(path: Path) -> Counter:
    rows = load_rows(path)
    for lineno, row in enumerate(rows, 1):
        validate_row(row, lineno)

    ids = [row["id"] for row in rows]
    duplicates_ids = [i for i, c in Counter(ids).items() if c > 1]
    if duplicates_ids:
        raise ValidationError(f"ids dupliqués : {duplicates_ids}")

    questions = [normalize_question(row["question"]) for row in rows]
    duplicate_q = [q for q, c in Counter(questions).items() if c > 1]
    if duplicate_q:
        raise ValidationError(f"questions dupliquées (normalisées) : {duplicate_q[:3]}...")

    counts = distribution(rows)
    if len(rows) < MIN_TOTAL:
        raise ValidationError(f"total {len(rows)} < {MIN_TOTAL} lignes")
    for key, minimum in MIN_DISTRIBUTION.items():
        if counts[key] < minimum:
            raise ValidationError(
                f"distribution : {key} = {counts[key]} < minimum requis {minimum}"
            )
    return counts


def update_hash(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    HASH_PATH.write_text(f"{digest}  {path.name}\n")
    print(f"sha256 régénéré -> {HASH_PATH.name} : {digest[:16]}...")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--update-hash", action="store_true", help="recongèle : réécrit le sha256 (dégel conscient)"
    )
    args = parser.parse_args()

    try:
        counts = check_dataset(args.dataset)
    except ValidationError as exc:
        print(f"ÉCHEC : {exc}", file=sys.stderr)
        return 1

    print(f"OK : {sum(counts.values())} lignes validées ({args.dataset.name})")
    print("Distribution :")
    for key in ("local", "hybrid", "global", "abstention"):
        print(f"  {key:>10} : {counts[key]:>3}  (min {MIN_DISTRIBUTION[key]})")
    if args.update_hash:
        update_hash(args.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
