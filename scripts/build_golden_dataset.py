"""Construit eval/datasets/golden_v1.jsonl à partir du corpus réel du dépôt.

Sources (jamais modifiées) :
- data/processed/{test,val}.jsonl     (questions/answers HotpotQA + docs)
- data/processed/kg_triplets.jsonl    (REBEL → KG, label 1 = triplet de support)

Catégories :
- local     : questions factuelles à 1 saut générées depuis des triplets
              (t = mot-relation, r = valeur ; lisible car c'est l'orientation
              de ce KG — cf. docs/audit_graphrag.md).
- hybrid    : questions multi-sauts « bridge » HotpotQA réelles (2 documents).
- global    : questions comparatives HotpotQA (« global/community-style »).
- abstention: questions réellement absentes du KG (aucun triplet) ; le
              système doit s'abstenir — expected_answer = réponse gold
              HotpotQA, à titre de traçabilité, et success = abstention.

Déterministe : SEED=42 ; tri stable ; RNG réservé aux échantillonnages.
Gel : eval/datasets/golden_v1.sha256 (via scripts/validate_dataset.py --update-hash).
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT_PATH = ROOT / "eval" / "datasets" / "golden_v1.jsonl"
SEED = 42
SMALL_WORDS = {"a", "an", "and", "as", "at", "in", "of", "on", "or", "the", "to", "s"}

TEMPLATES = {
    "composer": "Who is the composer of {h}?",
    "cast member": "Who is a cast member of {h}?",
    "country": "In which country is {h} located?",
    "date of birth": "What is the date of birth of {h}?",
    "located in administrative territorial entity": "In which administrative entity is {h} located?",
    "sport": "Which sport is {h} associated with?",
    "instance of": "What is {h} an instance of?",
    "part of": "What is {h} part of?",
    "position held": "Which position was held by {h}?",
    "capital": "What is the capital of {h}?",
}
LOCAL_MAX_PER_TEMPLATE = 4
N_LOCAL = 22
N_MULTIHOP = 16
N_GLOBAL = 9
N_ABSTENTION_EXTRA = 2
MAX_ENTITIES = 12
MAX_EXPECTED_TRIPLES = 6


def pretty(name: str) -> str:
    """Casse lisible pour le texte de question (kg est en minuscules)."""
    words = name.split()
    out = []
    for i, w in enumerate(words):
        out.append(w if (w in SMALL_WORDS and i > 0) else w.capitalize())
    return " ".join(out)


def normalize_question(q: str) -> str:
    s = q.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def entity_like(value: str) -> bool:
    """Les valeurs (r) dater de type numérique/date ne sont pas des entités."""
    return bool(value) and not re.search(r"\d", value)


def rows_split(split: str) -> list[dict]:
    return load_jsonl(DATA / f"{split}.jsonl")


def triplets_by_qid() -> dict[str, list[dict]]:
    by_qid = defaultdict(list)
    for t in load_jsonl(DATA / "kg_triplets.jsonl"):
        by_qid[t["qid"]].append(t)
    for qid in by_qid:
        by_qid[qid].sort(key=lambda t: (t["h"], t["r"], t["t"]))
    return by_qid


def doc_tag(supporting: list[dict]) -> str:
    titles = [s["title"] for s in supporting[:2]]
    return "docs:" + "|".join(titles)


def build_local(by_qid: dict[str, list[dict]]) -> list[dict]:
    """Questions 1-saut générées depuis des triplets réels (label==1)."""
    random.seed(SEED)
    seen_q = set()
    candidates = defaultdict(list)
    for qid, tlist in sorted(by_qid.items()):
        for t in tlist:
            if t.get("label") != 1 or t["t"] not in TEMPLATES:
                continue
            h, r, rel = t["h"], t["r"], t["t"]
            if not h or r in ("has_info", "") or h == rel or h == r or not entity_like(r):
                continue
            q = TEMPLATES[rel].format(h=pretty(h))
            norm = normalize_question(q)
            if norm in seen_q:
                continue
            seen_q.add(norm)
            candidates[rel].append(
                {
                    "qid": qid,
                    "triplet": t,
                    "question": q,
                    "norm": norm,
                }
            )
    picked = []
    for rel in sorted(candidates):
        pool = sorted(candidates[rel], key=lambda c: (c["qid"], c["triplet"]["h"]))
        pick = random.sample(pool, min(LOCAL_MAX_PER_TEMPLATE, len(pool)))
        picked.extend(pick)

    random.shuffle(picked)
    rows, final_seen = [], set()
    for cand in picked:
        if cand["norm"] in final_seen or len(rows) >= N_LOCAL:
            continue
        final_seen.add(cand["norm"])
        t = cand["triplet"]
        value = t["r"]
        rows.append(
            {
                "id": None,
                "question": cand["question"],
                "expected_answer": value,
                "expected_entities": sorted({t["h"], t["r"]}),
                "expected_triples": [[t["h"], t["r"], t["t"]]],
                "hops": 1,
                "mode": "local",
                "tags": ["generated_from_triplet", f"triplet:({t['h']}, {t['r']}, {t['t']})", f"doc:{t['source_title']}"],
            }
        )
    return rows


def build_from_split(
    split: str, question_type: str, n: int, mode: str, by_qid: dict[str, list[dict]]
) -> list[dict]:
    """Questions HotpotQA réelles (bridge -> multi-hop, comparison -> global)."""
    random.seed(SEED + 1 if question_type == "bridge" else SEED + 2)
    pool = []
    for row in rows_split(split):
        qid = row["id"]
        if row.get("type") != question_type or qid not in by_qid:
            continue
        support_trips = [t for t in by_qid[qid] if t.get("label") == 1]
        if not support_trips:
            continue
        pool.append((row, support_trips))
    rows = []
    for row, support_trips in random.sample(pool, min(n, len(pool))):
        entities = {
            t[h_or_r]
            for t in support_trips
            for h_or_r in ("h", "r")
            if entity_like(t[h_or_r]) and t[h_or_r] != "has_info"
        }
        expected_triples = [[t["h"], t["r"], t["t"]] for t in support_trips[:MAX_EXPECTED_TRIPLES]]
        rows.append(
            {
                "id": None,
                "question": row["question"],
                "expected_answer": row["answer"],
                "expected_entities": sorted(entities)[:MAX_ENTITIES],
                "expected_triples": expected_triples,
                "hops": 2,
                "mode": mode,
                "tags": [f"hotpotqa:{split}", f"type:{row['type']}", doc_tag(row["supporting"])],
            }
        )
    return rows


def build_abstention(by_qid: dict[str, list[dict]]) -> list[dict]:
    """Questions absentes du KG : le graphe ne peut pas y répondre."""
    rng = random.Random(SEED + 3)
    quotas = {"test": 7, "val": N_ABSTENTION_EXTRA}
    rows = []
    for split in ("test", "val"):
        pool = [r for r in rows_split(split) if r["id"] not in by_qid]
        rng.shuffle(pool)
        for row in pool[: quotas[split]]:
            rows.append(
                {
                    "id": None,
                    "question": row["question"],
                    "expected_answer": row["answer"],
                    "expected_entities": [],
                    "expected_triples": [],
                    "hops": 2,
                    "mode": "hybrid",
                    "tags": ["abstention", "missing_from_graph", f"hotpotqa:{split}", doc_tag(row["supporting"])],
                }
            )
    return rows


def main() -> int:
    by_qid = triplets_by_qid()
    locals_ = build_local(by_qid)
    multihops = build_from_split("test", "bridge", N_MULTIHOP, "hybrid", by_qid)
    globals_ = build_from_split("test", "comparison", N_GLOBAL, "global", by_qid)
    abstains = build_abstention(by_qid)

    rows = locals_ + multihops + globals_ + abstains
    seen_norm = set()
    final = []
    for row in rows:
        norm = normalize_question(row["question"])
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        row["id"] = f"g1-{len(final) + 1:03d}"
        row["expected_entities"] = sorted(set(row["expected_entities"]))
        final.append(row)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for row in final:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(
        "abstention" if "abstention" in row["tags"] else row["mode"] for row in final
    )
    print(f"Écrit {OUT_PATH} : {len(final)} lignes")
    for k in ("local", "hybrid", "global", "abstention"):
        print(f"  {k}: {counts[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
