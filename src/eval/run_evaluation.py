"""Script d'évaluation unifié pour toutes configurations (J8-J9).

Usage:
    python -m src.eval.run_evaluation --config C1 --split val
    python -m src.eval.run_evaluation --config C2 --split val
    python -m src.eval.run_evaluation --config C3 --split val
    python -m src.eval.run_evaluation --config C4 --split val  # fine-tuned + no context
    python -m src.eval.run_evaluation --config C5 --split val  # fine-tuned + GNN

Configs:
    C1: base model, sans contexte
    C2: base model, top-5 cosinus
    C3: base model, GNN top-k
    C4: fine-tuned model, sans contexte
    C5: fine-tuned model, GNN top-k

Écrit results/{config}_{split}.csv avec colonnes:
    id, question, pred, gold, EM, F1, sources, latence
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from src.eval.metrics import exact_match, f1
from src.llm.load_llm import generate_answer, load_base_model, load_qlora_model
from src.llm.prompt import build_prompt_from_triplets

CONFIGS = {
    "C1": {"model_type": "base", "context": "none"},
    "C2": {"model_type": "base", "context": "cosine"},
    "C3": {"model_type": "base", "context": "gnn"},
    "C4": {"model_type": "ft", "context": "none"},
    "C5": {"model_type": "ft", "context": "gnn"},
}

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "checkpoints/qlora_run1/final"
MAX_NEW_TOKENS = 64
SEED = 42

torch.manual_seed(SEED)


def load_split(split: str) -> list[dict]:
    with open(f"data/processed/{split}.jsonl") as f:
        return [json.loads(line) for line in f]


def load_triplets_by_qid() -> dict[str, list[dict]]:
    from collections import defaultdict
    by_qid = defaultdict(list)
    with open("data/processed/kg_triplets.jsonl") as f:
        for line in f:
            t = json.loads(line)
            by_qid[t["qid"]].append(t)
    return by_qid


def load_cos_topk(split: str) -> dict[str, list[int]]:
    result = {}
    path = f"data/processed/cos_topk_{split}.jsonl"
    if not os.path.exists(path):
        return result
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            result[d["qid"]] = d["topk_indices"]
    return result


def load_gnn_topk(split: str) -> dict[str, list[dict]]:
    result = {}
    path = f"data/processed/gnn_topk_{split}.jsonl"
    if not os.path.exists(path):
        return result
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            result[d["qid"]] = d["triplets"]
    return result


def parse_sources(response: str) -> tuple[str, list[int]]:
    """Parse la réponse pour extraire answer et sources."""
    if "Sources:" in response:
        parts = response.split("Sources:")
        answer = parts[0].strip()
        sources_str = parts[1].strip()
        import re
        sources = [int(x) for x in re.findall(r"\[(\d+)\]", sources_str)]
    else:
        answer = response.strip()
        sources = []
    return answer, sources


def get_context_triplets(
    qid: str,
    qid_to_triplets: dict[str, list[dict]],
    cos_topk: dict[str, list[int]],
    gnn_topk: dict[str, list[dict]],
    context_type: str,
) -> list[dict]:
    """Retourne les triplets de contexte selon le type."""
    triplets = qid_to_triplets.get(qid, [])

    if context_type == "none":
        return []
    elif context_type == "cosine":
        indices = cos_topk.get(qid, [])
        return [triplets[i] for i in indices if i < len(triplets)]
    elif context_type == "gnn":
        return gnn_topk.get(qid, [])
    else:
        return []


def evaluate(
    config: str,
    model,
    tokenizer,
    split_data: list[dict],
    qid_to_triplets: dict[str, list[dict]],
    cos_topk: dict[str, list[int]],
    gnn_topk: dict[str, list[dict]],
) -> list[dict]:
    """Évalue une configuration sur un split."""
    context_type = CONFIGS[config]["context"]
    results = []

    for ex in split_data:
        qid = ex["id"]
        question = ex["question"]
        gold = ex["answer"]

        context_triplets = get_context_triplets(
            qid, qid_to_triplets, cos_topk, gnn_topk, context_type
        )

        prompt = build_prompt_from_triplets(question, context_triplets)

        start = time.time()
        response = generate_answer(model, tokenizer, prompt, MAX_NEW_TOKENS)
        latency = time.time() - start

        pred_answer, pred_sources = parse_sources(response)

        em = exact_match(pred_answer, gold)
        f1_score = f1(pred_answer, gold)

        results.append({
            "id": qid,
            "question": question,
            "pred": pred_answer,
            "gold": gold,
            "EM": int(em),
            "F1": round(f1_score, 4),
            "sources": str(pred_sources),
            "latency": round(latency, 3),
            "raw_response": response,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--split", required=True, choices=["val", "test"])
    args = parser.parse_args()

    config_info = CONFIGS[args.config]
    print(f"=== Config {args.config}: {config_info} ===")

    split_data = load_split(args.split)
    qid_to_triplets = load_triplets_by_qid()
    cos_topk = load_cos_topk(args.split)
    gnn_topk = load_gnn_topk(args.split)

    print(f"Chargement modèle ({config_info['model_type']})...")
    if config_info["model_type"] == "base":
        model, tokenizer = load_base_model("cuda", use_4bit=True)
    else:
        model, tokenizer = load_qlora_model(ADAPTER_PATH, "cuda")

    print(f"Modèle chargé sur {model.device}")
    print(f"Évaluation sur {len(split_data)} exemples ({args.split})...")

    results = evaluate(
        args.config, model, tokenizer, split_data,
        qid_to_triplets, cos_topk, gnn_topk
    )

    # Sauvegarder CSV
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.config}_{args.split}.csv"

    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "question", "pred", "gold", "EM", "F1", "sources", "latency"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print(f"Résultats sauvés: {out_path}")

    # Stats rapides
    avg_em = sum(r["EM"] for r in results) / len(results)
    avg_f1 = sum(r["F1"] for r in results) / len(results)
    avg_lat = sum(r["latency"] for r in results) / len(results)
    format_ok = sum(1 for r in results if "Sources:" in r["raw_response"]) / len(results) * 100

    print(f"  EM: {avg_em:.4f}")
    print(f"  F1: {avg_f1:.4f}")
    print(f"  Latence moy: {avg_lat:.3f}s")
    print(f"  Format OK: {format_ok:.1f}%")


if __name__ == "__main__":
    main()