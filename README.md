# QA Multi-Sauts : Retrieval Appris (GNN) et LLM Fine-tuné (QLoRA)

Pipeline : KG (REBEL + Neo4j) → GNN GATv2 de scoring des triplets → LLM fine-tuné QLoRA.
Dataset : HotpotQA (distractor). Environnement : RTX 3050 Ti 4 Go, Ubuntu.

## Architecture du Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   REBEL     │───▶│   Neo4j     │───▶│   GNN       │───▶│   Top-k     │───▶│   Qwen2.5   │
│  Extraction │    │  Stockage   │    │  GATv2      │    │  Selection  │    │  QLoRA 4-bit│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     │                  │                 │                 │                 │
  18k triplets      17k nœuds         GATv2 + BM25      Top-5 triplets    1.5B params
  2000 questions    43k arêtes        R@5=0.7292        + contexte         4-bit NF4
  HotpotQA          1-2 sauts         Val R@5=0.729     pour LLM           0.6s/req
```

## Quickstart (5 commandes)

```bash
# 1. Installer
git clone <repo> && cd qa-multisaut
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Données + Modèles (si modèles pré-entraînés disponibles)
# Sinon: voir section Reproduction Complète ci-dessous
docker compose up -d neo4j && sleep 20

# 3. Lancer l'API
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1

# 4. Tester l'API
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'

# 5. Lancer les tests
.venv/bin/python -m pytest tests/ -v
```

## Métriques Clés (Test - 300 questions)

| Métrique | Test | Val | Δ | Source |
|----------|------|-----|---|--------|
| **EM** | **0.1633** | 0.1950 | -0.0317 | j12 |
| **F1** | **0.2636** | 0.2784 | -0.0148 | j12 |
| **Format OK** | **99.7%** | 100.0% | -0.3% | j12 |
| **Latence moy** | **0.607s** | 0.618s | -0.011s | j12 |
| **Supporting Facts Recall@5** | **0.9009** | 0.9357 | -0.0348 | j12 |
| **Hallucination (emb 0.5)** | N/A | **41%** | — | j14 |

## Ablations (VAL - 140 questions évaluées)

| Configuration | R@3 | R@5 | R@10 | Rang |
|---------------|-----|-----|------|------|
| **GNN A8 (GATv2+BM25)** | **0.5971** | **0.7292** | 0.8917 | 1 |
| BM25 seul | 0.5339 | 0.7074 | 0.8905 | 2 |
| Neo4j sans GNN (1-2 hops + BM25) | 0.5175 | 0.6634 | 0.8685 | 3 |
| Neo4j + Cross-Encoder | 0.4180 | 0.4426 | 0.4572 | 4 |

**Gain GNN vs BM25 seul : +0.0218 R@5**

## Comparaisons Clés

| Comparaison | Résultat | Source |
|-------------|----------|--------|
| Run1 vs Run2 | Run1 F1 0.278 > Run2 0.249 (surapprentissage Run2) | j10 |
| Base vs Fine-tuné | Base 0.024 → Run1 0.278 (**+0.254**) | j10 |
| Fusionné vs Adaptateur | 1.81× plus rapide, même VRAM | j10 |
| Hallucination | 41% (emb 0.5) vs 76% (lexical) | j14 |

## Limites Connues

1. **Labelling paragraph-level** : 30.75% questions sans triplet positif (58.7% sur 2000 questions) → borne structurelle du recall
2. **REBEL imparfait** : 3 966 relations uniques non normalisées, extraction bruitée
3. **GPU 4 Go** : contraint batch size, sequence length, élimine DPO/full fine-tuning
4. **Hallucination 41%** : modèle 1.5B + peu de données SFT (686 ex, 2 epochs)
5. **Abstention imparfaite** : seuil `max_score_threshold=0.0` inefficace (sigmoid ≥ 0)
6. **Pas d'évaluation test pour ablations** : calculées sur val uniquement
7. **Entity detection basique** : substring matching ("county" générique → 3 910 triplets bruit)

## Installation

```bash
# Cloner
git clone <repo>
cd qa-multisaut

# Créer environnement
python -m venv .venv
source .venv/bin/activate

# Dépendances
pip install -r requirements.txt

# Modèles HuggingFace (premiers lancements)
# REBEL, Qwen2.5-1.5B, bge-small seront téléchargés automatiquement
```

## Reproduction Complète (dans l'ordre)

### 1. Données HotpotQA → splits
```bash
PYTHONPATH=. .venv/bin/python src/prep/sample_splits.py
# Crée data/processed/train.jsonl (1500), val.jsonl (200), test.jsonl (300)
```

### 2. Extraction REBEL (triplets bruts)
```bash
PYTHONPATH=. .venv/bin/python src/kg/run_extraction.py
# ~22 min total (train 14min + test 8min, val déjà fait)
# Crée data/processed/kg_raw_{split}_part*.jsonl
```

### 3. Normalisation + Étiquetage
```bash
PYTHONPATH=. .venv/bin/python src/kg/normalize_label.py
# Crée data/processed/kg_triplets.jsonl (18897 triplets)
# Rapport: reports/j4_kg_stats.md
```

### 4. Neo4j (optionnel, pour ablation B/C)
```bash
docker compose up -d neo4j
sleep 20
PYTHONPATH=. .venv/bin/python src/kg/load_neo4j.py
# 17257 nœuds, 43040 arêtes
# Rapport: reports/j11_neo4j.md
```

### 5. Construction datasets PyG (pour GNN)
```bash
PYTHONPATH=. .venv/bin/python src/gnn/dataset.py
# Crée data/processed/pyg_{train,val,test}.pt
```

### 6. Entraînement GNN (abslations A1-A8)
```bash
PYTHONPATH=. .venv/bin/python src/gnn/train.py --config A1_gatv2_h128_l2_avec_question
# Best: A8_gatv2_h128_l2_bm25 (config A8 + feature BM25)
# Checkpoint: checkpoints/gnn_best.pt
# Rapports: reports/j7_ablations.md, j7_decision.md
```

### 7. Prédictions top-k (val + test)
```bash
PYTHONPATH=. .venv/bin/python src/gnn/predict_topk.py
# Crée data/processed/gnn_topk_{val,test}.jsonl
```

### 8. Construction dataset SFT
```bash
PYTHONPATH=. .venv/bin/python src/llm/build_sft.py
# Crée data/processed/sft_train.jsonl (686 ex)
# Rapports: reports/j8_sft_stats.md, j8_sft_samples.md
```

### 9. Fine-tuning QLoRA (Run1)
```bash
PYTHONPATH=. .venv/bin/python src/llm/train_qlora.py --lora_r 16 --epochs 2 --lr 2e-4
# Checkpoints: checkpoints/qlora_run1/final/
# Modèle fusionné: checkpoints/merged_final/
```

### 10. Évaluation Baselines LLM (val)
```bash
PYTHONPATH=. .venv/bin/python src/llm/eval_baselines.py
# C1: sans contexte, C2: cosinus top-5, C3: GNN top-5
# Rapport: reports/j8_baselines_llm.md
```

### 11. Évaluation Fine-tuné (val)
```bash
PYTHONPATH=. .venv/bin/python src/llm/eval_finetuned.py
# Compare Base+GNN vs Run1 vs Run2
# Rapport: reports/j10_selection.md (Run1 sélectionné)
```

### 12. Évaluation Finale (test)
```bash
PYTHONPATH=. .venv/bin/python -m src.eval.run_evaluation --config C5 --split test
# Résultats: results/C5_test.csv
# Rapport: reports/j12_final_test.md
```

### 13. Ablations Finales
```bash
# BM25 seul, Neo4j sans GNN, Neo4j + Cross-Encoder
# Résultats dans reports/j13_ablations_final.md
```

### 14. Taux d'Hallucination
```bash
# Juge embedding bge-small sur 100 réponses val
# Rapport: reports/j14_hallucination.md
```

## Lancement de l'API

```bash
# Nécessite: Neo4j running, modèles entraînés
docker compose up -d neo4j
sleep 20

# Lancer API (1 worker, batch 1 pour VRAM 4GB)
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Endpoints
- `GET /health` → statut + version + modèle chargé
- `POST /ask` `{question}` → `{answer, sources[], latency_ms, config, abstained}`
- `GET /docs` → Swagger UI

### Test API
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

## Démo Locale (sans API)

```bash
.venv/bin/python scripts/demo.py
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

## Structure du Dépôt

```
qa-multisaut/
├── checkpoints/           # Modèles entraînés
│   ├── gnn_best.pt        # GNN A8 (gatv2_h128_l2 + BM25)
│   ├── qlora_run1/final/  # LoRA Run1 (r=16, 2ep)
│   ├── qlora_run2/final/  # LoRA Run2 (r=32, 4ep)
│   └── merged_final/      # Modèle fusionné 4-bit (Run1)
├── configs/
│   └── config.yaml        # Config globale
├── data/
│   ├── processed/         # Données intermédiaires (gitignored)
│   │   ├── train/val/test.jsonl
│   │   ├── kg_triplets.jsonl
│   │   ├── pyg_{train,val,test}.pt
│   │   ├── gnn_topk_{val,test}.jsonl
│   │   └── sft_train.jsonl
│   └── raw/               # Données brutes (gitignored)
├── reports/               # Rapports d'étapes
│   ├── j4_kg_stats.md
│   ├── j5_baseline_triplet.md
│   ├── j7_ablations.md
│   ├── j7_decision.md
│   ├── j8_baselines_llm.md
│   ├── j8_sft_stats.md
│   ├── j8_sft_samples.md
│   ├── j10_selection.md
│   ├── j11_neo4j.md
│   ├── j12_final_test.md
│   ├── j13_ablations_final.md
│   └── j14_hallucination.md
├── results/               # CSV d'évaluation (gitignored)
├── scripts/
│   └── demo.py            # Démo 3 exemples
├── src/
│   ├── api/               # FastAPI
│   ├── eval/              # Évaluation
│   ├── gnn/               # GNN (model, train, dataset, predict)
│   ├── kg/                # Knowledge Graph (REBEL, Neo4j, normalize)
│   ├── llm/               # LLM (prompt, load, train, eval)
│   ├── orchestrator/      # LangGraph pipeline
│   ├── pipeline/          # (vide, réservé)
│   ├── prep/              # Préparation données
│   └── retrieval/         # Baselines BM25/cosinus
├── tests/
│   └── test_api.py        # Tests API
├── mlruns/                # MLflow (gitignored)
├── docker-compose.yml     # Neo4j
├── requirements.txt
├── requirements.lock      # Versions exactes
├── STATUS.md              # État d'avancement
├── rapport.md             # Rapport final
└── README.md
```

## Versions Clés (reproductibilité)

```bash
# Générer le lock
pip freeze > requirements.lock
```

Versions utilisées :
- torch 2.14.0
- torch_geometric 2.6.1
- transformers 4.44.0
- peft 0.12.0
- trl 0.11.1
- bitsandbytes 0.43.3
- sentence-transformers 3.0.1
- langgraph 0.2.0
- fastapi 0.112.0
- neo4j 5.15.0
- python 3.14

## Graines Fixées
- `SEED = 42` partout (Python, NumPy, PyTorch, random)
- Split HotpotQA : seed 42 (sample_splits.py)
- Dataset SFT : seed 42 (build_sft.py)
- Entraînement GNN/QLoRA : seed 42

## Licence
MIT🔗 Modèle (adaptateurs LoRA) : https://huggingface.co/2Mohamed1/qa-multisaut-qlora
🔗 Modèle entraîné (adaptateurs LoRA) : https://huggingface.co/2Mohamed1/qa-multisaut-qlora

## Dataset doré gelé (P03) — golden_v1

`eval/datasets/golden_v1.jsonl` : 56 lignes QA annotées graphe, construites depuis le corpus réel
HotpotQA + KG (`scripts/build_golden_dataset.py`, seed 42). Distribution : 22 local (1 saut),
16 hybrid (bridge, 2 docs), 9 global (comparatives), 9 abstention (absentes du KG).
Chaque ligne : `{id, question, expected_answer, expected_entities, expected_triples, hops, mode, tags}` ;
les réponses sont vérifiables sur les documents référencés dans `tags` (`doc:`/`docs:`).

**Gel** : `eval/datasets/golden_v1.sha256` ; test pytest `tests/test_dataset_frozen.py` échoue si le
JSONL est modifié sans dégel.

```bash
make validate-dataset          # schéma + unicité + distribution (56 lignes)
make accept-dataset-change     # dégel conscient : régénère le sha256 (à committer avec le jsonl)
```
aucune ligne ne doit être modifiée après le commit de gel sans passer par `make accept-dataset-change`.
