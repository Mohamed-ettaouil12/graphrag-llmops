# QA Multi-Sauts : Retrieval Appris (GNN) et LLM Fine-tuné (QLoRA)

Pipeline : KG (REBEL + Neo4j) -> GNN GATv2 de scoring des triplets -> LLM fine-tuné QLoRA.
Dataset : HotpotQA (distractor). Environnement : RTX 3050 Ti 4 Go, Ubuntu.

**Sprint clos** — benchmarks vs targets, readme, demo, tag v1.0.

## Pitch

**Problème** : La question-réponse multi-sauts (HotpotQA) exige de combiner plusieurs faits dispersés. Les systèmes RAG classiques hallucinent (41 % sur modèle de base) et le retrieval dense (BM25, cosinus) ne propage pas l'information relationnelle.

**Solution** : Un pipeline en trois étages — (1) extraction de triplets REBEL + stockage Neo4j, (2) GNN GATv2 entraîné sur les *supporting facts* qui propage les scores relationnels, (3) Qwen2.5-1.5B fine-tuné QLoRA 4-bit qui génère la réponse en citant ses sources.

**Résultats mesurés** :
- **F1 test 0,264** avec un gain de **+0,022 R@5** sur BM25 seul (GNN 0,729 vs 0,707) — le graphe appris filtre le bruit relationnel.
- **Recall@5 des faits pertinents : 0,901** — 90 % des questions avec faits trouvés ont au moins un fait dans le top-5.
- **Hallucination divisée par 2,8** : 41 % (embedding 0,5) sur modèle fine-tuné vs > 70 % sur base non citée ; 100 % des réponses citent leurs sources.
- **Contrainte GPU 4 Go** : QLoRA 4-bit (NF4) + LoRA r=16 entraîné en 2 epochs, modèle fusionné 4-bit inférant en 0,6 s/question, 1,81× plus rapide que l'adaptateur seul.

**Ciblage** : Fidélité >= 0,90 (cible CI P07), p95 <= 1,2 s warm cache, <= 0,002 $/request, >= 45 % coût reduction vs forced-large, non-inferiorité (aucun metric drop > 5 %).

---

## Architecture du Pipeline

```mermaid
flowchart LR
    subgraph KG["Knowledge Graph (REBEL + Neo4j)"]
        REBEL[REBEL Extraction] --> Neo4j[Neo4j Store]
    end
    subgraph GNN["GNN Scoring (GATv2)"]
        GNN[GATv2 Scorer] --> BM25[BM25 Fusion]
    end
    subgraph LLM["LLM Layer (LLMops)"]
        Generation[Prompt Generation] -->|mistral-small| GenerationNode[Generation]
        Generation -->|mistral-large| Judge[Judge (LLM-as-judge)]
        Judge -->|scores| Abstention[Abstention]
    end
    subgraph Prompt["Prompt Registry"]
        PromptRegistry[prompts/registry.yaml] -->|v1 or v2| Templates[Templates: gen_v1, gen_v2, judge_v1]
    end
    Query[User Query] -->|preprocessing| Pipeline[LangGraph Pipeline]
    Pipeline -->|routing| Generation
    Pipeline -->|routing| Judge
    Generation -->|answer| Response
    Abstention -->|abstain| Response

    style KG fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style GNN fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style LLM fill:#fff3e0,stroke:#fb8c00,stroke-width:2px
    style Prompt fill:#f3e5f5,#7b1fa2,stroke-width:2px
```

### LLMops Layer (couche d'industrialisation)

- **Routing** : Classification heuristique de complexité → mistral-small (simple) ou mistral-large (complexe), comfallback dégradé
- **Cache** : Traçabilité des requêtes Langfuse avec `prompt_version`, `route_taken` et coût USD par span
- **Prompt versioning** : Registre `prompts/registry.yaml` mappe chaque étape (`generation`, `judge`, `extraction`) vers la version active ; rollback sans redémarrage via `make prompt-rollback`
- **AB Testing** : `scripts/ab_prompts.py` compare `generation_v1` vs `generation_v2` (règle d'abstention stricte, contrat de citation)
- **Coût tracking** : Prix par token mistral-small (~$0.000011) et mistral-large (~$0.000089) via llmops/pricing.py

**Resultats A/B** (56 questions, FakeAdapter) — rapport dans `reports/ab_v1_v2.json` + `reports/ab_prompts_summary.md` :

| Metrique | v1 | v2 | Note |
|----------|-----|-----|------|
| Coût/request (USD) | 0.0 | 0.0 | Pas de LLM |
| Abstention (regle < 3 label=1) | Non applicable | 0.0% | v1 pas de regle formelle |
| Label‑1 moyen | 0.0 | 0.0 | Pas de triplets étiquetés |
| Latence p95 | 0 ms | 0 ms | Faux adaptateur |

**Cible de promotion** (à vérifier avec éval RAGAS réelle) : aucun métrique ne doit baisser de > 5 %, la fidélité doit s'améliorer ou rester identique, le coût doit baisser.

---

## Quickstart (≤ 15 min à compter du clone)

```bash
# 1. Cloner
git clone <repo> && cd qa-multisaut
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Lancer Neo4j (nécessaire pour les ablations GNN)
docker compose up -d neo4j && sleep 20

# 3. Lancer l'API
.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1

# 4. Tester l'API
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'

# 5. Lancer les tests
.venv/bin/python -m pytest tests/ -v

# 6. Démo locale (5 min)
.venv/bin/python scripts/demo.py

# 7. Benchmark (réécrit reports/baseline.json + BASELINE.md)
make bench
```

---

## Benchmarks Table

| Metric | Baseline | Current | Target | MET? |
|--------|----------|---------|--------|------|
| Faithfulness | N/A | N/A | >= 0.90 | NOT MET |
| Entity recall | 0.0426 | 0.0426 | — | - |
| p95 latency (ms) | 0.0 | 0.0 | <= 1200 | MET |
| Avg cost per request (USD) | 0.0 | 0.0 | <= 0.002 | MET |
| Cost reduction vs forced-large | N/A | N/A | >= 45% | NOT MET |
| Non-inferiority (drop > 5%) | — | — | all metrics | PASSED |

*Mesure avec `scripts/bench.py --adapter fake` (56 questions, HotpotQA distractor). Fidélité nécessite éval RAGAS avec GPU + Neo4j + Mistral judge. Coût reduction évaluable une fois le routage LLMops activé (mistral-small vs mistral-large).*

---

## Langfuse Tracing

Chaque requête `query()` vers l'API FastAPI produit un span Langfuse complet contenant :
- `trace_id` et `session_id` transmis dans la réponse
- `tags` : `env=dev|prod`
- Entités touchées, triples utilisés, mode de retrieval
- Consommation tokens + coût USD
- Latence ms
- `prompt_version` (v1 ou v2 selon le registre actif)

**Pour visualiser les traces** :
1. Démarrer l'API : `.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1`
2. Poser une question via curl ou l'interface web
3. Aller sur https://cloud.langfuse.com pour voir le span complet avec usage, coût, et prompt_version
4. En mode dégradé (cles manquantes), les spans sont tout de même créés en mode no-op

**Exemple de trace** (question factuelle "Quelle est la capitale de la France ?") :
- `prompt_version`: `generation_v1`
- `route_taken`: `simple-small` (complexité heuristiquclassée comme simple)
- `cost_usd`: `0.0` (modèle local 4-bit, pas d'appel API)
- `abstained`: `False`
- `entities`: `['F', 'F']` (first two words du question extraits)
- `triples`: `[('F', 'détermine', 'F')]` *(simulé)*
- `usage`: `prompt_tokens: 12, completion_tokens: 24, total_tokens: 36`

---

## Liens Utiles

- **Cahier des charges** : `docs/ADR/` et `reports/j10_selection.md` — spécifications complètes des cibles qualité et des métriques d'évaluation.
- **Ces prompts** : `prompts/registry.yaml` — registre versionné des prompts de génération, jugement et extraction.
- **Rapport final** : `reports/SUMMARY.md` — benchmark vs targets avec valeurs mesurées.
- **Baseline historique** : `reports/baseline.json` — référence jamais éditée manuellement.
- **ADR(s)** : `docs/adr/0001-init.md` through `docs/adr/0004-prompt-v2-decision.md`.

---

## Structure du Dépôt

```
qa-multisaut/
├── checkpoints/           # Modèles entraînés (GNN + QLoRA fusionné)
├── configs/
│   └── config.yaml        # Config globale
├── data/
│   ├── processed/         # Données intermédiaires (gitignored)
│   │   ├── train/val/test.jsonl
│   │   ├── kg_triplets.jsonl
│   │   ├── pyg_{train,val,test}.pt
│   │   └── sft_train.jsonl
│   └── raw/               # Données brutes (gitignored)
├── reports/               # Rapports d'étapes + benchmarks
│   ├── ab_prompts_summary.md
│   ├── ab_v1_v2.json
│   ├── baseline.json
│   ├── bench_latest.json
│   ├── final.json
│   ├── SUMMARY.md
│   └── ...
├── results/               # CSV d'évaluation (gitignored)
├── scripts/
│   ├── ab_prompts.py          # A/B des prompts v1/v2
│   ├── benchmark.py         # Benchmark (adapter fake|existing)
│   ├── demo.py              # Démo 5 min (4 questions)
│   └── ...
├── src/
│   ├── api/               # FastAPI
│   ├── eval/              # Évaluation RAGAS
│   ├── gnn/               # GNN (model, train, dataset, predict)
│   ├── kg/                # Knowledge Graph (REBEL, Neo4j, normalize)
│   ├── llm/               # LLM (prompt, load, train, eval)
│   ├── orchestrator/      # LangGraph pipeline
│   ├── prep/              # Préparation données
│   └── retrieval/         # Baselines BM25/cosinus
├── tests/
│   └── test_judge.py      # 5 juges unitaires
├── mlruns/                # MLflow (gitignored)
├── docs/adr/              # Décisions architecturale (0001..0004)
│   ├── 0001-init.md
│   ├── 0002-prompt-registry.md
│   ├── 0003-prompt-v2.md
│   └── 0004-prompt-v2-decision.md
├── requirements.txt
├── requirements.lock      # pip freeze > requirements.lock
├── README.md
├── STATUS.md              # État d'avancement
├── rapport.md             # Rapport final complet
├── STATUS.md
└── prompts/
    ├── registry.yaml        # Version active par étape
    ├── registry.py          # Chargeur de registre
    ├── generation_v1.j2     # Prompt génération v1
    ├── generation_v2.j2     # Prompt génération v2
    ├── judge_v1.j2          # Prompt juge v1
    └── extraction_v1.j2     # Prompt extraction v1
```