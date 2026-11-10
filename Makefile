# Makefile — LLMOps socle pour le pipeline GraphRAG existant (src/)
# Le pipeline legacy n'est pas modifié : voir docs/audit_graphrag.md

PY ?= .venv/bin/python

.PHONY: install test lint eval eval-legacy bench compare validate-dataset accept-dataset-change

install:
	$(PY) -m pip install -q -r requirements.txt
	$(PY) -m pip install -q pytest ruff

test:
	$(PY) -m pytest tests/ -v --cov=llmops --cov=eval --cov=ops --cov-fail-under=70

# Lint scoping : nouveaux fichiers uniquement.
# tests/test_api.py est legacy (P01 read-only) : exclu du lint.
lint:
	$(PY) -m ruff check llmops/ app/ eval/run_ragas.py tests/ --exclude tests/test_api.py scripts/build_golden_dataset.py scripts/validate_dataset.py

# Évaluation pipeline legacy (GPU + checkpoints + data/processed requis)
eval-legacy:
	PYTHONPATH=. $(PY) -m src.eval.run_evaluation --config C5 --split val

# P04 — éval RAGAS reproductible (2 étapes, envs isolés, docs/audit_graphrag.md G11) :
#   collect : venv repo  (.venv, torch)  -> reports/stages/contexts_latest.json
#   judge   : venv eval  (ragas 0.1.21, litellm + MISTRAL_API_KEY) -> reports/eval_*.json
EVAL_PY = eval/.venv/bin/python

eval-bootstrap:
	bash eval/bootstrap.sh

eval: eval-bootstrap
	PYTHONPATH=. $(PY) eval/run_ragas.py collect
	PYTHONPATH=. $(EVAL_PY) eval/run_ragas.py judge

# Variante sans GPU / sans clé Mistral (n'importe quelle machine) :
# adapter _FakeAdapter + métriques RAGAS nulles (--no-judge), sortie 0 attendue.
eval-smoke: eval-bootstrap
	PYTHONPATH=. $(PY) eval/run_ragas.py collect --adapter fake --limit 6
	PYTHONPATH=. $(EVAL_PY) eval/run_ragas.py judge --no-judge

# Benchmark du dataset frozen (P05) :
# - `make bench` : exécute eval si le dernier rapport a plus de 24h,
#   puis lance scripts/bench.py pour réplayer le dataset et écrire
#   reports/baseline.json + reports/BASELINE.md
# - `make compare` : affiche side-by-side baseline vs bench_latest.json
bench:
	@bash eval/bootstrap.sh 2>/dev/null || true
	PYTHONPATH=. $(PY) scripts/bench.py

# Compare baseline vs latest bench
compare:
	@echo "=== Baseline vs Latest Bench ==="
	@echo ""
	@if [ -f reports/baseline.json ]; then \
		echo "--- Baseline ---"; \
		python3 -c "import json; d=json.load(open('reports/baseline.json')); m=d['metrics']; print(f'p50: {m[\"p50_latency_ms\"]}ms, p95: {m[\"p95_latency_ms\"]}ms, entity_recall: {m[\"entity_recall\"]}, faithfulness: N/A'); "; \
	else \
		echo "No baseline.json found — run 'make bench' first"; \
	fi
	@if [ -f reports/bench_latest.json ]; then \
		echo "--- Latest Bench ---"; \
		python3 -c "import json; d=json.load(open('reports/bench_latest.json')); m=d['metrics']; print(f'p50: {m[\"p50_latency_ms\"]}ms, p95: {m[\"p95_latency_ms\"]}ms, entity_recall: {m[\"entity_recall\"]}, faithfulness: N/A'); "; \
	else \
		echo "No bench_latest.json found — run 'make bench' first"; \
	fi
	@echo ""

# P03 — dataset doré gelé (eval/datasets/golden_v1.jsonl + .sha256)
validate-dataset:
	$(PY) scripts/validate_dataset.py

# Dégel CONSCIENT : réexécuter uniquement après une modification validée du
# dataset ; réécrit golden_v1.sha256, à committer avec le jsonl.
accept-dataset-change:
	$(PY) scripts/validate_dataset.py --update-hash
	@echo "golden_v1.sha256 régénéré — committer eval/datasets/golden_v1.jsonl et .sha256 ensemble"

# Promouvoir une amélioration validée vers la nouvelle baseline.
# Usage: make update-baseline
# - Exécute `make bench` pour générer les nouveaux métriques (baseline.json + BASELINE.md)
# - L'utilisateur examine les rapports, puis commit avec un message signé :
#     git add reports/ baseline.json BASELINE.md
#     git commit -m "chore(bench): promote baseline after improvement — <short description>"
#     git tag -a baseline-$(shell git rev-parse --short HEAD) -m "New baseline"
update-baseline:
	@bash eval/bootstrap.sh 2>/dev/null || true
	$(PY) scripts/bench.py
	@echo ""
	@echo "=== NEW BASELINE GENERATED ==="
	@echo "Please review reports/baseline.json and reports/BASELINE.md"
	@echo "When satisfied, commit with:"
	@echo "  git add reports/ baseline.json BASELINE.md"
	@echo "  git commit -m 'chore(bench): promote baseline after improvement — <description>'"
	@echo "  git tag -a baseline-$(shell git rev-parse --short HEAD) -m 'New baseline'"
