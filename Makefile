# Makefile — LLMOps socle pour le pipeline GraphRAG existant (src/)
# Le pipeline legacy n'est pas modifié : voir docs/audit_graphrag.md

PY ?= .venv/bin/python

.PHONY: install test lint eval bench validate-dataset accept-dataset-change

install:
	$(PY) -m pip install -q -r requirements.txt
	$(PY) -m pip install -q pytest ruff

test:
	$(PY) -m pytest tests/ -v

# Lint scoping : nouveaux fichiers uniquement.
# tests/test_api.py est legacy (P01 read-only) : exclu du lint.
lint:
	$(PY) -m ruff check llmops/ app/ tests/ --exclude tests/test_api.py scripts/build_golden_dataset.py scripts/validate_dataset.py

# Évaluation pipeline (GPU + checkpoints + data/processed requis)
eval:
	PYTHONPATH=. $(PY) -m src.eval.run_evaluation --config C5 --split val

# Bench latence sur 3 questions (GPU + checkpoints requis)
bench:
	$(PY) scripts/demo.py

# P03 — dataset doré gelé (eval/datasets/golden_v1.jsonl + .sha256)
validate-dataset:
	$(PY) scripts/validate_dataset.py

# Dégel CONSCIENT : réexécuter uniquement après une modification validée du
# dataset ; réécrit golden_v1.sha256, à committer avec le jsonl.
accept-dataset-change:
	$(PY) scripts/validate_dataset.py --update-hash
	@echo "golden_v1.sha256 régénéré — committer eval/datasets/golden_v1.jsonl et .sha256 ensemble"
