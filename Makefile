# Makefile — LLMOps socle pour le pipeline GraphRAG existant (src/)
# Le pipeline legacy n'est pas modifié : voir docs/audit_graphrag.md

PY ?= .venv/bin/python

.PHONY: install test lint eval bench

install:
	$(PY) -m pip install -q -r requirements.txt
	$(PY) -m pip install -q pytest ruff

test:
	$(PY) -m pytest tests/ -v

# Lint scoping : nouveaux fichiers uniquement.
# tests/test_api.py est legacy (P01 read-only) : exclu du lint.
lint:
	$(PY) -m ruff check llmops/ tests/ --exclude tests/test_api.py

# Évaluation pipeline (GPU + checkpoints + data/processed requis)
eval:
	PYTHONPATH=. $(PY) -m src.eval.run_evaluation --config C5 --split val

# Bench latence sur 3 questions (GPU + checkpoints requis)
bench:
	$(PY) scripts/demo.py
