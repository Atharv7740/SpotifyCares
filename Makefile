.PHONY: setup check-llm ingest intents label demo full-eval test clean ui

VENV = .venv
PY = $(VENV)/bin/python

setup:
	python3.12 -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt
	mkdir -p data/raw data/processed data/golden eval/results

check-llm:
	$(PY) -m src.llm

ingest:
	$(PY) -m src.ingest

intents:
	$(PY) -m src.intents

label:
	$(PY) -m tools.label

demo:
	$(PY) -m eval.run_eval --subset 20

ui:
	$(PY) serve.py


full-eval:
	$(PY) -m eval.run_eval

test:
	$(PY) -m pytest tests/ -x

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .ruff_cache
