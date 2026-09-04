# eeg-sleep-stager — convenience targets.
# `make demo` runs the whole pipeline end-to-end on a small subject subset.

CONFIG ?= config.yaml
SUBJECTS ?= 6

.PHONY: install test lint demo demo-web sample ingest etl split train evaluate clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

ingest:
	eeg-sleep-stager ingest --limit $(SUBJECTS) --config $(CONFIG)

etl:
	eeg-sleep-stager etl --master "local[*]" --config $(CONFIG)

split:
	eeg-sleep-stager split --config $(CONFIG)

train:
	eeg-sleep-stager train --model cnn --config $(CONFIG)
	eeg-sleep-stager train --model baseline --config $(CONFIG)

evaluate:
	eeg-sleep-stager evaluate --model cnn --config $(CONFIG)

# S2 -> S7 on a small subset.
demo: ingest etl split train evaluate

# Build the bundled demo sample from the epoch store (one held-out test night).
sample:
	python scripts/make_sample.py --config $(CONFIG)

# Build the sample if missing, then launch the Gradio web demo.
demo-web: app_assets/sample_night.npz
	python app.py

app_assets/sample_night.npz:
	python scripts/make_sample.py --config $(CONFIG)

clean:
	rm -rf data/processed models reports .pytest_cache
