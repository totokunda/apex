# Apex

Apex is a diffusion engine that wraps model manifests, preprocessing/postprocessing, and an API server into a single runnable package.

## Requirements
- Python 3.10+
- Git for pulling the repo and its submodules
- CUDA-capable GPU recommended for fast inference (CPU-only works for some components)

## Quick start
```bash
git clone https://github.com/totokunda/apex.git
cd apex
git submodule update --init --recursive

python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
pip install -e .
```
Or run `bash scripts/install.sh` for a guided setup (CUDA, dependencies, and Python packages).

## Running locally
- Start the dev API server with auto-reload: `python -m src dev` (FastAPI on http://127.0.0.1:8765)
- Start the production stack: `python -m src start -f Procfile`
- Stop running services: `python -m src stop`

## Project layout
- `src/` – core engine code (API, manifests, schedulers, preprocess/postprocess helpers)
- `scripts/` – setup and maintenance scripts
- `manifest/` – model and engine definitions
- `tests/` – lightweight checks and smoke tests

## Running tests
From an activated virtualenv: `pytest`
