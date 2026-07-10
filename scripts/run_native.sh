#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Copy .env.example to .env and set FIREWORKS_API_KEY"
  exit 1
fi

# Prefer project virtualenv (created via: python3.12 -m venv .venv)
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "No Python found. Create a venv first:"
  echo "  python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python3
fi

set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a

mkdir -p output
"$PYTHON" -m app.main run --input ./input/tasks.json --output ./output/results.json
echo "Results:"
cat output/results.json
