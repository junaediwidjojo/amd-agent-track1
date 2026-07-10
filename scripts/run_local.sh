#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Copy .env.example to .env and set FIREWORKS_API_KEY"
  exit 1
fi

mkdir -p output
docker compose build
docker compose run --rm agent

echo "Results written to output/results.json"
cat output/results.json
