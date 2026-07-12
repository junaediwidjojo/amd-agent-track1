#!/usr/bin/env bash
# Simulate hackathon grading VM: linux/amd64, 2 vCPU, 4 GB RAM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="${GRADING_IMAGE:-amd-agent:grading-test}"
INPUT_DIR="${GRADING_INPUT:-$ROOT/input}"
OUTPUT_DIR="${GRADING_OUTPUT:-$ROOT/output/grading-test}"

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is not running. Start Docker Desktop, then rerun this script."
  exit 1
fi

if [[ ! -f "$ROOT/.env" ]]; then
  echo "ERROR: Missing .env — copy .env.example and set FIREWORKS_API_KEY."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR/results.json"

echo "==> Building linux/amd64 image (hybrid local + Fireworks)..."
docker buildx build \
  --platform linux/amd64 \
  -f Dockerfile \
  -t "$IMAGE" \
  "$ROOT"

echo "==> Running under grading constraints (2 CPUs, 4G RAM)..."
set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a

docker run --rm \
  --platform linux/amd64 \
  --cpus=2 \
  --memory=4g \
  --memory-swap=4g \
  --pids-limit=256 \
  -v "$INPUT_DIR:/input:ro" \
  -v "$OUTPUT_DIR:/output" \
  -e "FIREWORKS_API_KEY=${FIREWORKS_API_KEY}" \
  -e "FIREWORKS_BASE_URL=${FIREWORKS_BASE_URL}" \
  -e "ALLOWED_MODELS=${ALLOWED_MODELS}" \
  "$IMAGE"

EXIT_CODE=$?
echo ""
echo "Container exit code: $EXIT_CODE"

if [[ ! -f "$OUTPUT_DIR/results.json" ]]; then
  echo "FAIL: /output/results.json was not written (OUTPUT_MISSING)."
  exit 1
fi

python3 -c "import json; json.load(open('$OUTPUT_DIR/results.json'))"

TASKS=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_DIR/results.json'))))")
echo "OK: Wrote $TASKS results to $OUTPUT_DIR/results.json"

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo "FAIL: Non-zero exit => hackathon RUNTIME_ERROR."
  exit 1
fi

echo "PASS: Grading simulation succeeded. Safe to push and resubmit."
