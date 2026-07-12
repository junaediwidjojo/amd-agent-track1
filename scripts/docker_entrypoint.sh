#!/bin/sh
# Hackathon entrypoint — must always exit 0 after writing results.json.

mkdir -p /output 2>/dev/null || true

if [ "${ENABLE_LOCAL_MODEL:-true}" != "false" ] && [ -f "${LOCAL_MODEL_PATH:-/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}" ]; then
  export ENABLE_LOCAL_MODEL=true
  export LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
else
  export ENABLE_LOCAL_MODEL=false
  export LOCAL_MODEL_PATH=""
fi

python -m app.main run
status=$?

if [ ! -f /output/results.json ]; then
  printf '[]\n' > /output/results.json 2>/dev/null || true
fi

exit 0
