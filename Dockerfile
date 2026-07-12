# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

ENV CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF" \
    FORCE_CMAKE=1

COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
 && find /install -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null || true \
 && find /install -type d -name "__pycache__" -prune -exec rm -rf {} +

FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ENABLE_LOCAL_MODEL=true \
    LOCAL_MODEL_PATH=/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    LOCAL_N_CTX=2048 \
    LOCAL_N_THREADS=2 \
    LOCAL_CALL_TIMEOUT_SECONDS=18 \
    LOCAL_CATEGORIES=sentiment,summarization,ner,factual \
    LOCAL_CONFIDENCE_THRESHOLD=0.75 \
    PROMPT_CACHE_ENABLED=false \
    MAX_RUNTIME_SECONDS=600

RUN mkdir -p /input /output /app/models

ARG GGUF_URL=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
RUN curl -fL --retry 3 --retry-delay 5 -o /app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf "$GGUF_URL"

COPY --from=builder /install /usr/local
COPY app/ /app/app/
COPY scripts/docker_entrypoint.sh /app/docker_entrypoint.sh

RUN chmod +x /app/docker_entrypoint.sh

ENTRYPOINT ["/app/docker_entrypoint.sh"]
