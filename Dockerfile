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
    binutils \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

ENV CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF" \
    FORCE_CMAKE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt \
 && find /install -type d -name "__pycache__" -prune -exec rm -rf {} + \
 && find /install -type d -name "tests" -prune -exec rm -rf {} + \
 && find /install -type d -name "test" -prune -exec rm -rf {} + \
 && find /install -name "*.a" -delete \
 && find /install -name "*.pyc" -delete \
 && find /install -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null || true

FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LOCAL_MODEL_PATH=/app/models/qwen2.5-3b-instruct-q4_k_m.gguf \
    LOCAL_N_CTX=2048 \
    LOCAL_N_THREADS=2 \
    LOCAL_CATEGORIES=sentiment,summarization,ner \
    LOCAL_CONFIDENCE_THRESHOLD=0.75 \
    PROMPT_CACHE_ENABLED=false \
    MAX_RUNTIME_SECONDS=600

RUN useradd --create-home --shell /bin/bash --uid 10001 agent \
 && mkdir -p /app/models /app/output \
 && chown agent:agent /app /app/models /app/output

COPY --from=builder /install /usr/local
COPY --chown=agent:agent models/qwen2.5-3b-instruct-q4_k_m.gguf /app/models/qwen2.5-3b-instruct-q4_k_m.gguf
COPY --chown=agent:agent app/ /app/app/

USER agent

ENTRYPOINT ["python", "-m", "app.main", "run"]