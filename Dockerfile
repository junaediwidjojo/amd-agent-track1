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

# Force a release build (no debug symbols) — an unpinned llama-cpp-python
# source build defaults to debug symbols and can bloat the install by 1-3GB.
ENV CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF" \
    FORCE_CMAKE=1

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
 && find /install -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null || true \
 && find /install -type d -name "__pycache__" -prune -exec rm -rf {} + \
 && find /install -type d -name "tests" -prune -exec rm -rf {} + \
 && find /install -type d -name "test" -prune -exec rm -rf {} +

# Download a small 3B 4-bit quantized model for local inference
# Qwen2.5-3B-Instruct Q4_K_M ~2.0 GB — fits comfortably in 4 GB RAM
RUN mkdir -p /install/models && \
    curl -L -o /install/models/qwen2.5-3b-instruct-q4_k_m.gguf \
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LOCAL_MODEL_PATH=/app/models/qwen2.5-3b-instruct-q4_k_m.gguf \
    LOCAL_N_THREADS=2 \
    LOCAL_CATEGORIES=sentiment,summarization,ner,factual \
    LOCAL_CONFIDENCE_THRESHOLD=0.75

COPY --from=builder /install/models ./models
COPY --from=builder /install /usr/local
COPY app/ ./app/

RUN useradd --create-home --shell /bin/bash agent && \
    chown -R agent:agent /app

USER agent

ENTRYPOINT ["python", "-m", "app.main", "run"]
