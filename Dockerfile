# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip

COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ENABLE_LOCAL_MODEL=false \
    LOCAL_MODEL_PATH="" \
    PROMPT_CACHE_ENABLED=false \
    MAX_RUNTIME_SECONDS=600

RUN mkdir -p /input /output

COPY --from=builder /install /usr/local
COPY app/ /app/app/
COPY scripts/docker_entrypoint.sh /app/docker_entrypoint.sh

RUN chmod +x /app/docker_entrypoint.sh

# Grading harness mounts /input and /output; run as root for writable /output.
ENTRYPOINT ["/app/docker_entrypoint.sh"]
