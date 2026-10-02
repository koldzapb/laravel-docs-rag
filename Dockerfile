# --- Build stage: install dependencies with uv and pre-download the ONNX models ---------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 \
    MODEL_CACHE_DIR=/opt/models
WORKDIR /app

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# Bake the embedding + reranking models into the image so containers start offline.
RUN /app/.venv/bin/rag download-models

# --- Runtime stage: just Python, the virtualenv and the models -------------------------------
FROM python:3.12-slim-bookworm

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data && chown app:app /data
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder --chown=app:app /opt/models /opt/models

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    MODEL_CACHE_DIR=/opt/models \
    HF_HUB_OFFLINE=1
USER app
WORKDIR /home/app
EXPOSE 8000

ENTRYPOINT ["rag"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
