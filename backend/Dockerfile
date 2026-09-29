# Backend image for Railway (API; later also the retraining cron service).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-group notebooks --no-install-project
COPY . .

# Migrations run on every deploy (additive-only, see CLAUDE.md), then the API.
# --proxy-headers: Railway terminates TLS in front of us; without this, generated
# URLs (OAuth redirects) come out as http:// with the wrong host.
CMD alembic upgrade head && \
    uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} \
    --proxy-headers --forwarded-allow-ips='*'
