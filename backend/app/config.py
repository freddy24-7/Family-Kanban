"""All configuration comes from environment variables (see ../.env.example)."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Local dev: load the repo-root .env. In production the host sets real env vars.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

log = logging.getLogger(__name__)


def _env(name: str, default: str = "") -> str:
    """Env var with surrounding whitespace stripped; unset OR empty -> default.
    (Hosting dashboards make it easy to save a blank value by accident.)"""
    value = os.environ.get(name, "").strip()
    return value or default


def _env_number[T: (int, float)](name: str, default: T, kind: type[T]) -> T:
    raw = _env(name)
    if not raw:
        return default
    try:
        return kind(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be a {kind.__name__}, got {raw!r}") from None


APP_ENV = _env("APP_ENV", "development")
IS_PRODUCTION = APP_ENV == "production"


def _normalise_database_url(url: str) -> str:
    """Hosts like Railway inject postgresql:// (or postgres://), which SQLAlchemy
    maps to the psycopg2 driver. We use psycopg 3."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


if IS_PRODUCTION and not _env("DATABASE_URL"):
    # Without this, production would silently try the local dev database.
    raise RuntimeError("DATABASE_URL must be set in production")
DATABASE_URL = _normalise_database_url(
    _env("DATABASE_URL", "postgresql+psycopg://famkanban:famkanban@localhost:5433/famkanban")
)

# Batch CLIs set DB_POOLED=false before importing app.db (see app/db.py).
DB_POOLED = _env("DB_POOLED", "true").lower() != "false"

AUTH_SECRET = _env("AUTH_SECRET")
if len(AUTH_SECRET) < 32:
    if IS_PRODUCTION:
        raise RuntimeError("AUTH_SECRET must be set to at least 32 characters in production")
    AUTH_SECRET = "dev-only-insecure-secret-do-not-use-in-production"
    log.warning("AUTH_SECRET missing or short; using an insecure development secret")

ACCESS_TOKEN_LIFETIME_SECONDS = _env_number("ACCESS_TOKEN_LIFETIME_SECONDS", 60 * 60 * 24 * 30, int)
INVITE_LIFETIME_DAYS = _env_number("INVITE_LIFETIME_DAYS", 7, int)

GOOGLE_OAUTH_CLIENT_ID = _env("GOOGLE_OAUTH_CLIENT_ID")
GOOGLE_OAUTH_CLIENT_SECRET = _env("GOOGLE_OAUTH_CLIENT_SECRET")

RESEND_API_KEY = _env("RESEND_API_KEY")
EMAIL_FROM = _env("EMAIL_FROM", "Gezinsbord <noreply@example.com>")

FRONTEND_URL = _env("FRONTEND_URL", "http://localhost:5173").rstrip("/")
CORS_ORIGINS = [
    origin.strip() for origin in _env("CORS_ORIGINS", FRONTEND_URL).split(",") if origin.strip()
]

# Predictions below this confidence are flagged for manual review. The stub
# classifier reports 0.0, so everything is flagged until a real model exists.
# Phase 3 replaces this default with a value chosen from calibration data.
LOW_CONFIDENCE_THRESHOLD = _env_number("LOW_CONFIDENCE_THRESHOLD", 0.6, float)

# --- LLM (Gemini): dormant until GEMINI_API_KEY is set ------------------------------
GEMINI_API_KEY = _env("GEMINI_API_KEY")
# Tried in order; falls through on overload/rate-limit/retired-model errors. The
# floating "-latest" alias last is the safety net against model retirement.
GEMINI_MODELS = [
    m.strip()
    for m in _env("GEMINI_MODELS", "gemini-3.5-flash,gemini-3.8-flash,gemini-flash-latest").split(
        ","
    )
    if m.strip()
]
# Hard cap per generation run (input + output tokens), to bound cost.
LLM_MAX_TOKENS_PER_RUN = _env_number("LLM_MAX_TOKENS_PER_RUN", 600_000, int)
