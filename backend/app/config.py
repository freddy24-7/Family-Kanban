"""All configuration comes from environment variables (see ../.env.example)."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Local dev: load the repo-root .env. In production the host sets real env vars.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

log = logging.getLogger(__name__)

APP_ENV = os.environ.get("APP_ENV", "development")
IS_PRODUCTION = APP_ENV == "production"


def _normalise_database_url(url: str) -> str:
    """Hosts like Railway inject postgresql:// (or postgres://), which SQLAlchemy
    maps to the psycopg2 driver. We use psycopg 3."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


DATABASE_URL = _normalise_database_url(
    os.environ.get(
        "DATABASE_URL", "postgresql+psycopg://famkanban:famkanban@localhost:5433/famkanban"
    )
)

AUTH_SECRET = os.environ.get("AUTH_SECRET", "")
if len(AUTH_SECRET) < 32:
    if IS_PRODUCTION:
        raise RuntimeError("AUTH_SECRET must be set to at least 32 characters in production")
    AUTH_SECRET = "dev-only-insecure-secret-do-not-use-in-production"
    log.warning("AUTH_SECRET missing or short; using an insecure development secret")

ACCESS_TOKEN_LIFETIME_SECONDS = int(
    os.environ.get("ACCESS_TOKEN_LIFETIME_SECONDS", str(60 * 60 * 24 * 30))
)
INVITE_LIFETIME_DAYS = int(os.environ.get("INVITE_LIFETIME_DAYS", "7"))

GOOGLE_OAUTH_CLIENT_ID = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "")
GOOGLE_OAUTH_CLIENT_SECRET = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "")

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", "Gezinsbord <noreply@example.com>")

FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173").rstrip("/")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", FRONTEND_URL).split(",")
    if origin.strip()
]

# Predictions below this confidence are flagged for manual review. The stub
# classifier reports 0.0, so everything is flagged until a real model exists.
# Phase 3 replaces this default with a value chosen from calibration data.
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", "0.6"))
