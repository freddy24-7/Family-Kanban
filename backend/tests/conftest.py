"""Test setup: a separate Postgres database built with the real Alembic
migrations (so tests also prove the migrations match the models)."""

import os
import subprocess
from pathlib import Path

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://famkanban:famkanban@localhost:5433/famkanban_test",
)
# Must happen before any app module is imported (app.db reads it at import).
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["RESEND_API_KEY"] = ""

import psycopg  # noqa: E402
import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app import (  # noqa: E402
    mailer,
    models,  # noqa: E402,F401  (registers all tables on Base.metadata)
    repository,
)
from app.db import Base, SessionFactory, engine  # noqa: E402
from app.main import app  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _recreate_test_database() -> None:
    admin_url = TEST_DATABASE_URL.replace("postgresql+psycopg", "postgresql").rsplit("/", 1)
    db_name = admin_url[1]
    with psycopg.connect(f"{admin_url[0]}/postgres", autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{db_name}"')
    subprocess.run(
        ["uv", "run", "alembic", "upgrade", "head"],
        cwd=BACKEND_DIR,
        env={**os.environ, "DATABASE_URL": TEST_DATABASE_URL},
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="session", autouse=True)
async def database():
    _recreate_test_database()
    yield
    await engine.dispose()


@pytest.fixture(autouse=True)
async def clean_tables(database):
    async with SessionFactory() as session:
        # Every table from the models, so new tables are cleaned automatically.
        tables = ", ".join(f'"{t.name}"' for t in Base.metadata.sorted_tables)
        await session.execute(text(f"TRUNCATE {tables} CASCADE"))
        await session.commit()
        await repository.ensure_stub_model_versions(session)
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def outbox(monkeypatch):
    """Captures emails instead of sending them."""
    sent: list[dict[str, str]] = []

    async def fake_send(to: str, subject: str, body_text: str) -> bool:
        sent.append({"to": to, "subject": subject, "body": body_text})
        return True

    monkeypatch.setattr(mailer, "send_email", fake_send)
    return sent
