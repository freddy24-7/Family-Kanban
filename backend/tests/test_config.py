import pytest

from app.config import _normalise_database_url


@pytest.mark.parametrize(
    "url",
    [
        "postgres://u:p@h:5432/db",
        "postgresql://u:p@h:5432/db",
        "postgresql+psycopg://u:p@h:5432/db",
    ],
)
def test_database_url_uses_psycopg3(url):
    assert _normalise_database_url(url) == "postgresql+psycopg://u:p@h:5432/db"


def _load_config(monkeypatch, **env):
    import importlib

    import app.config

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(app.config)


def test_blank_values_fall_back_to_defaults(monkeypatch):
    config = _load_config(
        monkeypatch, INVITE_LIFETIME_DAYS="", LOW_CONFIDENCE_THRESHOLD="  ", FRONTEND_URL=""
    )
    assert config.INVITE_LIFETIME_DAYS == 7
    assert config.LOW_CONFIDENCE_THRESHOLD == 0.8
    assert config.FRONTEND_URL == "http://localhost:5173"


def test_non_numeric_value_gives_clear_error(monkeypatch):
    with pytest.raises(RuntimeError, match="ACCESS_TOKEN_LIFETIME_SECONDS must be a int"):
        _load_config(monkeypatch, ACCESS_TOKEN_LIFETIME_SECONDS="x7Hq9secret")


def test_production_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="DATABASE_URL must be set"):
        _load_config(monkeypatch, APP_ENV="production", AUTH_SECRET="s" * 40)


@pytest.fixture(autouse=True)
def _restore_config():
    """Reload config with the real test environment after each test."""
    yield
    import importlib

    import app.config

    importlib.reload(app.config)
