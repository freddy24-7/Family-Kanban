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
