from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_get():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_head():
    assert client.head("/health").status_code == 200
