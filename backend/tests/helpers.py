"""Shared test helpers."""

from httpx import AsyncClient
from sqlalchemy import text

from app.db import SessionFactory

PASSWORD = "correct-horse-battery"


async def register_and_login(client: AsyncClient, email: str, name: str = "Test") -> dict:
    """Returns auth headers for a freshly registered user."""
    response = await client.post(
        "/auth/register", json={"email": email, "password": PASSWORD, "display_name": name}
    )
    assert response.status_code == 201, response.text
    response = await client.post("/auth/login", data={"username": email, "password": PASSWORD})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


async def make_superuser(email: str) -> None:
    async with SessionFactory() as session:
        await session.execute(
            text('UPDATE "user" SET is_superuser = true WHERE email = :e'), {"e": email}
        )
        await session.commit()


async def create_household(client: AsyncClient, headers: dict, name: str = "Familie") -> str:
    response = await client.post("/households", json={"name": name}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()["id"]
