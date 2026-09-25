import re

from tests.helpers import PASSWORD, register_and_login


async def test_health(client):
    assert (await client.get("/health")).json() == {"status": "ok"}
    assert (await client.head("/health")).status_code == 200


async def test_register_login_me(client, outbox):
    headers = await register_and_login(client, "anna@example.com", "Anna")
    me = (await client.get("/users/me", headers=headers)).json()
    assert me["email"] == "anna@example.com"
    assert me["display_name"] == "Anna"
    assert me["is_superuser"] is False
    assert me["is_verified"] is False


async def test_registration_sends_dutch_verification_email_that_verifies(client, outbox):
    headers = await register_and_login(client, "bram@example.com")
    assert len(outbox) == 1
    assert outbox[0]["subject"] == "Bevestig je e-mailadres"
    token = re.search(r"verify-email\?token=(\S+)", outbox[0]["body"]).group(1)

    assert (await client.post("/auth/verify", json={"token": token})).status_code == 200
    assert (await client.get("/users/me", headers=headers)).json()["is_verified"] is True


async def test_short_password_rejected(client):
    response = await client.post(
        "/auth/register", json={"email": "c@example.com", "password": "short", "display_name": "C"}
    )
    assert response.status_code == 400


async def test_register_cannot_self_promote_to_admin(client, outbox):
    response = await client.post(
        "/auth/register",
        json={
            "email": "sneaky@example.com",
            "password": PASSWORD,
            "display_name": "S",
            "is_superuser": True,
        },
    )
    assert response.status_code == 201
    assert response.json()["is_superuser"] is False


async def test_logout_revokes_token(client, outbox):
    headers = await register_and_login(client, "dirk@example.com")
    assert (await client.post("/auth/logout", headers=headers)).status_code == 204
    assert (await client.get("/users/me", headers=headers)).status_code == 401
