import re
from datetime import UTC, datetime, timedelta

from app.clock import SimulatedClock, get_clock
from app.main import app
from tests.helpers import create_household, register_and_login


def invite_token(outbox) -> str:
    invite_mail = next(m for m in outbox if m["subject"].startswith("Uitnodiging"))
    return re.search(r"invite\?token=(\S+)", invite_mail["body"]).group(1)


async def invite(client, headers, household, email="kid@example.com", **roles):
    response = await client.post(
        f"/households/{household}/invites", json={"email": email, **roles}, headers=headers
    )
    assert response.status_code == 201, response.text
    return response.json()


async def test_invite_accept_flow(client, outbox):
    parent = await register_and_login(client, "parent@example.com", "Mama")
    household = await create_household(client, parent, "Familie Jansen")
    await invite(client, parent, household, "Kid@Example.com")
    token = invite_token(outbox)
    assert "Mama" in outbox[-1]["body"] and "Familie Jansen" in outbox[-1]["body"]

    kid = await register_and_login(client, "kid@example.com", "Sem")
    response = await client.post("/invites/accept", json={"token": token}, headers=kid)
    assert response.status_code == 200
    assert response.json()["id"] == household

    members = (await client.get(f"/households/{household}", headers=kid)).json()["members"]
    sem = next(m for m in members if m["display_name"] == "Sem")
    assert (sem["is_planner"], sem["is_reviewer"]) == (False, False)

    # Token is single use, and the pending list is now empty.
    assert (
        await client.post("/invites/accept", json={"token": token}, headers=kid)
    ).status_code == 409
    assert (await client.get(f"/households/{household}/invites", headers=parent)).json() == []


async def test_invite_for_other_email_rejected(client, outbox):
    parent = await register_and_login(client, "p2@example.com")
    household = await create_household(client, parent)
    await invite(client, parent, household, "intended@example.com")
    stranger = await register_and_login(client, "stranger@example.com")
    response = await client.post(
        "/invites/accept", json={"token": invite_token(outbox)}, headers=stranger
    )
    assert response.status_code == 403


async def test_expired_invite_rejected(client, outbox):
    parent = await register_and_login(client, "p3@example.com")
    household = await create_household(client, parent)
    await invite(client, parent, household, "late@example.com")
    late = await register_and_login(client, "late@example.com")

    app.dependency_overrides[get_clock] = lambda: SimulatedClock(
        datetime.now(UTC) + timedelta(days=8)
    )
    response = await client.post(
        "/invites/accept", json={"token": invite_token(outbox)}, headers=late
    )
    assert response.status_code == 410


async def test_non_planner_cannot_invite_or_label(client, outbox):
    parent = await register_and_login(client, "p4@example.com")
    household = await create_household(client, parent)
    await invite(client, parent, household, "member@example.com")
    member = await register_and_login(client, "member@example.com")
    await client.post("/invites/accept", json={"token": invite_token(outbox)}, headers=member)

    response = await client.post(
        f"/households/{household}/invites", json={"email": "x@example.com"}, headers=member
    )
    assert response.status_code == 403

    topic_id = (
        await client.post(
            f"/households/{household}/topics", json={"text": "Afwassen"}, headers=member
        )
    ).json()["id"]
    response = await client.put(
        f"/households/{household}/topics/{topic_id}/labels",
        json={"category": "chores", "effort": "S"},
        headers=member,
    )
    assert response.status_code == 403
