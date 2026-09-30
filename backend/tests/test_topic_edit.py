import re

from sqlalchemy import func, select

from app.db import SessionFactory
from app.models import Prediction, Topic
from tests.helpers import create_household, register_and_login


async def _family(client, outbox):
    parent = await register_and_login(client, "p@example.com", "Mama")
    h = await create_household(client, parent)
    await client.post(
        f"/households/{h}/invites",
        json={"email": "k@example.com", "is_child": True},
        headers=parent,
    )
    token = re.search(r"invite\?token=(\S+)", outbox[-1]["body"]).group(1)
    kid = await register_and_login(client, "k@example.com", "Sem")
    await client.post("/invites/accept", json={"token": token}, headers=kid)
    return parent, kid, h


async def test_edit_text_reclassifies_and_keeps_labels(client, outbox):
    parent, kid, h = await _family(client, outbox)
    topic = (
        await client.post(f"/households/{h}/topics", json={"text": "gras maaein"}, headers=kid)
    ).json()
    await client.put(
        f"/households/{h}/topics/{topic['id']}/labels",
        json={"category": "home_maintenance", "effort": "M"},
        headers=parent,
    )

    r = await client.patch(
        f"/households/{h}/topics/{topic['id']}", json={"text": "gras maaien"}, headers=kid
    )
    assert r.status_code == 200
    assert r.json()["text"] == "gras maaien"
    assert r.json()["labels"]["category"] == "home_maintenance"  # labels survive a typo fix
    async with SessionFactory() as s:
        n = await s.scalar(
            select(func.count()).select_from(Prediction).where(Prediction.topic_id == topic["id"])
        )
        assert n == 4  # 2 original + 2 after the edit (append-only)

    # Only the due date: no re-classification; null clears it again.
    assert (
        await client.patch(
            f"/households/{h}/topics/{topic['id']}", json={"due_by": "2026-10-10"}, headers=parent
        )
    ).json()["due_by"] == "2026-10-10"
    assert (
        await client.patch(
            f"/households/{h}/topics/{topic['id']}", json={"due_by": None}, headers=parent
        )
    ).json()["due_by"] is None


async def test_only_creator_or_planner_can_change(client, outbox):
    parent, kid, h = await _family(client, outbox)
    topic = (
        await client.post(f"/households/{h}/topics", json={"text": "belasting"}, headers=parent)
    ).json()
    url = f"/households/{h}/topics/{topic['id']}"
    assert (await client.patch(url, json={"text": "x"}, headers=kid)).status_code == 403
    assert (await client.delete(url, headers=kid)).status_code == 403


async def test_soft_delete_hides_topic_but_keeps_row(client, outbox):
    parent, kid, h = await _family(client, outbox)
    topic = (
        await client.post(f"/households/{h}/topics", json={"text": "dubbel"}, headers=kid)
    ).json()
    url = f"/households/{h}/topics/{topic['id']}"
    assert (await client.delete(url, headers=kid)).status_code == 204
    assert (await client.get(url, headers=parent)).status_code == 404
    assert (await client.get(f"/households/{h}/backlog", headers=parent)).json() == []
    async with SessionFactory() as s:
        row = await s.get(Topic, topic["id"])
        assert row is not None and row.deleted_at is not None


async def test_cannot_delete_planned_topic(client, outbox):
    parent, kid, h = await _family(client, outbox)
    topic = (
        await client.post(f"/households/{h}/topics", json={"text": "afwassen"}, headers=parent)
    ).json()
    sprint = (
        await client.post(
            f"/households/{h}/sprints",
            json={"name": "W", "start_date": "2026-10-05", "end_date": "2026-10-11"},
            headers=parent,
        )
    ).json()
    await client.post(
        f"/households/{h}/sprints/{sprint['id']}/items",
        json={"topic_id": topic["id"]},
        headers=parent,
    )
    assert (
        await client.delete(f"/households/{h}/topics/{topic['id']}", headers=parent)
    ).status_code == 409
