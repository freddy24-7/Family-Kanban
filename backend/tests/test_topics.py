from datetime import UTC, datetime

from app.clock import SimulatedClock, get_clock
from app.main import app
from tests.helpers import create_household, make_superuser, register_and_login


async def test_topic_gets_stub_predictions_and_needs_review(client, outbox):
    headers = await register_and_login(client, "p@example.com")
    household = await create_household(client, headers)

    response = await client.post(
        f"/households/{household}/topics",
        json={"text": "  Boodschappen doen voor het weekend ", "due_by": "2026-10-03"},
        headers=headers,
    )
    assert response.status_code == 201
    topic = response.json()
    assert topic["text"] == "Boodschappen doen voor het weekend"
    assert topic["source"] == "real"
    assert topic["prediction"] == {
        "category": "other",
        "category_confidence": 0.0,
        "effort": "M",
        "effort_confidence": 0.0,
        "needs_review": True,  # the stub knows nothing, so everything is flagged
    }
    assert topic["labels"]["category"] is None


async def test_topic_uses_injected_clock(client, outbox):
    headers = await register_and_login(client, "q@example.com")
    household = await create_household(client, headers)
    clock = SimulatedClock(datetime(2027, 7, 1, 9, 0, tzinfo=UTC))
    app.dependency_overrides[get_clock] = lambda: clock

    topic = (
        await client.post(
            f"/households/{household}/topics", json={"text": "Gras maaien"}, headers=headers
        )
    ).json()
    assert topic["occurred_at"].startswith("2027-07-01T09:00:00")


async def test_planner_labels_record_whether_prediction_changed(client, outbox):
    headers = await register_and_login(client, "r@example.com")
    household = await create_household(client, headers)
    topic_id = (
        await client.post(
            f"/households/{household}/topics", json={"text": "Dakgoot"}, headers=headers
        )
    ).json()["id"]

    # Stub predicted other/M: category corrected, effort confirmed.
    response = await client.put(
        f"/households/{household}/topics/{topic_id}/labels",
        json={"category": "home_maintenance", "effort": "M"},
        headers=headers,
    )
    assert response.status_code == 200
    labels = response.json()["labels"]
    assert labels["category"] == "home_maintenance"
    assert labels["source"] == "planner"

    from app.db import SessionFactory
    from app.models import Topic

    async with SessionFactory() as session:
        stored = await session.get(Topic, topic_id)
        assert stored.category_prediction_changed is True
        assert stored.effort_prediction_changed is False


async def test_invalid_label_rejected(client, outbox):
    headers = await register_and_login(client, "s@example.com")
    household = await create_household(client, headers)
    topic_id = (
        await client.post(f"/households/{household}/topics", json={"text": "x"}, headers=headers)
    ).json()["id"]
    response = await client.put(
        f"/households/{household}/topics/{topic_id}/labels",
        json={"category": "gardening", "effort": "XL"},
        headers=headers,
    )
    assert response.status_code == 422


async def test_simulated_household_topics_are_marked_simulated(client, outbox):
    headers = await register_and_login(client, "admin@example.com")
    await make_superuser("admin@example.com")
    household = (
        await client.post(
            "/admin/households", json={"name": "Demo", "kind": "simulated"}, headers=headers
        )
    ).json()["id"]

    topic = (
        await client.post(
            f"/households/{household}/topics", json={"text": "Hond uitlaten"}, headers=headers
        )
    ).json()
    assert topic["source"] == "simulated"
