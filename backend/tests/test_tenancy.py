"""Household isolation: one family must never see or touch another's data."""

import pytest

from tests.helpers import create_household, register_and_login


@pytest.fixture
async def two_families(client, outbox):
    a = await register_and_login(client, "a@example.com", "A")
    b = await register_and_login(client, "b@example.com", "B")
    household_a = await create_household(client, a, "Familie A")
    household_b = await create_household(client, b, "Familie B")
    topic = await client.post(
        f"/households/{household_a}/topics", json={"text": "Vuilnis buiten zetten"}, headers=a
    )
    return a, b, household_a, household_b, topic.json()["id"]


async def test_creator_is_planner_and_reviewer(client, two_families):
    a, _, household_a, _, _ = two_families
    detail = (await client.get(f"/households/{household_a}", headers=a)).json()
    assert [(m["display_name"], m["is_planner"], m["is_reviewer"]) for m in detail["members"]] == [
        ("A", True, True)
    ]


async def test_list_only_own_households(client, two_families):
    a, _, household_a, _, _ = two_families
    assert [h["id"] for h in (await client.get("/households", headers=a)).json()] == [household_a]


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/households/{h}", None),
        ("GET", "/households/{h}/topics", None),
        ("POST", "/households/{h}/topics", {"text": "inbreken"}),
        ("GET", "/households/{h}/topics/{t}", None),
        ("PUT", "/households/{h}/topics/{t}/labels", {"category": "chores", "effort": "S"}),
        ("POST", "/households/{h}/invites", {"email": "x@example.com"}),
        ("GET", "/households/{h}/invites", None),
        ("GET", "/households/{h}/backlog", None),
        ("GET", "/households/{h}/sprints", None),
        (
            "POST",
            "/households/{h}/sprints",
            {"name": "x", "start_date": "2026-01-01", "end_date": "2026-01-02"},
        ),
        ("GET", "/households/{h}/sprints/{t}", None),
        ("POST", "/households/{h}/sprints/{t}/start", None),
        ("POST", "/households/{h}/sprints/{t}/complete", {}),
        ("POST", "/households/{h}/sprints/{t}/items", {"topic_id": "{t}"}),
        ("PUT", "/households/{h}/sprints/{t}/items/{t}/status", {"status": "done"}),
        ("PUT", "/households/{h}/sprints/{t}/items/{t}/review", {"completed": False}),
        ("PATCH", "/households/{h}/sprints/{t}/items/{t}", {"assignee_id": None}),
        ("DELETE", "/households/{h}/sprints/{t}/items/{t}", None),
    ],
)
async def test_other_family_gets_404(client, two_families, method, path, body):
    _, b, household_a, _, topic_a = two_families
    url = path.format(h=household_a, t=topic_a)
    response = await client.request(method, url, json=body, headers=b)
    assert response.status_code == 404


async def test_topic_not_reachable_through_own_household_path(client, two_families):
    """Knowing another family's topic id must not help: lookups are household-scoped."""
    _, b, _, household_b, topic_a = two_families
    response = await client.get(f"/households/{household_b}/topics/{topic_a}", headers=b)
    assert response.status_code == 404


async def test_unauthenticated_rejected(client, two_families):
    _, _, household_a, _, _ = two_families
    assert (await client.get(f"/households/{household_a}/topics")).status_code == 401
