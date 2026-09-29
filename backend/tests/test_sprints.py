"""Sprint planning, board moves and review, with a parent (planner/reviewer) and a kid."""

import re

import pytest

from tests.helpers import create_household, register_and_login


@pytest.fixture
async def family(client, outbox):
    parent = await register_and_login(client, "mama@example.com", "Mama")
    household = await create_household(client, parent, "Familie Test")
    await client.post(
        f"/households/{household}/invites",
        json={"email": "kid@example.com", "is_child": True},
        headers=parent,
    )
    token = re.search(r"invite\?token=(\S+)", outbox[-1]["body"]).group(1)
    kid = await register_and_login(client, "kid@example.com", "Sem")
    await client.post("/invites/accept", json={"token": token}, headers=kid)
    members = (await client.get(f"/households/{household}", headers=parent)).json()["members"]
    ids = {m["display_name"]: m["user_id"] for m in members}
    topics = []
    for text in ["gras maaien", "gymtas inpakken", "melk halen"]:
        r = await client.post(
            f"/households/{household}/topics", json={"text": text}, headers=parent
        )
        topics.append(r.json()["id"])
    return {"parent": parent, "kid": kid, "h": household, "ids": ids, "topics": topics}


async def test_full_sprint_flow(client, family):
    f, h = family, family["h"]
    backlog = (await client.get(f"/households/{h}/backlog", headers=f["parent"])).json()
    assert len(backlog) == 3

    sprint = (
        await client.post(
            f"/households/{h}/sprints",
            json={"name": "Week 40", "start_date": "2026-09-28", "end_date": "2026-10-04"},
            headers=f["parent"],
        )
    ).json()
    assert sprint["status"] == "planned"
    s = f"/households/{h}/sprints/{sprint['id']}"

    mow = (
        await client.post(
            f"{s}/items",
            json={"topic_id": f["topics"][0], "assignee_id": f["ids"]["Mama"]},
            headers=f["parent"],
        )
    ).json()
    gym = (
        await client.post(
            f"{s}/items",
            json={"topic_id": f["topics"][1], "assignee_id": f["ids"]["Sem"]},
            headers=f["parent"],
        )
    ).json()
    assert gym["assignee_name"] == "Sem" and gym["topic"]["text"] == "gymtas inpakken"
    assert len((await client.get(f"/households/{h}/backlog", headers=f["parent"])).json()) == 1

    assert (await client.post(f"{s}/start", headers=f["parent"])).json()["status"] == "active"

    # The kid moves their own item, but not someone else's.
    moved = await client.put(
        f"{s}/items/{gym['id']}/status", json={"status": "done"}, headers=f["kid"]
    )
    assert moved.status_code == 200 and moved.json()["status"] == "done"
    denied = await client.put(
        f"{s}/items/{mow['id']}/status", json={"status": "in_progress"}, headers=f["kid"]
    )
    assert denied.status_code == 403

    # Only reviewers review; completing needs every item reviewed.
    review = {"completed": True, "effort_actual": "S"}
    assert (
        await client.put(f"{s}/items/{gym['id']}/review", json=review, headers=f["kid"])
    ).status_code == 403
    assert (
        await client.put(f"{s}/items/{gym['id']}/review", json=review, headers=f["parent"])
    ).status_code == 200
    assert (await client.post(f"{s}/complete", json={}, headers=f["parent"])).status_code == 409
    not_done = {"completed": False, "note": "Regen"}
    assert (
        await client.put(f"{s}/items/{mow['id']}/review", json=not_done, headers=f["parent"])
    ).status_code == 200

    done = (
        await client.post(f"{s}/complete", json={"notes": "Goede week"}, headers=f["parent"])
    ).json()
    assert done["status"] == "completed"
    assert done["review"] == {
        **done["review"],
        "notes": "Goede week",
        "completed_count": 1,
        "total_count": 2,
    }
    reviewed = {i["topic"]["text"]: (i["completed"], i["effort_actual"]) for i in done["items"]}
    assert reviewed == {"gymtas inpakken": (True, "S"), "gras maaien": (False, None)}

    # Not completed -> back in the backlog; completed -> gone for good.
    texts = {
        t["text"]
        for t in (await client.get(f"/households/{h}/backlog", headers=f["parent"])).json()
    }
    assert texts == {"gras maaien", "melk halen"}
    sprints = (await client.get(f"/households/{h}/sprints", headers=f["kid"])).json()
    assert [(x["name"], x["item_count"], x["done_count"]) for x in sprints] == [("Week 40", 2, 1)]


async def test_sprint_rules(client, family):
    f, h = family, family["h"]
    body = {"name": "A", "start_date": "2026-10-05", "end_date": "2026-10-11"}
    sprint = (await client.post(f"/households/{h}/sprints", json=body, headers=f["parent"])).json()
    s = f"/households/{h}/sprints/{sprint['id']}"

    assert (
        await client.post(f"/households/{h}/sprints", json=body, headers=f["parent"])
    ).status_code == 409
    backwards = {"name": "B", "start_date": "2026-10-11", "end_date": "2026-10-05"}
    assert (
        await client.post(f"/households/{h}/sprints", json=backwards, headers=f["parent"])
    ).status_code == 422
    assert (
        await client.post(f"/households/{h}/sprints", json=body, headers=f["kid"])
    ).status_code == 403

    item = {"topic_id": f["topics"][0]}
    assert (await client.post(f"{s}/items", json=item, headers=f["parent"])).status_code == 201
    assert (
        await client.post(f"{s}/items", json=item, headers=f["parent"])
    ).status_code == 409  # already planned
    stranger = {"topic_id": f["topics"][1], "assignee_id": "00000000-0000-0000-0000-000000000000"}
    assert (await client.post(f"{s}/items", json=stranger, headers=f["parent"])).status_code == 422

    # Moving is only possible once the sprint is active.
    items = (await client.get(s, headers=f["parent"])).json()["items"]
    r = await client.put(
        f"{s}/items/{items[0]['id']}/status", json={"status": "done"}, headers=f["parent"]
    )
    assert r.status_code == 409
    # Removing an item puts the topic back in the backlog.
    assert (
        await client.delete(f"{s}/items/{items[0]['id']}", headers=f["parent"])
    ).status_code == 204
    assert len((await client.get(f"/households/{h}/backlog", headers=f["parent"])).json()) == 3

    # A completed review without actual effort is incomplete.
    await client.post(f"{s}/items", json=item, headers=f["parent"])
    await client.post(f"{s}/start", headers=f["parent"])
    items = (await client.get(s, headers=f["parent"])).json()["items"]
    r = await client.put(
        f"{s}/items/{items[0]['id']}/review", json={"completed": True}, headers=f["parent"]
    )
    assert r.status_code == 422
