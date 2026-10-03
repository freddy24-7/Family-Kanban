"""Phase 9: the Planner Assistant (rules baseline and Gemini) and the load work model."""

import re
import uuid

import pytest

from app import config, llm
from app.domain import Category, Effort
from app.llm import LLMResult
from app.services import planner_assistant as pa
from sim.family import Person
from sim.world import completion_probability, suited, weekly_hours
from tests.helpers import create_household, register_and_login

# --- pure functions -------------------------------------------------------------------------


def _ctx(max_items=10) -> pa.PlanningContext:
    adult = pa.MemberContext("lid 1", uuid.uuid4(), "Mama", False)
    kid = pa.MemberContext(
        "lid 2",
        uuid.uuid4(),
        "Sem",
        True,
        planned=4,
        done=1,
        done_hours_per_week=0.5,
        by_category={"finance": [0, 2], "chores": [1, 2]},
    )
    tasks = [
        pa.TaskContext("t1", uuid.uuid4(), "belasting invullen", "finance", "L", 10, None),
        pa.TaskContext("t2", uuid.uuid4(), "afwas", "chores", "S", 3, None),
        pa.TaskContext("t3", uuid.uuid4(), "zolder opruimen", "home_maintenance", "L", 2, None),
    ]
    return pa.PlanningContext(tasks, [adult, kid], max_items)


def test_rule_plan_respects_capacity_and_history():
    ctx = _ctx()
    adult, kid = ctx.members
    assert pa.estimated_capacity(adult) == 3.0  # no history: the prior for adults
    assert pa.estimated_capacity(kid) == 0.5  # finished 1 of 4: no extra
    plan = {i.topic_id: i.user_id for i in pa.rule_plan(ctx)}
    t1, t2, t3 = ctx.tasks
    assert plan[t1.topic_id] == adult.user_id  # the kid fails finance -> never the kid
    assert plan[t2.topic_id] == kid.user_id  # adult has 0 h left after the 3 h tax return
    assert t3.topic_id not in plan  # nobody has 3 h left: better not planned than not done
    assert len(pa.rule_plan(_ctx(max_items=1))) == 1


def test_rule_plan_never_gives_a_failed_category_even_with_capacity():
    """Regression: a 0% completion rate used to count as 'no history' (`0.0 or 1.0`)."""
    ctx = _ctx()
    adult, kid = ctx.members
    adult.planned, adult.done, adult.done_hours_per_week = 4, 0, 0.0  # busy: 0 h estimated
    kid.planned, kid.done, kid.done_hours_per_week = 4, 4, 10.0  # plenty of time
    kid.by_category = {"finance": [0, 2], "chores": [2, 2]}
    plan = {i.topic_id: i.user_id for i in pa.rule_plan(ctx)}
    assert ctx.tasks[0].topic_id not in plan  # finance: the kid failed 2 of 2
    assert plan[ctx.tasks[1].topic_id] == kid.user_id


def test_llm_output_is_validated():
    ctx = _ctx(max_items=2)
    raw = [
        {"task": "t2", "member": "Lid 2", "reason": "lid 2 deed dit vorige week"},
        {"task": "t2", "member": "lid 1", "reason": "dubbel"},  # duplicate task
        {"task": "t9", "member": "lid 1", "reason": "verzonnen taak"},  # invented ref
        {"task": "t1", "member": "lid 7", "reason": "verzonnen lid"},
        "geen object",
        {"task": "t1", "member": "lid 1", "reason": "ok"},
        {"task": "t3", "member": "lid 1", "reason": "boven max_items"},
    ]
    items = pa.validate(raw, ctx)
    assert [(i.topic_id, i.user_id) for i in items] == [
        (ctx.tasks[1].topic_id, ctx.members[1].user_id),
        (ctx.tasks[0].topic_id, ctx.members[0].user_id),
    ]
    assert pa.with_names(items[0].reason, ctx) == "Sem deed dit vorige week"


def test_load_model():
    adult = Person(name="A", gender="female", role="adult")
    kid5 = Person(name="B", gender="male", role="kid", age=5)
    kid9 = Person(name="C", gender="male", role="kid", age=9)
    assert weekly_hours(adult) == 5.0 and weekly_hours(kid5) == 0.0
    assert completion_probability(Effort.S, Category.CHORES, kid5, 0) == 0.0
    fits = completion_probability(Effort.M, Category.CHORES, adult, 0)
    overloaded = completion_probability(Effort.M, Category.CHORES, adult, 4.0)
    assert overloaded == pytest.approx(fits * 0.3)
    assert not suited(Category.FINANCE, kid9) and suited(Category.CHORES, kid9)


# --- through the API -----------------------------------------------------------------------


@pytest.fixture
async def family(client, outbox):
    parent = await register_and_login(client, "mama@example.com", "Mama")
    h = await create_household(client, parent, "Familie Test")
    await client.post(
        f"/households/{h}/invites",
        json={"email": "sem@example.com", "is_child": True},
        headers=parent,
    )
    token = re.search(r"invite\?token=(\S+)", outbox[-1]["body"]).group(1)
    kid = await register_and_login(client, "sem@example.com", "Sem Austli")
    await client.post("/invites/accept", json={"token": token}, headers=kid)
    for text in ["Gymtas van Sem inpakken", "Gras maaien", "Belasting invullen"]:
        await client.post(f"/households/{h}/topics", json={"text": text}, headers=parent)
    return {"parent": parent, "kid": kid, "h": h}


async def test_rules_proposal_without_gemini(client, family, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    url = f"/households/{family['h']}/planning/proposal"
    r = await client.post(url, json={"max_items": 2}, headers=family["parent"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "rules" and body["note"] == "llm_not_configured"
    assert len(body["items"]) == 2
    assert all(i["assignee_name"] in ("Mama", "Sem Austli") and i["text"] for i in body["items"])
    # Read-only: nothing was planned.
    backlog = await client.get(f"/households/{family['h']}/backlog", headers=family["parent"])
    assert len(backlog.json()) == 3
    # Only planners may ask.
    assert (await client.post(url, json={}, headers=family["kid"])).status_code == 403


async def test_llm_proposal_never_sees_names(client, family, monkeypatch):
    seen = {}

    async def fake(system, prompt, item_schema, temperature=1.0):
        seen["prompt"] = prompt
        refs = dict(re.findall(r'"ref": "(t\d+)", "text": "([^"]+)"', prompt))
        gym = next(ref for ref, text in refs.items() if text.startswith("Gymtas"))
        items = [{"task": gym, "member": "lid 2", "reason": "lid 2 kan dit zelf"}]
        return LLMResult(items=items, model="fake", tokens_in=5, tokens_out=5)

    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")
    monkeypatch.setattr(llm, "generate_json_list", fake)
    r = await client.post(
        f"/households/{family['h']}/planning/proposal", json={}, headers=family["parent"]
    )
    body = r.json()
    assert body["method"] == "llm" and body["note"] is None
    for name in ("Mama", "Sem", "Austli"):  # display name "Sem Austli", text says "Sem"
        assert name not in seen["prompt"]
    assert "naamkind" in seen["prompt"]  # "Gymtas van Sem" -> "Gymtas van naamkind"
    [item] = body["items"]
    assert item["text"] == "Gymtas van Sem inpakken"  # the family sees the real text
    assert item["assignee_name"] == "Sem Austli" and item["reason"] == "Sem Austli kan dit zelf"


async def test_llm_failure_falls_back_to_rules(client, family, monkeypatch):
    async def broken(*args, **kwargs):
        raise llm.LLMUnavailable("all models failed")

    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")
    monkeypatch.setattr(llm, "generate_json_list", broken)
    r = await client.post(
        f"/households/{family['h']}/planning/proposal", json={}, headers=family["parent"]
    )
    assert r.json()["method"] == "rules" and r.json()["note"] == "llm_failed"
    assert r.json()["items"]
