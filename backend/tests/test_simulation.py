"""World model rules, pool determinism/resume, and a full (small) simulated run."""

import itertools
import random
import re
from datetime import date

import numpy as np
import pytest
from sqlalchemy import func, select

from app import config, llm
from app.db import SessionFactory
from app.domain import Category, Effort
from app.llm import LLMResult
from app.models import SimTicket, Sprint, Topic
from app.services import simulation as service
from sim.family import build_world
from sim.pool import build_pool
from sim.runner import run_simulation
from sim.world import (
    SCENARIOS,
    SIM_FAMILY,
    Event,
    PlannerBehaviour,
    Scenario,
    effort_shift,
    plan_week,
    state_at,
)

START = date(2026, 7, 6)
REQ = re.compile(r"^(\d+)\. category=(\w+); effort=(\w)", re.M)


def fake_gemini():
    counter = itertools.count()

    async def generate_json_list(system, prompt, item_schema, temperature=1.0):
        items = [
            {"index": int(i), "text": f"sim taak {next(counter)} {c}", "category": c, "effort": e}
            for i, c, e in REQ.findall(prompt)
        ]
        return LLMResult(items=items, model="fake", tokens_in=10, tokens_out=5)

    return generate_json_list


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")
    monkeypatch.setattr(llm, "generate_json_list", fake_gemini())


SHIFT = Scenario(
    name="t",
    events=[
        Event(
            week=2,
            kind="effort_shift",
            category=Category.GROCERIES,
            effort={Effort.S: 0, Effort.M: 1, Effort.L: 0},
        ),
        Event(week=1, kind="add_pet", species="dog"),
        Event(week=3, kind="new_typist"),
    ],
)


def test_no_shift_means_truth_equals_text_effort():
    state = state_at(SIM_FAMILY, SCENARIOS["baseline"], START, 5)
    assert effort_shift(state, Category.GROCERIES) is None
    plans = plan_week(state, build_world(state.spec, random.Random(1)), np.random.default_rng(3))
    assert plans and all(p.true_effort == p.text_effort for p in plans)


def test_concept_drift_changes_truth_not_text():
    state = state_at(SIM_FAMILY, SHIFT, START, 2)
    world = build_world(state.spec, random.Random(1))
    plans = [p for _ in range(20) for p in plan_week(state, world, np.random.default_rng(_))]
    groceries = [p for p in plans if p.category == Category.GROCERIES]
    assert groceries and all(p.true_effort == Effort.M for p in groceries)
    assert any(p.text_effort == Effort.S for p in groceries)  # texts still written as small tasks


def test_events_change_the_family():
    before, after = (state_at(SIM_FAMILY, SHIFT, START, w) for w in (0, 3))
    assert (before.spec.dogs, after.spec.dogs) == (0, 1)
    assert SIM_FAMILY.dogs == 0  # the preset itself is not mutated
    world = build_world(after.spec, random.Random(1))
    typists = {
        p.submitter.name
        for s in range(30)
        for p in plan_week(after, world, np.random.default_rng(s))
    }
    eight_year_old = next(p.name for p in world.people if p.age == 8)
    assert eight_year_old in typists


def test_same_seed_same_week():
    state = state_at(SIM_FAMILY, SHIFT, START, 3)
    world = build_world(state.spec, random.Random(1))
    a = plan_week(state, world, np.random.default_rng(9))
    b = plan_week(state, world, np.random.default_rng(9))
    assert a == b


async def _pool(session, run) -> list[tuple]:
    rows = await session.execute(
        select(SimTicket.week, SimTicket.true_category, SimTicket.true_effort, SimTicket.submitter)
        .where(SimTicket.simulation_run_id == run.id)
        .order_by(SimTicket.week, SimTicket.text)
    )
    return rows.all()


async def test_resumed_pool_equals_uninterrupted_pool(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")
    planner = PlannerBehaviour()
    async with SessionFactory() as session:
        monkeypatch.setattr(llm, "generate_json_list", fake_gemini())
        full = await service.create_run(
            session, SIM_FAMILY, SHIFT, planner, START, 4, seed=5, family_seed=2
        )
        await build_pool(session, full, SIM_FAMILY, 2)

        broken = await service.create_run(
            session, SIM_FAMILY, SHIFT, planner, START, 4, seed=5, family_seed=2
        )
        calls = {"n": 0}
        working = fake_gemini()

        async def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 3:
                raise llm.LLMUnavailable("down")
            return await working(*a, **k)

        monkeypatch.setattr(llm, "generate_json_list", flaky)
        with pytest.raises(llm.LLMUnavailable):
            await build_pool(session, broken, SIM_FAMILY, 2)
        await build_pool(session, broken, SIM_FAMILY, 2)  # resume

        assert await _pool(session, broken) == await _pool(session, full)


async def test_full_run_and_deterministic_replay(gemini):
    planner = PlannerBehaviour(rubber_stamp_rate=0.5)
    async with SessionFactory() as session:
        run = await service.create_run(
            session, SIM_FAMILY, SHIFT, planner, START, 4, seed=7, family_seed=3
        )
        await build_pool(session, run, SIM_FAMILY, 3)
        run = await run_simulation(session, run)
        assert run.status == "completed" and len(run.weekly_stats) == 4
        week = run.weekly_stats[2]
        assert week["n"] > 0 and 0 <= week["true_category_acc"] <= 1
        assert week["events"] and "effort_shift" in week["events"][0]

        topics = await session.scalar(
            select(func.count()).select_from(Topic).where(Topic.household_id == run.household_id)
        )
        pool = await session.scalar(
            select(func.count()).select_from(SimTicket).where(SimTicket.simulation_run_id == run.id)
        )
        assert topics == pool  # every pool ticket replayed exactly once
        sprints = (
            await session.scalars(select(Sprint).where(Sprint.household_id == run.household_id))
        ).all()
        assert sprints and all(s.status == "completed" for s in sprints)

        replay = await service.create_run(
            session, SIM_FAMILY, None, planner, None, 0, seed=7, pool_run=run
        )
        assert replay.household_id != run.household_id
        replay = await run_simulation(session, replay)
        strip = lambda stats: [{k: v for k, v in s.items()} for s in stats]  # noqa: E731
        assert strip(replay.weekly_stats) == strip(run.weekly_stats)


async def test_simulation_api(gemini, client, outbox):
    from tests.helpers import make_superuser, register_and_login

    user = await register_and_login(client, "u@example.com")
    assert (await client.get("/admin/simulations", headers=user)).status_code == 403
    admin = await register_and_login(client, "root@example.com")
    await make_superuser("root@example.com")
    assert "drift-demo" in (await client.get("/admin/simulations/scenarios", headers=admin)).json()

    started = await client.post(
        "/admin/simulations",
        json={"scenario": "drift-demo", "weeks": 3, "seed": 1, "family_seed": 1},
        headers=admin,
    )
    assert started.status_code == 202
    run = (await client.get(f"/admin/simulations/{started.json()['id']}", headers=admin)).json()
    assert run["status"] == "completed" and len(run["weekly_stats"]) == 3

    replay = await client.post(
        f"/admin/simulations/{run['id']}/replay", json={"seed": 1}, headers=admin
    )
    assert replay.status_code == 202 and replay.json()["pool_run_id"] == run["id"]
    again = (await client.get(f"/admin/simulations/{replay.json()['id']}", headers=admin)).json()
    assert again["weekly_stats"] == run["weekly_stats"]
    assert len((await client.get("/admin/simulations", headers=admin)).json()) == 2
