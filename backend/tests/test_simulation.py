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
from app.domain import Category, Effort, Task
from app.llm import LLMResult
from app.models import SimTicket, Sprint, Topic
from app.models import SimulationRun as SimulationRunModel
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


async def test_monitoring_api(gemini, client, outbox):
    from tests.helpers import make_superuser, register_and_login

    admin = await register_and_login(client, "mon@example.com")
    await make_superuser("mon@example.com")
    user = await register_and_login(client, "plain@example.com")
    assert (await client.get("/admin/monitoring/models", headers=user)).status_code == 403

    run = (
        await client.post(
            "/admin/simulations",
            json={"scenario": "drift-demo", "weeks": 6, "seed": 2, "family_seed": 2},
            headers=admin,
        )
    ).json()
    report = (
        await client.get(
            f"/admin/monitoring/simulations/{run['id']}?reference=first_weeks&window=2",
            headers=admin,
        )
    ).json()
    assert report["reference_kind"] == "first_weeks" and report["n_tickets"] > 0
    assert len(report["weekly"]) == 6
    week = report["weekly"][-1]
    assert {
        "psi_category",
        "p_category",
        "category_accuracy",
        "true_category_accuracy",
        "alarms",
    } <= set(week)
    assert report["detection"] is not None and report["detection"]["detections"]

    # Stub models carry no holdout profile: no reference, no weekly entries (not an error).
    assert (await client.get(f"/admin/monitoring/simulations/{run['id']}", headers=admin)).json()[
        "reference"
    ] is None
    assert (
        await client.get("/admin/monitoring/real?reference=first_weeks", headers=admin)
    ).status_code == 200
    names = {
        m["name"] for m in (await client.get("/admin/monitoring/models", headers=admin)).json()
    }
    assert {"category-stub-0", "effort-stub-0"} <= names


async def test_resume_and_replay_after_failures(client, outbox, monkeypatch):
    from tests.helpers import make_superuser, register_and_login

    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake")
    admin = await register_and_login(client, "fix@example.com")
    await make_superuser("fix@example.com")
    working, calls = fake_gemini(), {"n": 0}

    async def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise llm.LLMUnavailable("proxy dropped")
        return await working(*a, **k)

    monkeypatch.setattr(llm, "generate_json_list", flaky)
    run = (
        await client.post(
            "/admin/simulations",
            json={"scenario": "baseline", "weeks": 3, "seed": 4, "family_seed": 4},
            headers=admin,
        )
    ).json()
    failed = (await client.get(f"/admin/simulations/{run['id']}", headers=admin)).json()
    assert failed["status"] == "failed" and failed["current_week"] == 0
    assert (
        await client.post(f"/admin/simulations/{run['id']}/replay", json={}, headers=admin)
    ).status_code == 409

    resumed = await client.post(f"/admin/simulations/{run['id']}/resume", headers=admin)
    assert resumed.status_code == 202
    done = (await client.get(f"/admin/simulations/{run['id']}", headers=admin)).json()
    assert done["status"] == "completed" and len(done["weekly_stats"]) == 3
    assert (
        await client.post(f"/admin/simulations/{run['id']}/resume", headers=admin)
    ).status_code == 409

    # A run that failed after its weeks began: not resumable, but its pool can be replayed.
    async with SessionFactory() as session:
        from app.models import SimulationRun

        stored = await session.get(SimulationRun, uuid_of(run["id"]))
        stored.status, stored.current_week = "failed", 2
        await session.commit()
    assert (
        await client.post(f"/admin/simulations/{run['id']}/resume", headers=admin)
    ).status_code == 409
    assert (
        await client.post(f"/admin/simulations/{run['id']}/replay", json={}, headers=admin)
    ).status_code == 202


def uuid_of(value: str):
    import uuid

    return uuid.UUID(value)


async def test_shadow_replay_uses_pinned_model_versions(gemini, client, outbox):
    """A replay can pin model versions (shadow evaluation) without promoting them."""
    from app.models import ModelVersion, Prediction
    from ml import registry
    from ml.models import TextModelConfig, build_text_classifier
    from tests.helpers import make_superuser, register_and_login

    admin = await register_and_login(client, "shadow@example.com")
    await make_superuser("shadow@example.com")
    texts, cats = (
        ["gras maaien", "melk halen", "zwemles"] * 4,
        ["home_maintenance", "groceries", "kids"] * 4,
    )
    model = build_text_classifier(TextModelConfig(features="word", C=10)).fit(texts, cats)
    with registry.sync_session() as s:
        candidate = registry.save_and_register(
            s,
            Task.CATEGORY,
            model,
            {"preprocessing": "role-mask-v1"},
            {},
            12,
            "h",
            None,
            "shadow test",
        )
        name = candidate.name
    source = (
        await client.post(
            "/admin/simulations",
            json={"scenario": "baseline", "weeks": 2, "seed": 3, "family_seed": 3},
            headers=admin,
        )
    ).json()

    bad = await client.post(
        f"/admin/simulations/{source['id']}/replay",
        json={"model_versions": {"category": "nope"}},
        headers=admin,
    )
    assert bad.status_code == 422
    shadow = (
        await client.post(
            f"/admin/simulations/{source['id']}/replay",
            json={"model_versions": {"category": name}},
            headers=admin,
        )
    ).json()
    assert shadow["model_versions"] == {"category": name}

    async with SessionFactory() as session:
        run = await session.get(SimulationRunModel, uuid_of(shadow["id"]))
        used = set(
            (
                await session.scalars(
                    select(ModelVersion.name)
                    .join(Prediction, Prediction.model_version_id == ModelVersion.id)
                    .where(
                        Prediction.household_id == run.household_id,
                        Prediction.task == Task.CATEGORY,
                    )
                )
            ).all()
        )
        assert used == {name}
        # Unpinned tasks keep using the active model (regression: they used to get none).
        effort_predictions = await session.scalar(
            select(func.count())
            .select_from(Prediction)
            .where(Prediction.household_id == run.household_id, Prediction.task == Task.EFFORT)
        )
        assert effort_predictions > 0
        still_active = await session.scalar(
            select(ModelVersion.status).where(ModelVersion.name == name)
        )
        assert str(still_active) == "candidate"  # never promoted


async def test_rollback_reactivates_parent():
    from app.models import ModelVersion
    from ml import registry
    from ml.models import TextModelConfig, build_text_classifier

    model = build_text_classifier(TextModelConfig(features="word", C=10)).fit(
        ["gras maaien", "melk halen"] * 3, ["S", "M"] * 3
    )
    with registry.sync_session() as s:
        stub = registry.active_version(s, Task.EFFORT)
        v1 = registry.save_and_register(s, Task.EFFORT, model, {}, {}, 6, "h", stub, "v1")
        registry.promote(s, v1)
        retired, active = registry.rollback(s, Task.EFFORT)
        assert (retired.name, active.name) == (v1.name, stub.name)
        assert str(s.get(ModelVersion, v1.id).status) == "retired"


async def test_planner_experiment_run_is_deterministic(gemini):
    """Phase 9: the rule planner under the load work model. Common random numbers make a
    replay with the same planner give the same outcome."""
    planner = PlannerBehaviour(policy="rules", work_model="load", max_items=8)
    async with SessionFactory() as session:
        run = await service.create_run(
            session, SIM_FAMILY, SHIFT, planner, START, 4, seed=11, family_seed=3
        )
        await build_pool(session, run, SIM_FAMILY, 3)
        run = await run_simulation(session, run)
        assert run.status == "completed"
        planned = [w for w in run.weekly_stats if w["sprint_items"]]
        assert planned and all(w["sprint_items"] <= 8 for w in planned)
        assert all(w["planner"]["method"] == "rules" for w in planned)
        assert all(0 <= w["done_hours"] <= w["planned_hours"] for w in planned)

        replay = await service.create_run(
            session, SIM_FAMILY, None, planner, None, 0, seed=11, pool_run=run
        )
        replay = await run_simulation(session, replay)
        keys = ("sprint_items", "sprint_done", "done_hours", "overloaded_items")
        assert [[w[k] for k in keys] for w in replay.weekly_stats] == [
            [w[k] for k in keys] for w in run.weekly_stats
        ]


async def test_load_model_luck_does_not_depend_on_the_planner(gemini):
    """Common random numbers: two DIFFERENT planners on the same pool and seed see the
    same arrival times and make the same labelling choices (review finding: a shared
    stream used to drift apart as soon as their outcomes differed)."""
    async with SessionFactory() as session:
        base = PlannerBehaviour(policy="rules", work_model="load", max_items=8)
        run = await service.create_run(
            session, SIM_FAMILY, SHIFT, base, START, 4, seed=5, family_seed=3
        )
        await build_pool(session, run, SIM_FAMILY, 3)
        run = await run_simulation(session, run)
        other = base.model_copy(update={"policy": "sim"})
        replay = await service.create_run(
            session, SIM_FAMILY, None, other, None, 0, seed=5, pool_run=run
        )
        replay = await run_simulation(session, replay)
        assert [w["sprint_done"] for w in replay.weekly_stats] != [
            w["sprint_done"] for w in run.weekly_stats
        ]  # the planners really did differ

        async def by_ticket(household_id):
            topics = await session.scalars(select(Topic).where(Topic.household_id == household_id))
            return {t.sim_ticket_id: t for t in topics}

        a, b = await by_ticket(run.household_id), await by_ticket(replay.household_id)
        assert {k: t.occurred_at for k, t in a.items()} == {k: t.occurred_at for k, t in b.items()}
        same_day = [
            k
            for k in a
            if a[k].labeled_at and b[k].labeled_at and a[k].labeled_at == b[k].labeled_at
        ]
        assert same_day
        assert all(a[k].category_label == b[k].category_label for k in same_day)
