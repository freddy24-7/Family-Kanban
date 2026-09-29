"""Demo families and ticket generation, with a fake Gemini that misbehaves on purpose."""

import random
import re

import pytest
from sqlalchemy import func, select

from app import config, llm
from app.clock import SystemClock
from app.db import SessionFactory
from app.domain import Category, LabelSource, Source
from app.llm import LLMResult
from app.models import Membership, Prediction, Topic, User
from app.services import demo
from ml.text import normalise_text
from sim.family import PRESETS, build_world, category_weights
from sim.generator import plan_batch
from tests.helpers import PASSWORD, make_superuser, register_and_login

REQUEST_LINE = re.compile(r"^(\d+)\. category=(\w+);", re.M)


def fake_gemini(misbehave: bool = True):
    """Answers each request in the prompt. With misbehave=True the first batch also
    contains: one invalid effort, one exact duplicate, one skipped request and one
    item labelled with a different category than requested."""
    calls = {"n": 0}

    async def generate_json_list(system, prompt, item_schema, temperature=1.0):
        calls["n"] += 1
        items = []
        for index, category in REQUEST_LINE.findall(prompt):
            items.append(
                {
                    "index": int(index),
                    "text": f"taak {index} voor {category}",
                    "category": category,
                    "effort": "M",
                }
            )
        if misbehave and calls["n"] == 1 and len(items) >= 4:
            items[0]["effort"] = "XL"  # invalid label value
            items[1]["text"] = items[2]["text"].upper() + "!"  # duplicate after normalisation
            items.pop(3)  # silently skipped request
            items[-1]["category"] = "other" if items[-1]["category"] != "other" else "chores"
        return LLMResult(items=items, model="fake-flash", tokens_in=100, tokens_out=50)

    return generate_json_list


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(llm, "generate_json_list", fake_gemini())


async def test_couple_without_kids_gets_no_kids_tickets():
    assert category_weights(PRESETS["couple-no-kids"])[Category.KIDS] == 0
    world = build_world(PRESETS["couple-no-kids"], random.Random(1))
    assert {r.category for r in plan_batch(world, 200, random.Random(2))}.isdisjoint(
        {Category.KIDS}
    )


async def test_young_kids_do_not_submit_tickets():
    world = build_world(PRESETS["typical-1"], random.Random(1))  # kids aged 3, 8, 14
    ages = {p.age for p in world.submitters if p.role == "kid"}
    assert ages == {14}


async def test_same_seed_same_family_names():
    a = build_world(PRESETS["typical-2"], random.Random(42))
    b = build_world(PRESETS["typical-2"], random.Random(42))
    assert a == b


def test_normalise_text():
    assert normalise_text("  Gras MAAIEN!! ") == normalise_text("gras maaien")
    assert normalise_text("Café") == "cafe"


async def test_create_demo_family_members():
    async with SessionFactory() as session:
        household = await demo.create_demo_family(session, PRESETS["typical-1"], random_seed=7)
        assert household.kind == Source.SIMULATED
        members = (
            (
                await session.scalars(
                    select(Membership).where(Membership.household_id == household.id)
                )
            )
            .unique()
            .all()
        )
        assert len(members) == 5
        assert all(m.user.is_simulated and not m.user.is_active for m in members)
        assert sorted((m.profile["role"], m.is_planner) for m in members) == [
            ("adult", True),
            ("adult", True),
            ("kid", False),
            ("kid", False),
            ("kid", False),
        ]


async def test_generation_validates_and_records_lineage(gemini):
    async with SessionFactory() as session:
        household = await demo.create_demo_family(session, PRESETS["typical-2"], random_seed=1)
        run = await demo.generate_tickets(session, household, 30, SystemClock(), random_seed=3)

        assert run.status == "completed"
        assert run.produced == 30
        assert run.rejected_invalid == 2  # the bad effort + the skipped request
        assert run.rejected_duplicate == 1
        assert run.category_mismatches == 1
        assert run.models_used == {"fake-flash": 2}  # a second batch filled the gap

        topics = (
            await session.scalars(select(Topic).where(Topic.household_id == household.id))
        ).all()
        assert len(topics) == 30
        assert all(t.source == Source.SIMULATED for t in topics)
        assert all(t.label_source == LabelSource.GENERATOR and t.category_label for t in topics)
        assert all(t.generation_run_id == run.id for t in topics)
        assert len({normalise_text(t.text) for t in topics}) == 30
        predictions = await session.scalar(
            select(func.count())
            .select_from(Prediction)
            .where(Prediction.household_id == household.id)
        )
        assert predictions == 60  # category + effort per topic


async def test_llm_outage_keeps_partial_work(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key")
    working = fake_gemini(misbehave=False)
    calls = {"n": 0}

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise llm.LLMUnavailable("All Gemini models failed: gemini-x:503")
        return await working(*args, **kwargs)

    monkeypatch.setattr(llm, "generate_json_list", flaky)
    async with SessionFactory() as session:
        household = await demo.create_demo_family(session, PRESETS["single-parent"], random_seed=1)
        run = await demo.generate_tickets(session, household, 60, SystemClock())
        assert run.status == "failed"
        assert "503" in run.error
        assert run.produced == 25  # the first batch was kept


async def test_generation_refuses_real_household(gemini, client, outbox):
    headers = await register_and_login(client, "real@example.com")
    household_id = (
        await client.post("/households", json={"name": "Echt"}, headers=headers)
    ).json()["id"]
    async with SessionFactory() as session:
        from app import repository

        household = await repository.get_household(session, household_id)
        with pytest.raises(ValueError):
            await demo.generate_tickets(session, household, 5, SystemClock())


async def test_demo_api_end_to_end(gemini, client, outbox):
    admin = await register_and_login(client, "admin@example.com")
    await make_superuser("admin@example.com")

    presets = (await client.get("/admin/demo/presets", headers=admin)).json()
    assert "typical-1" in presets and len(presets) == 12

    family = await client.post(
        "/admin/demo/families",
        json={"preset": "big-family", "training_eligible": True},
        headers=admin,
    )
    assert family.status_code == 201
    household_id = family.json()["id"]
    assert family.json()["kind"] == "simulated"

    started = await client.post(
        f"/admin/demo/families/{household_id}/tickets", json={"count": 10}, headers=admin
    )
    assert started.status_code == 202
    run = (
        await client.get(f"/admin/demo/generation-runs/{started.json()['id']}", headers=admin)
    ).json()
    assert run["status"] == "completed" and run["produced"] == 10

    topics = (await client.get(f"/households/{household_id}/topics", headers=admin)).json()
    assert len(topics) == 10 and topics[0]["labels"]["source"] == "generator"


async def test_demo_api_requires_superuser_and_key(client, outbox, monkeypatch):
    user = await register_and_login(client, "u@example.com")
    assert (await client.get("/admin/demo/presets", headers=user)).status_code == 403

    admin = await register_and_login(client, "a@example.com")
    await make_superuser("a@example.com")
    household_id = (
        await client.post("/admin/demo/families", json={"preset": "typical-1"}, headers=admin)
    ).json()["id"]
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    response = await client.post(
        f"/admin/demo/families/{household_id}/tickets", json={"count": 5}, headers=admin
    )
    assert response.status_code == 503


async def test_custom_spec_validation(client, outbox):
    admin = await register_and_login(client, "b@example.com")
    await make_superuser("b@example.com")
    bad = {"spec": {"adults": [{"gender": "male"}], "housing": "apartment", "garden": True}}
    assert (await client.post("/admin/demo/families", json=bad, headers=admin)).status_code == 422
    both = {"preset": "typical-1", "spec": {"adults": [{"gender": "male"}]}}
    assert (await client.post("/admin/demo/families", json=both, headers=admin)).status_code == 422


async def test_simulated_members_cannot_log_in(client):
    async with SessionFactory() as session:
        household = await demo.create_demo_family(session, PRESETS["young-family"], random_seed=1)
        email = (
            await session.scalars(
                select(User.email).join(Membership).where(Membership.household_id == household.id)
            )
        ).first()
    response = await client.post("/auth/login", data={"username": email, "password": PASSWORD})
    assert response.status_code == 400


async def test_network_error_cascades_to_next_model(monkeypatch):
    """A dropped connection on the first model must fall through to the next."""
    import httpx
    from google.genai import types as genai_types

    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(config, "GEMINI_MODELS", ["model-a", "model-b"])
    monkeypatch.setattr(llm, "RETRY_BASE_SECONDS", 0)
    tried = []

    class FakeModels:
        async def generate_content(self, model, contents, config):
            tried.append(model)
            if model == "model-a":
                raise httpx.RemoteProtocolError("Server disconnected")
            return genai_types.GenerateContentResponse(
                candidates=[
                    genai_types.Candidate(
                        content=genai_types.Content(
                            parts=[genai_types.Part(text='{"items": [1, 2]}')]
                        )
                    )
                ]
            )

    class FakeClient:
        class aio:
            models = FakeModels()

    monkeypatch.setattr(llm, "_get_client", lambda: FakeClient)
    result = await llm.generate_json_list("sys", "prompt", {"type": "INTEGER"})
    assert tried == ["model-a", "model-a", "model-a", "model-b"]  # 3 attempts, then cascade
    assert result.model == "model-b" and result.items == [1, 2]
    assert result.failed_models == ["model-a:RemoteProtocolError"]


async def test_unexpected_crash_marks_run_failed(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key")

    async def boom(*args, **kwargs):
        raise KeyError("surprise")

    monkeypatch.setattr(llm, "generate_json_list", boom)
    async with SessionFactory() as session:
        household = await demo.create_demo_family(session, PRESETS["single-parent"], random_seed=1)
        run = await demo.start_generation(session, household, 10)
        with pytest.raises(KeyError):
            await demo.continue_generation(session, household, run, SystemClock())
        assert run.status == "failed" and "KeyError" in run.error


async def test_unsupported_thinking_level_falls_back_to_low(monkeypatch):
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types

    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(config, "GEMINI_MODELS", ["picky-model"])
    monkeypatch.setattr(llm, "_thinking_level_for", {})
    levels = []

    class FakeModels:
        async def generate_content(self, model, contents, config):
            levels.append(str(config.thinking_config.thinking_level.value).lower())
            if levels[-1] == "minimal":
                raise genai_errors.ClientError(
                    400,
                    {
                        "error": {
                            "message": "Thinking level MINIMAL is not supported for this model."
                        }
                    },
                )
            return genai_types.GenerateContentResponse(
                candidates=[
                    genai_types.Candidate(
                        content=genai_types.Content(parts=[genai_types.Part(text='{"items": []}')])
                    )
                ]
            )

    class FakeClient:
        class aio:
            models = FakeModels()

    monkeypatch.setattr(llm, "_get_client", lambda: FakeClient)
    await llm.generate_json_list("sys", "p", {"type": "INTEGER"})
    await llm.generate_json_list("sys", "p", {"type": "INTEGER"})
    assert levels == ["minimal", "low", "low"]  # remembered after the first rejection


async def test_transient_503_is_retried_on_same_model(monkeypatch):
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types

    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(config, "GEMINI_MODELS", ["model-a", "model-b"])
    monkeypatch.setattr(llm, "RETRY_BASE_SECONDS", 0)
    tried = []

    class FakeModels:
        async def generate_content(self, model, contents, config):
            tried.append(model)
            if len(tried) == 1:
                raise genai_errors.ServerError(503, {"error": {"message": "high demand"}})
            return genai_types.GenerateContentResponse(
                candidates=[
                    genai_types.Candidate(
                        content=genai_types.Content(parts=[genai_types.Part(text='{"items": []}')])
                    )
                ]
            )

    class FakeClient:
        class aio:
            models = FakeModels()

    monkeypatch.setattr(llm, "_get_client", lambda: FakeClient)
    result = await llm.generate_json_list("sys", "p", {"type": "INTEGER"})
    assert tried == ["model-a", "model-a"] and result.model == "model-a"
