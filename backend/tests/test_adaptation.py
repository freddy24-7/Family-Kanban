import pytest

from ml.adaptation import adjust, segment_ratio

SMALL = {"S": 0.7, "M": 0.25, "L": 0.05}


def test_too_few_reviews_means_no_adjustment():
    assert segment_ratio([(SMALL, "M"), (SMALL, "M")]) is None


def test_reviews_matching_the_model_change_nothing():
    reviews = (
        [(SMALL, "S")] * 14 + [(SMALL, "M")] * 5 + [(SMALL, "L")]
    )  # exactly the model's mix (70/25/5)
    ratio = segment_ratio(reviews)
    assert all(r == pytest.approx(1.0) for r in ratio.values())


def test_household_whose_groceries_take_longer_flips_the_prediction():
    """The model says S (70%) but this family's last 10 groceries were mostly M."""
    reviews = [(SMALL, "M")] * 8 + [(SMALL, "S")] * 2
    ratio = segment_ratio(reviews)
    adjusted = adjust(SMALL, ratio)
    assert max(adjusted, key=adjusted.get) == "M"
    assert sum(adjusted.values()) == pytest.approx(1.0)


def test_prior_strength_damps_small_samples():
    few = segment_ratio([(SMALL, "M")] * 3)
    many = segment_ratio([(SMALL, "M")] * 12)
    assert few["M"] < many["M"]  # 3 reviews move the needle less than 12


async def test_household_adjustment_in_the_prediction_pipeline():
    """A household whose groceries keep turning out M gets M predicted, with the raw
    model output kept for lineage; another household is unaffected."""
    from datetime import UTC, date, datetime

    from sqlalchemy import select

    from app import repository
    from app.clock import SimulatedClock
    from app.db import SessionFactory
    from app.domain import Effort, Source, Task
    from app.models import ModelVersion, Prediction
    from app.services import demo, sprints, topics
    from app.tenancy import HouseholdAccess
    from ml import registry
    from ml.calibration import TemperatureScaled
    from ml.models import TextModelConfig, build_text_classifier
    from sim.family import PRESETS

    texts = ["melk halen", "brood kopen", "gras maaien", "dakgoot", "melk en kaas"] * 4
    efforts = ["S", "S", "M", "L", "S"] * 4
    cats = ["groceries", "groceries", "home_maintenance", "home_maintenance", "groceries"] * 4
    with registry.sync_session() as s:
        for task, labels in ((Task.CATEGORY, cats), (Task.EFFORT, efforts)):
            m = TemperatureScaled(
                build_text_classifier(TextModelConfig(features="word", C=10)).fit(texts, labels),
                1.0,
            )
            v = registry.save_and_register(
                s, task, m, {"preprocessing": "role-mask-v1"}, {}, 20, "h", None, "t"
            )
            registry.promote(s, v)

    clock = SimulatedClock(datetime(2026, 9, 7, 9, tzinfo=UTC))
    async with SessionFactory() as session:
        moved = await demo.create_demo_family(session, PRESETS["single-parent"], random_seed=1)
        other = await demo.create_demo_family(session, PRESETS["single-parent"], random_seed=2)
        members = await repository.list_members(session, moved.id)
        planner = next(m for m in members if m.is_planner)
        access = HouseholdAccess(moved, planner.user, True, True, True)
        sprint = await sprints.create_sprint(
            session, moved, "w", date(2026, 9, 7), date(2026, 9, 13)
        )
        for _ in range(6):  # six groceries tickets that each took M
            t = await topics.create_topic(
                session, moved, None, "melk halen", None, clock, adapt=True
            )
            await sprints.add_item(session, sprint, t.id, None)
        await sprints.start_sprint(session, sprint, clock)
        for item in await repository.list_sprint_items(session, sprint.id):
            await sprints.review_item(session, access, sprint, item, True, Effort.M, None, clock)

        fresh = await topics.create_topic(
            session, moved, None, "melk halen", None, clock, adapt=True
        )
        control = await topics.create_topic(
            session, other, None, "melk halen", None, clock, adapt=True
        )
        rows = {
            p.topic_id: p
            for p in (
                await session.scalars(select(Prediction).where(Prediction.task == Task.EFFORT))
            ).all()
        }
        assert rows[fresh.id].predicted == "M" and rows[fresh.id].adjustment["n"] == 6
        assert (
            max(rows[fresh.id].adjustment["raw"], key=rows[fresh.id].adjustment["raw"].get) == "S"
        )
        assert rows[control.id].predicted == "S" and rows[control.id].adjustment is None
        assert moved.kind == Source.SIMULATED and await session.scalar(
            select(ModelVersion).limit(1)
        )
