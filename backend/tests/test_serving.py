"""Role tokens, calibration, promotion gate, registry round trip and serving."""

import numpy as np
import pytest
from sqlalchemy import select

from app.db import SessionFactory
from app.domain import ModelStatus, Task
from app.models import ModelVersion, Prediction
from ml import registry
from ml.calibration import TemperatureScaled, apply_temperature, fit_temperature
from ml.evaluate import macro_f1, paired_bootstrap_delta
from ml.models import TextModelConfig, build_text_classifier
from ml.text import mask_member_names
from ml.train import promotion_decision
from tests.helpers import create_household, register_and_login

TEXTS = [
    "gras maaien",
    "dakgoot schoonmaken",
    "melk halen",
    "brood en kaas",
    "naamkind naar zwemles",
    "gymtas naamkind",
] * 4
CATS = ["home_maintenance", "home_maintenance", "groceries", "groceries", "kids", "kids"] * 4


def test_role_tokens():
    members = [("Lieke", True), ("Jeroen", False), ("Anne", False), ("Anne-Marie", True)]
    assert (
        mask_member_names("Lieke naar zwemles, JEROEN rijdt", members)
        == "naamkind naar zwemles, naamouder rijdt"
    )
    assert (
        mask_member_names("Liekes gymtas en Lieke's jas", members)
        == "naamkind gymtas en naamkind jas"
    )
    assert mask_member_names("Anne-Marie en Anne", members) == "naamkind en naamouder"
    assert mask_member_names("Liekerd", members) == "Liekerd"  # not a name match


def test_temperature_scaling_direction():
    y = np.array(["a", "b"] * 50)
    right = np.where(y == "a", 0, 1)
    # Always right but only 60% confident -> underconfident -> sharpen (T < 1).
    under = np.full((100, 2), 0.4)
    under[np.arange(100), right] = 0.6
    assert fit_temperature(under, y, ["a", "b"]) < 1
    # Always 99% sure of "a", but only half the labels are "a" -> overconfident -> soften (T > 1).
    over = np.tile([0.99, 0.01], (100, 1))
    assert fit_temperature(over, y, ["a", "b"]) > 1
    p = apply_temperature(under, 0.5)
    assert np.allclose(p.sum(axis=1), 1) and p.max() > 0.6


def _holdout(delta, low, high, missing=0, real=None):
    h = {"report": {"missing": missing}, "delta_vs_champion": {"delta": delta, "ci95": (low, high)}}
    if real is not None:
        h["delta_vs_champion_real"] = {"delta": real}
    return h


def test_promotion_gate():
    ok = lambda d, lo, hi, **kw: _holdout(d, lo, hi, **kw)  # noqa: E731
    assert promotion_decision({"h1": ok(0.5, 0.4, 0.6)}, "h1")[0]
    passed, reason = promotion_decision({"h1": ok(0.02, -0.01, 0.05)}, "h1")  # within noise
    assert not passed and "not clearly above zero" in reason
    passed, reason = promotion_decision({"h1": ok(0.5, 0.4, 0.6, missing=3)}, "h1")
    assert not passed and "missing" in reason
    passed, reason = promotion_decision({"h1": ok(0.5, 0.4, 0.6, real=-0.02)}, "h1")
    assert not passed and "real data" in reason


def test_gate_primary_and_secondary_holdouts():
    """New world (primary) clearly better; old world a bit worse but within noise: pass.
    Old world clearly worse: fail (needs a deliberate decision)."""
    new, old_noise, old_worse = (
        _holdout(0.3, 0.2, 0.4),
        _holdout(-0.02, -0.06, 0.02),
        _holdout(-0.1, -0.15, -0.05),
    )
    assert promotion_decision({"new": new, "old": old_noise}, "new")[0]
    passed, reason = promotion_decision({"new": new, "old": old_worse}, "new")
    assert not passed and "old: clearly worse" in reason
    # A small gain on a secondary holdout doesn't need to be significant.
    assert promotion_decision({"new": new, "old": _holdout(0.01, -0.03, 0.05)}, "new")[0]
    assert not promotion_decision({"new": new}, "missing-name")[0]


def test_paired_bootstrap_detects_improvement():
    y = np.array(["a", "b"] * 50)
    better, worse = y.copy(), y.copy()
    worse[:20] = np.where(worse[:20] == "a", "b", "a")
    delta, low, high = paired_bootstrap_delta(y, better, worse, macro_f1)
    assert delta > 0 and low > 0
    assert paired_bootstrap_delta(y, y, y, macro_f1) == (0.0, 0.0, 0.0)


def _register_and_promote(estimator, preprocessing="role-mask-v1", sklearn_version=None) -> str:
    with registry.sync_session() as session:
        meta = {"preprocessing": preprocessing}
        version = registry.save_and_register(
            session,
            Task.CATEGORY,
            estimator,
            meta,
            {"primary_metric": "macro_f1"},
            24,
            "hash",
            None,
            "test",
        )
        if sklearn_version:  # simulate an artifact pickled with another library version
            import uuid

            from app.models import ModelArtifact

            artifact = session.get(
                ModelArtifact, uuid.UUID(version.artifact_uri.removeprefix("db://"))
            )
            artifact.meta = {**artifact.meta, "sklearn_version": sklearn_version}
            session.commit()
        registry.promote(session, version)
        return version.name


async def test_trained_model_serves_predictions_with_role_tokens(client, outbox):
    model = TemperatureScaled(
        build_text_classifier(TextModelConfig(features="word", C=10)).fit(TEXTS, CATS), 0.7
    )
    name = _register_and_promote(model)
    assert name == "category-v1"

    headers = await register_and_login(client, "ouder@example.com", "Jeroen")
    household = await create_household(client, headers)
    # The parent's own name is masked as "naamouder"; an unknown kid name stays text.
    topic = (
        await client.post(
            f"/households/{household}/topics", json={"text": "melk halen"}, headers=headers
        )
    ).json()
    assert topic["prediction"]["category"] == "groceries"
    assert 0.5 < topic["prediction"]["category_confidence"] <= 1

    async with SessionFactory() as session:
        versions = {v.name: v.status for v in (await session.scalars(select(ModelVersion))).all()}
        assert versions["category-v1"] == ModelStatus.ACTIVE
        assert versions["category-stub-0"] == ModelStatus.RETIRED
        pred = await session.scalar(select(Prediction).where(Prediction.task == Task.CATEGORY))
        assert set(pred.probabilities) == {"groceries", "home_maintenance", "kids"}


async def test_unloadable_model_fails_soft(client, outbox):
    model = build_text_classifier(TextModelConfig(features="word", C=10)).fit(TEXTS, CATS)
    _register_and_promote(model, sklearn_version="0.1.0")
    headers = await register_and_login(client, "x@example.com")
    household = await create_household(client, headers)
    response = await client.post(
        f"/households/{household}/topics", json={"text": "melk halen"}, headers=headers
    )
    assert response.status_code == 201  # the ticket is still created
    prediction = response.json()["prediction"]
    assert prediction["category"] is None and prediction["needs_review"] is True
    assert prediction["effort"] == "M"  # the effort stub still works


def test_version_mismatch_is_refused():
    data, digest = registry.serialize(
        build_text_classifier(TextModelConfig(features="word")).fit(TEXTS, CATS)
    )
    with pytest.raises(registry.ModelLoadError, match="scikit-learn"):
        registry._deserialize(data, digest, {"sklearn_version": "0.1.0"})
    with pytest.raises(registry.ModelLoadError, match="hash"):
        registry._deserialize(data, "0" * 64, {})


def test_temperature_respects_non_alphabetical_class_order():
    """Effort columns are S, M, L (not alphabetical). A searchsorted lookup would
    point at the wrong column or crash."""
    y = np.array(["S", "M", "L"] * 30)
    proba = np.tile([0.2, 0.2, 0.2], (90, 1))
    proba[np.arange(90), np.arange(90) % 3] = 0.6  # right answer always gets 0.6
    assert fit_temperature(proba, y, ["S", "M", "L"]) < 1  # underconfident -> sharpen
