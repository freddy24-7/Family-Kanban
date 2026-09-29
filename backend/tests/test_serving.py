"""Role tokens, calibration, promotion gate, registry round trip and serving."""

import numpy as np
import pytest
from sqlalchemy import select

from app.db import SessionFactory
from app.domain import ModelStatus, Task
from app.models import ModelVersion, Prediction
from ml import registry
from ml.calibration import TemperatureScaled, apply_temperature, fit_temperature
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


def test_promotion_gate():
    assert promotion_decision({"h1": {"macro_f1": 0.9}}, {"h1": {"macro_f1": 0.1}})[0]
    ok, reason = promotion_decision(
        {"h1": {"macro_f1": 0.9}, "h2": {"macro_f1": 0.5}},
        {"h1": {"macro_f1": 0.1}, "h2": {"macro_f1": 0.6}},
    )
    assert not ok and "h2" in reason


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
