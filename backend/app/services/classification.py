"""Runs the active models on topics and logs append-only Prediction rows."""

import logging
import uuid
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app import config, repository
from app.clock import Clock
from app.domain import Task
from app.models import ModelVersion, Prediction, Topic
from ml import adaptation, registry
from ml.text import ROLE_MASK_VERSION, mask_member_names

log = logging.getLogger(__name__)


async def classify_topics(
    session: AsyncSession,
    topics: Sequence[Topic],
    clock: Clock,
    versions: dict[Task, ModelVersion] | None = None,
    adapt: bool | None = None,
) -> list[Prediction]:
    """Predict every task for the given topics with the active model versions. Each
    prediction records which model version produced it, so its accuracy can later be
    attributed to the right model. Caller commits.

    - `versions` pins models per task (shadow evaluation); unpinned tasks keep the
      active model. (A first version replaced ALL active models with the pinned ones.)
    - `adapt` (default: config.EFFORT_ADAPTATION) applies the per-household effort
      adjustment (ml/adaptation.py); the raw model output is kept in `adjustment`.
    - Fails soft per task: if a model can't be loaded or can't predict, the topic is
      still created (it just has no prediction for that task and is flagged for review)."""
    if not topics:
        return []
    adapt = config.EFFORT_ADAPTATION if adapt is None else adapt
    members: dict[uuid.UUID, list[tuple[str, bool]]] = {}
    active = {**(await repository.active_model_versions(session)), **(versions or {})}
    predictions: list[Prediction] = []
    category_of: dict[uuid.UUID, str] = {}
    ratios: dict[uuid.UUID, dict[str, tuple[dict[str, float], int]]] = {}
    # Category first: the effort adjustment needs each topic's category.
    for task in sorted(active, key=lambda t: t != Task.CATEGORY):
        version = active[task]
        try:
            model = await registry.load_model(session, version.artifact_uri)
        except Exception:
            log.exception("Could not load model %s; skipping %s predictions", version.name, task)
            continue
        texts = [topic.text for topic in topics]
        if model.preprocessing == ROLE_MASK_VERSION:
            texts = [
                mask_member_names(topic.text, await _members(session, topic.household_id, members))
                for topic in topics
            ]
        elif model.preprocessing is not None:
            log.error("Model %s needs unknown preprocessing %r", version.name, model.preprocessing)
            continue
        try:
            # A model can load and still fail to predict (e.g. a missing dependency).
            outputs = model.predict(texts)
        except Exception:
            log.exception("Model %s failed to predict; skipping %s predictions", version.name, task)
            continue
        for topic, output in zip(topics, outputs, strict=True):
            predicted, confidence, probabilities = (
                output.predicted,
                output.confidence,
                output.probabilities,
            )
            adjustment = None
            if task == Task.CATEGORY:
                category_of[topic.id] = output.predicted
            elif task == Task.EFFORT and adapt and topic.id in category_of:
                category = category_of[topic.id]
                segment = await _segment_ratio(
                    session, topic.household_id, version, category, ratios
                )
                if segment is not None:
                    ratio, n = segment
                    adjusted = adaptation.adjust(output.probabilities, ratio)
                    probabilities = {e: round(p, 4) for e, p in adjusted.items()}
                    predicted = max(probabilities, key=probabilities.get)
                    confidence = probabilities[predicted]
                    adjustment = {
                        "category": category,
                        "n": n,
                        "ratio": {e: round(r, 4) for e, r in ratio.items()},
                        "raw": output.probabilities,
                    }
            predictions.append(
                Prediction(
                    topic_id=topic.id,
                    household_id=topic.household_id,
                    model_version_id=version.id,
                    task=task,
                    predicted=predicted,
                    confidence=confidence,
                    probabilities=probabilities,
                    source=topic.source,
                    occurred_at=clock.now(),
                    adjustment=adjustment,
                )
            )
    session.add_all(predictions)
    return predictions


async def _segment_ratio(
    session: AsyncSession,
    household_id: uuid.UUID,
    version: ModelVersion,
    category: str,
    cache: dict[uuid.UUID, dict[str, tuple[dict[str, float], int]]],
) -> tuple[dict[str, float], int] | None:
    """The household's adjustment for one category, computed once per call."""
    if household_id not in cache:
        reviews = await repository.recent_effort_reviews(
            session, household_id, version.id, config.ADAPTATION_WINDOW
        )
        cache[household_id] = {}
        for category_name, rows in reviews.items():
            ratio = adaptation.segment_ratio(rows)
            if ratio is not None:
                cache[household_id][category_name] = (ratio, len(rows))
    return cache[household_id].get(category)


async def _members(
    session: AsyncSession, household_id: uuid.UUID, cache: dict[uuid.UUID, list[tuple[str, bool]]]
) -> list[tuple[str, bool]]:
    if household_id not in cache:
        cache[household_id] = [
            (m.user.display_name, m.is_child)
            for m in await repository.list_members(session, household_id)
        ]
    return cache[household_id]
