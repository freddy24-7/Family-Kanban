"""Runs the active models on topics and logs append-only Prediction rows."""

import logging
import uuid
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app import repository
from app.clock import Clock
from app.models import Prediction, Topic
from ml import registry
from ml.text import ROLE_MASK_VERSION, mask_member_names

log = logging.getLogger(__name__)


async def classify_topics(
    session: AsyncSession, topics: Sequence[Topic], clock: Clock
) -> list[Prediction]:
    """Predict every task for the given topics with the currently active model
    versions. Each prediction records which model version produced it, so its
    accuracy can later be attributed to the right model. Caller commits.

    Fails soft per task: if a model can't be loaded, the topic is still created
    (it just has no prediction for that task and is flagged for review)."""
    if not topics:
        return []
    members: dict[uuid.UUID, list[tuple[str, bool]]] = {}
    predictions: list[Prediction] = []
    for task, version in (await repository.active_model_versions(session)).items():
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
        for topic, output in zip(topics, model.predict(texts), strict=True):
            predictions.append(
                Prediction(
                    topic_id=topic.id,
                    household_id=topic.household_id,
                    model_version_id=version.id,
                    task=task,
                    predicted=output.predicted,
                    confidence=output.confidence,
                    probabilities=output.probabilities,
                    source=topic.source,
                    occurred_at=clock.now(),
                )
            )
    session.add_all(predictions)
    return predictions


async def _members(
    session: AsyncSession, household_id: uuid.UUID, cache: dict[uuid.UUID, list[tuple[str, bool]]]
) -> list[tuple[str, bool]]:
    if household_id not in cache:
        cache[household_id] = [
            (m.user.display_name, m.is_child)
            for m in await repository.list_members(session, household_id)
        ]
    return cache[household_id]
