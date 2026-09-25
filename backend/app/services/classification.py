"""Runs the active models on topics and logs append-only Prediction rows."""

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app import repository
from app.clock import Clock
from app.models import Prediction, Topic
from ml.registry import load_model


async def classify_topics(
    session: AsyncSession, topics: Sequence[Topic], clock: Clock
) -> list[Prediction]:
    """Predict every task for the given topics with the currently active model
    versions. Each prediction records which model version produced it, so its
    accuracy can later be attributed to the right model. Caller commits."""
    if not topics:
        return []
    predictions: list[Prediction] = []
    texts = [topic.text for topic in topics]
    for task, version in (await repository.active_model_versions(session)).items():
        outputs = load_model(version.artifact_uri).predict(texts)
        for topic, output in zip(topics, outputs, strict=True):
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
