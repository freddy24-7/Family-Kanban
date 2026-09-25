"""Topic intake and labelling."""

import uuid
from collections.abc import Sequence
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from app import config, repository
from app.clock import Clock
from app.domain import Category, Effort, LabelSource, Task
from app.models import Household, Prediction, Topic
from app.schemas import LabelsRead, PredictionRead, TopicRead
from app.services.classification import classify_topics


async def create_topic(
    session: AsyncSession,
    household: Household,
    created_by: uuid.UUID | None,
    text: str,
    due_by: date | None,
    clock: Clock,
) -> Topic:
    topic = Topic(
        id=uuid.uuid4(),
        household_id=household.id,
        text=text.strip(),
        due_by=due_by,
        created_by=created_by,
        occurred_at=clock.now(),
        source=repository.source_for(household),
    )
    session.add(topic)
    await session.flush()
    await classify_topics(session, [topic], clock)
    await session.commit()
    return topic


async def set_labels(
    session: AsyncSession,
    topic: Topic,
    category: Category,
    effort: Effort,
    labeled_by: uuid.UUID,
    label_source: LabelSource,
    clock: Clock,
) -> Topic:
    """Record ground-truth labels, and whether the human changed the prediction
    they were shown (to detect rubber-stamping)."""
    latest = (await repository.latest_predictions(session, topic.household_id, [topic.id])).get(
        topic.id, {}
    )
    category_prediction = latest.get(Task.CATEGORY)
    effort_prediction = latest.get(Task.EFFORT)
    topic.category_label = category
    topic.effort_label = effort
    topic.label_source = label_source
    topic.labeled_at = clock.now()
    topic.labeled_by = labeled_by
    topic.category_prediction_changed = (
        None if category_prediction is None else category_prediction.predicted != category
    )
    topic.effort_prediction_changed = (
        None if effort_prediction is None else effort_prediction.predicted != effort
    )
    await session.commit()
    return topic


def needs_review(predictions: dict[Task, Prediction]) -> bool:
    if len(predictions) < len(Task):
        return True
    return any(p.confidence < config.LOW_CONFIDENCE_THRESHOLD for p in predictions.values())


async def to_read_models(session: AsyncSession, topics: Sequence[Topic]) -> list[TopicRead]:
    if not topics:
        return []
    household_id = topics[0].household_id
    latest = await repository.latest_predictions(session, household_id, [t.id for t in topics])
    result = []
    for topic in topics:
        predictions = latest.get(topic.id, {})
        category = predictions.get(Task.CATEGORY)
        effort = predictions.get(Task.EFFORT)
        result.append(
            TopicRead(
                id=topic.id,
                household_id=topic.household_id,
                text=topic.text,
                due_by=topic.due_by,
                created_by=topic.created_by,
                occurred_at=topic.occurred_at,
                source=topic.source,
                prediction=PredictionRead(
                    category=category.predicted if category else None,
                    category_confidence=category.confidence if category else None,
                    effort=effort.predicted if effort else None,
                    effort_confidence=effort.confidence if effort else None,
                    needs_review=needs_review(predictions),
                ),
                labels=LabelsRead(
                    category=topic.category_label,
                    effort=topic.effort_label,
                    source=topic.label_source,
                    labeled_at=topic.labeled_at,
                ),
            )
        )
    return result
