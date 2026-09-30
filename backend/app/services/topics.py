"""Topic intake and labelling."""

import uuid
from collections.abc import Sequence
from datetime import date

from fastapi import HTTPException, status
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
    model_versions: dict | None = None,
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
    await classify_topics(session, [topic], clock, model_versions)
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


def _check_can_change(topic: Topic, user_id: uuid.UUID, is_planner: bool) -> None:
    if not is_planner and topic.created_by != user_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "Only the creator or a planner can change this"
        )


async def update_topic(
    session: AsyncSession,
    topic: Topic,
    user_id: uuid.UUID,
    is_planner: bool,
    text: str | None,
    due_by: date | None,
    due_by_set: bool,
    clock: Clock,
) -> Topic:
    """Edit text and/or due date. A changed text is re-classified: the new
    predictions are appended (the old ones stay, predictions are append-only).
    Confirmed labels are kept: a typo fix doesn't change what the task is."""
    _check_can_change(topic, user_id, is_planner)
    changed_text = text is not None and text.strip() != topic.text
    if changed_text:
        topic.text = text.strip()
    if due_by_set:
        topic.due_by = due_by
    topic.updated_at = clock.now()
    if changed_text:
        await classify_topics(session, [topic], clock)
    await session.commit()
    return topic


async def delete_topic(
    session: AsyncSession, topic: Topic, user_id: uuid.UUID, is_planner: bool, clock: Clock
) -> None:
    """Soft delete: hidden from the app and excluded from training data."""
    _check_can_change(topic, user_id, is_planner)
    if await repository.topic_is_planned_or_done(session, topic.id):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Topic is in a sprint or already done; remove it from the sprint first",
        )
    topic.deleted_at = clock.now()
    await session.commit()
