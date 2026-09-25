import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app import repository
from app.clock import Clock, get_clock
from app.db import get_session
from app.domain import LabelSource
from app.schemas import LabelsUpdate, TopicCreate, TopicRead
from app.services import topics as service
from app.tenancy import HouseholdAccess, household_access, planner_access

router = APIRouter(prefix="/households/{household_id}/topics", tags=["topics"])


@router.post("", response_model=TopicRead, status_code=status.HTTP_201_CREATED)
async def create_topic(
    body: TopicCreate,
    access: HouseholdAccess = Depends(household_access),
    session: AsyncSession = Depends(get_session),
    clock: Clock = Depends(get_clock),
):
    topic = await service.create_topic(
        session, access.household, access.user.id, body.text, body.due_by, clock
    )
    return (await service.to_read_models(session, [topic]))[0]


@router.get("", response_model=list[TopicRead])
async def list_topics(
    access: HouseholdAccess = Depends(household_access),
    session: AsyncSession = Depends(get_session),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    topics = await repository.list_topics(session, access.household.id, limit, offset)
    return await service.to_read_models(session, topics)


@router.get("/{topic_id}", response_model=TopicRead)
async def get_topic(
    topic_id: uuid.UUID,
    access: HouseholdAccess = Depends(household_access),
    session: AsyncSession = Depends(get_session),
):
    topic = await repository.get_topic(session, access.household.id, topic_id)
    if topic is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Topic not found")
    return (await service.to_read_models(session, [topic]))[0]


@router.put("/{topic_id}/labels", response_model=TopicRead)
async def set_labels(
    topic_id: uuid.UUID,
    body: LabelsUpdate,
    access: HouseholdAccess = Depends(planner_access),
    session: AsyncSession = Depends(get_session),
    clock: Clock = Depends(get_clock),
):
    """Planner confirms or corrects the predicted category/effort (labelling point #1)."""
    topic = await repository.get_topic(session, access.household.id, topic_id)
    if topic is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Topic not found")
    await service.set_labels(
        session, topic, body.category, body.effort, access.user.id, LabelSource.PLANNER, clock
    )
    return (await service.to_read_models(session, [topic]))[0]
