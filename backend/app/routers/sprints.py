import uuid
from collections.abc import Sequence

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app import repository
from app.clock import Clock, get_clock
from app.db import get_session
from app.domain import ItemStatus
from app.models import Sprint, SprintItem
from app.schemas import (
    SprintCompleteIn,
    SprintCreate,
    SprintDetail,
    SprintItemCreate,
    SprintItemMove,
    SprintItemRead,
    SprintItemReviewIn,
    SprintItemUpdate,
    SprintRead,
    SprintReviewRead,
    TopicRead,
    TopicSuggestionRead,
)
from app.services import similar
from app.services import sprints as service
from app.services import topics as topics_service
from app.tenancy import HouseholdAccess, household_access, planner_access, reviewer_access

router = APIRouter(prefix="/households/{household_id}", tags=["sprints"])


@router.get("/backlog", response_model=list[TopicRead])
async def backlog(
    access: HouseholdAccess = Depends(household_access),
    session: AsyncSession = Depends(get_session),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Topics not yet planned into an open sprint and not completed."""
    topics = await repository.list_backlog(session, access.household.id, limit, offset)
    return await topics_service.to_read_models(session, topics)


@router.get("/backlog/suggestions", response_model=list[TopicSuggestionRead])
async def backlog_suggestions(
    access: HouseholdAccess = Depends(household_access),
    session: AsyncSession = Depends(get_session),
    k: int = Query(3, ge=1, le=10),
    limit: int = Query(100, ge=1, le=500),
):
    """Per backlog topic: similar earlier, reviewed tasks of this household (Phase 8).
    Same topics, same order as GET /backlog."""
    topics = await repository.list_backlog(session, access.household.id, limit, 0)
    return await similar.suggestions(session, access.household, topics, k)


# --- helpers ----------------------------------------------------------------------------


async def _sprint(session: AsyncSession, access: HouseholdAccess, sprint_id: uuid.UUID) -> Sprint:
    sprint = await repository.get_sprint(session, access.household.id, sprint_id)
    if sprint is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sprint not found")
    return sprint


async def _item(session: AsyncSession, sprint: Sprint, item_id: uuid.UUID) -> SprintItem:
    item = await repository.get_sprint_item(session, sprint.id, item_id)
    if item is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Sprint item not found")
    return item


async def _item_reads(
    session: AsyncSession, household_id: uuid.UUID, items: Sequence[SprintItem]
) -> list[SprintItemRead]:
    names = {
        m.user_id: m.user.display_name for m in await repository.list_members(session, household_id)
    }
    topics = {
        t.id: t for t in await topics_service.to_read_models(session, [i.topic for i in items])
    }
    return [
        SprintItemRead(
            id=i.id,
            topic=topics[i.topic_id],
            assignee_id=i.assignee_id,
            assignee_name=names.get(i.assignee_id),
            status=i.status,
            position=i.position,
            completed=i.completed,
            effort_actual=i.effort_actual,
            review_note=i.review_note,
            reviewed_at=i.reviewed_at,
        )
        for i in items
    ]


def _sprint_read(sprint: Sprint, items: Sequence[SprintItem]) -> dict:
    return {
        "id": sprint.id,
        "name": sprint.name,
        "start_date": sprint.start_date,
        "end_date": sprint.end_date,
        "status": sprint.status,
        "started_at": sprint.started_at,
        "completed_at": sprint.completed_at,
        "item_count": len(items),
        "done_count": sum(1 for i in items if i.status == ItemStatus.DONE),
    }


async def _detail(session: AsyncSession, sprint: Sprint) -> SprintDetail:
    items = await repository.list_sprint_items(session, sprint.id)
    review = await repository.get_sprint_review(session, sprint.id)
    return SprintDetail(
        **_sprint_read(sprint, items),
        items=await _item_reads(session, sprint.household_id, items),
        review=SprintReviewRead.model_validate(review) if review else None,
    )


# --- sprints -----------------------------------------------------------------------------


@router.get("/sprints", response_model=list[SprintRead])
async def list_sprints(
    access: HouseholdAccess = Depends(household_access),
    session: AsyncSession = Depends(get_session),
):
    result = []
    for sprint in await repository.list_sprints(session, access.household.id):
        result.append(
            SprintRead(
                **_sprint_read(sprint, await repository.list_sprint_items(session, sprint.id))
            )
        )
    return result


@router.post("/sprints", response_model=SprintDetail, status_code=status.HTTP_201_CREATED)
async def create_sprint(
    body: SprintCreate,
    access: HouseholdAccess = Depends(planner_access),
    session: AsyncSession = Depends(get_session),
):
    sprint = await service.create_sprint(
        session, access.household, body.name, body.start_date, body.end_date
    )
    return await _detail(session, sprint)


@router.get("/sprints/{sprint_id}", response_model=SprintDetail)
async def get_sprint(
    sprint_id: uuid.UUID,
    access: HouseholdAccess = Depends(household_access),
    session: AsyncSession = Depends(get_session),
):
    return await _detail(session, await _sprint(session, access, sprint_id))


@router.post("/sprints/{sprint_id}/start", response_model=SprintDetail)
async def start_sprint(
    sprint_id: uuid.UUID,
    access: HouseholdAccess = Depends(planner_access),
    session: AsyncSession = Depends(get_session),
    clock: Clock = Depends(get_clock),
):
    sprint = await service.start_sprint(session, await _sprint(session, access, sprint_id), clock)
    return await _detail(session, sprint)


@router.post("/sprints/{sprint_id}/complete", response_model=SprintDetail)
async def complete_sprint(
    sprint_id: uuid.UUID,
    body: SprintCompleteIn,
    access: HouseholdAccess = Depends(reviewer_access),
    session: AsyncSession = Depends(get_session),
    clock: Clock = Depends(get_clock),
):
    sprint = await _sprint(session, access, sprint_id)
    await service.complete_sprint(session, access, sprint, body.notes, clock)
    return await _detail(session, sprint)


# --- items -------------------------------------------------------------------------------


@router.post(
    "/sprints/{sprint_id}/items", response_model=SprintItemRead, status_code=status.HTTP_201_CREATED
)
async def add_item(
    sprint_id: uuid.UUID,
    body: SprintItemCreate,
    access: HouseholdAccess = Depends(planner_access),
    session: AsyncSession = Depends(get_session),
):
    sprint = await _sprint(session, access, sprint_id)
    item = await service.add_item(session, sprint, body.topic_id, body.assignee_id)
    return (await _item_reads(session, sprint.household_id, [item]))[0]


@router.patch("/sprints/{sprint_id}/items/{item_id}", response_model=SprintItemRead)
async def update_item(
    sprint_id: uuid.UUID,
    item_id: uuid.UUID,
    body: SprintItemUpdate,
    access: HouseholdAccess = Depends(planner_access),
    session: AsyncSession = Depends(get_session),
):
    sprint = await _sprint(session, access, sprint_id)
    item = await service.update_item(
        session, sprint, await _item(session, sprint, item_id), body.assignee_id
    )
    return (await _item_reads(session, sprint.household_id, [item]))[0]


@router.delete("/sprints/{sprint_id}/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_item(
    sprint_id: uuid.UUID,
    item_id: uuid.UUID,
    access: HouseholdAccess = Depends(planner_access),
    session: AsyncSession = Depends(get_session),
):
    sprint = await _sprint(session, access, sprint_id)
    await service.remove_item(session, sprint, await _item(session, sprint, item_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put("/sprints/{sprint_id}/items/{item_id}/status", response_model=SprintItemRead)
async def move_item(
    sprint_id: uuid.UUID,
    item_id: uuid.UUID,
    body: SprintItemMove,
    access: HouseholdAccess = Depends(household_access),
    session: AsyncSession = Depends(get_session),
    clock: Clock = Depends(get_clock),
):
    sprint = await _sprint(session, access, sprint_id)
    item = await _item(session, sprint, item_id)
    item = await service.move_item(session, access, sprint, item, body.status, body.position, clock)
    return (await _item_reads(session, sprint.household_id, [item]))[0]


@router.put("/sprints/{sprint_id}/items/{item_id}/review", response_model=SprintItemRead)
async def review_item(
    sprint_id: uuid.UUID,
    item_id: uuid.UUID,
    body: SprintItemReviewIn,
    access: HouseholdAccess = Depends(reviewer_access),
    session: AsyncSession = Depends(get_session),
    clock: Clock = Depends(get_clock),
):
    """Labelling point #2: was it done, and how much effort did it actually take?"""
    sprint = await _sprint(session, access, sprint_id)
    item = await _item(session, sprint, item_id)
    item = await service.review_item(
        session, access, sprint, item, body.completed, body.effort_actual, body.note, clock
    )
    return (await _item_reads(session, sprint.household_id, [item]))[0]
