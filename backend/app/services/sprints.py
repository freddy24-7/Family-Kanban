"""Sprint planning, the kanban board and the sprint review (labelling point #2)."""

import uuid
from datetime import date

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app import repository
from app.clock import Clock
from app.domain import Effort, ItemStatus, SprintStatus
from app.models import Household, Sprint, SprintItem, SprintReview
from app.tenancy import HouseholdAccess


def _conflict(message: str) -> HTTPException:
    return HTTPException(status.HTTP_409_CONFLICT, message)


async def create_sprint(
    session: AsyncSession, household: Household, name: str, start: date, end: date
) -> Sprint:
    if end < start:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "end_date is before start_date")
    if await repository.sprints_with_status(session, household.id, SprintStatus.PLANNED):
        raise _conflict("There is already a planned sprint")
    sprint = Sprint(
        household_id=household.id,
        name=name,
        start_date=start,
        end_date=end,
        status=SprintStatus.PLANNED,
    )
    session.add(sprint)
    await session.commit()
    return sprint


async def start_sprint(session: AsyncSession, sprint: Sprint, clock: Clock) -> Sprint:
    if sprint.status != SprintStatus.PLANNED:
        raise _conflict("Only a planned sprint can be started")
    if await repository.sprints_with_status(session, sprint.household_id, SprintStatus.ACTIVE):
        raise _conflict("Another sprint is still active; complete it first")
    sprint.status, sprint.started_at = SprintStatus.ACTIVE, clock.now()
    await session.commit()
    return sprint


async def _check_assignee(
    session: AsyncSession, household_id: uuid.UUID, assignee_id: uuid.UUID | None
) -> None:
    if assignee_id is not None and not await repository.get_membership(
        session, assignee_id, household_id
    ):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Assignee is not a member of this household"
        )


async def add_item(
    session: AsyncSession, sprint: Sprint, topic_id: uuid.UUID, assignee_id: uuid.UUID | None
) -> SprintItem:
    if sprint.status == SprintStatus.COMPLETED:
        raise _conflict("Sprint is completed")
    topic = await repository.get_topic(session, sprint.household_id, topic_id)
    if topic is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Topic not found")
    if await repository.topic_is_planned_or_done(session, topic.id):
        raise _conflict("Topic is already planned or done")
    await _check_assignee(session, sprint.household_id, assignee_id)
    item = SprintItem(
        sprint_id=sprint.id,
        household_id=sprint.household_id,
        topic_id=topic.id,
        assignee_id=assignee_id,
        status=ItemStatus.TODO,
    )
    session.add(item)
    await session.commit()
    return await repository.get_sprint_item(session, sprint.id, item.id)


async def update_item(
    session: AsyncSession, sprint: Sprint, item: SprintItem, assignee_id: uuid.UUID | None
) -> SprintItem:
    if sprint.status == SprintStatus.COMPLETED:
        raise _conflict("Sprint is completed")
    await _check_assignee(session, sprint.household_id, assignee_id)
    item.assignee_id = assignee_id
    await session.commit()
    return item


async def remove_item(session: AsyncSession, sprint: Sprint, item: SprintItem) -> None:
    if sprint.status == SprintStatus.COMPLETED:
        raise _conflict("Sprint is completed")
    await session.delete(item)
    await session.commit()


async def move_item(
    session: AsyncSession,
    access: HouseholdAccess,
    sprint: Sprint,
    item: SprintItem,
    new_status: ItemStatus,
    position: int,
    clock: Clock,
) -> SprintItem:
    """Assignees move their own items; planners may move any item."""
    if sprint.status != SprintStatus.ACTIVE:
        raise _conflict("Items can only be moved in an active sprint")
    if not access.is_planner and item.assignee_id != access.user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You can only move your own items")
    if item.status != new_status:
        item.status_changed_at = clock.now()
    item.status, item.position = new_status, position
    await session.commit()
    return item


async def review_item(
    session: AsyncSession,
    access: HouseholdAccess,
    sprint: Sprint,
    item: SprintItem,
    completed: bool,
    effort_actual: Effort | None,
    note: str | None,
    clock: Clock,
) -> SprintItem:
    if sprint.status != SprintStatus.ACTIVE:
        raise _conflict("Only items of the active sprint can be reviewed")
    if completed and effort_actual is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "A completed item needs effort_actual"
        )
    item.completed = completed
    item.effort_actual = effort_actual
    item.review_note = note.strip() if note and note.strip() else None
    item.reviewed_at, item.reviewed_by = clock.now(), access.user.id
    await session.commit()
    return item


async def complete_sprint(
    session: AsyncSession, access: HouseholdAccess, sprint: Sprint, notes: str | None, clock: Clock
) -> SprintReview:
    """Close the sprint. Every item must be reviewed first; items not completed
    return to the backlog automatically (they're no longer in an open sprint)."""
    if sprint.status != SprintStatus.ACTIVE:
        raise _conflict("Only the active sprint can be completed")
    items = await repository.list_sprint_items(session, sprint.id)
    unreviewed = [str(i.id) for i in items if i.reviewed_at is None]
    if unreviewed:
        raise _conflict(f"{len(unreviewed)} item(s) still need a review")
    review = SprintReview(
        sprint_id=sprint.id,
        household_id=sprint.household_id,
        notes=notes.strip() if notes and notes.strip() else None,
        completed_count=sum(1 for i in items if i.completed),
        total_count=len(items),
        reviewed_by=access.user.id,
    )
    sprint.status, sprint.completed_at = SprintStatus.COMPLETED, clock.now()
    session.add(review)
    await session.commit()
    return review
