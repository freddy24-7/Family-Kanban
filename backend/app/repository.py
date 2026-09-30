"""Data access for domain tables. Tenant data is always queried by household_id:
functions that read tenant data take a household_id and filter on it."""

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain import ModelStatus, Source, SprintStatus, Task
from app.models import (
    FamilyProfile,
    GenerationRun,
    Household,
    Invite,
    Membership,
    ModelVersion,
    Prediction,
    SimulationRun,
    Sprint,
    SprintItem,
    SprintReview,
    Topic,
    User,
)

# --- Households & memberships -------------------------------------------------


async def get_household(session: AsyncSession, household_id: uuid.UUID) -> Household | None:
    return await session.get(Household, household_id)


async def list_households(session: AsyncSession) -> Sequence[Household]:
    """Platform-admin view: every household."""
    return (await session.scalars(select(Household).order_by(Household.created_at))).all()


async def list_households_for_user(
    session: AsyncSession, user_id: uuid.UUID
) -> Sequence[Household]:
    stmt = (
        select(Household)
        .join(Membership, Membership.household_id == Household.id)
        .where(Membership.user_id == user_id)
        .order_by(Household.created_at)
    )
    return (await session.scalars(stmt)).all()


async def get_membership(
    session: AsyncSession, user_id: uuid.UUID, household_id: uuid.UUID
) -> Membership | None:
    return await session.get(Membership, (user_id, household_id))


async def list_members(session: AsyncSession, household_id: uuid.UUID) -> Sequence[Membership]:
    stmt = (
        select(Membership)
        .where(Membership.household_id == household_id)
        .order_by(Membership.joined_at)
    )
    return (await session.scalars(stmt)).unique().all()


async def get_user(session: AsyncSession, user_id: uuid.UUID) -> User | None:
    return await session.get(User, user_id)


# --- Invites --------------------------------------------------------------------


async def get_invite_by_token_hash(session: AsyncSession, token_hash: str) -> Invite | None:
    return await session.scalar(select(Invite).where(Invite.token_hash == token_hash))


async def list_pending_invites(session: AsyncSession, household_id: uuid.UUID) -> Sequence[Invite]:
    stmt = (
        select(Invite)
        .where(Invite.household_id == household_id, Invite.accepted_at.is_(None))
        .order_by(Invite.created_at)
    )
    return (await session.scalars(stmt)).all()


# --- Topics & predictions -------------------------------------------------------


async def get_topic(
    session: AsyncSession, household_id: uuid.UUID, topic_id: uuid.UUID
) -> Topic | None:
    stmt = select(Topic).where(
        Topic.id == topic_id, Topic.household_id == household_id, Topic.deleted_at.is_(None)
    )
    return await session.scalar(stmt)


async def list_topics(
    session: AsyncSession, household_id: uuid.UUID, limit: int, offset: int
) -> Sequence[Topic]:
    stmt = (
        select(Topic)
        .where(Topic.household_id == household_id, Topic.deleted_at.is_(None))
        .order_by(Topic.occurred_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return (await session.scalars(stmt)).all()


async def latest_predictions(
    session: AsyncSession, household_id: uuid.UUID, topic_ids: Sequence[uuid.UUID]
) -> dict[uuid.UUID, dict[Task, Prediction]]:
    """Most recent prediction per (topic, task)."""
    if not topic_ids:
        return {}
    stmt = (
        select(Prediction)
        .where(Prediction.household_id == household_id, Prediction.topic_id.in_(topic_ids))
        .order_by(Prediction.created_at)
    )
    result: dict[uuid.UUID, dict[Task, Prediction]] = {}
    for prediction in (await session.scalars(stmt)).all():
        result.setdefault(prediction.topic_id, {})[prediction.task] = prediction
    return result


# --- Model versions -------------------------------------------------------------


async def active_model_versions(session: AsyncSession) -> dict[Task, ModelVersion]:
    stmt = select(ModelVersion).where(ModelVersion.status == ModelStatus.ACTIVE)
    return {mv.task: mv for mv in (await session.scalars(stmt)).all()}


async def ensure_stub_model_versions(session: AsyncSession) -> None:
    """Idempotent: make sure every task has an active model. Before Phase 3 that
    is the stub, which predicts 'other' / 'M' with zero confidence."""
    active = await active_model_versions(session)
    stub_labels = {Task.CATEGORY: "other", Task.EFFORT: "M"}
    for task, label in stub_labels.items():
        if task in active:
            continue
        name = f"{task.value}-stub-0"
        if await session.scalar(select(ModelVersion).where(ModelVersion.name == name)):
            continue
        session.add(
            ModelVersion(
                task=task,
                name=name,
                status=ModelStatus.ACTIVE,
                artifact_uri=f"stub://{label}",
                notes="Placeholder until the first trained model (Phase 3).",
            )
        )
    await session.commit()


def source_for(household: Household) -> Source:
    """Provenance is derived from the household, never chosen by the caller."""
    return household.kind


# --- Demo families ----------------------------------------------------------------


async def get_family_profile(
    session: AsyncSession, household_id: uuid.UUID
) -> FamilyProfile | None:
    return await session.get(FamilyProfile, household_id)


async def list_topic_texts(session: AsyncSession, household_id: uuid.UUID) -> Sequence[str]:
    return (
        await session.scalars(select(Topic.text).where(Topic.household_id == household_id))
    ).all()


async def get_generation_run(session: AsyncSession, run_id: uuid.UUID) -> GenerationRun | None:
    return await session.get(GenerationRun, run_id)


async def find_demo_household(session: AsyncSession, preset_key: str) -> Household | None:
    """Oldest training-eligible simulated household created from this preset."""
    stmt = (
        select(Household)
        .join(FamilyProfile, FamilyProfile.household_id == Household.id)
        .where(
            FamilyProfile.preset_key == preset_key,
            Household.kind == Source.SIMULATED,
            Household.training_eligible.is_(True),
        )
        .order_by(Household.created_at)
        .limit(1)
    )
    return await session.scalar(stmt)


async def count_topics(session: AsyncSession, household_id: uuid.UUID) -> int:
    stmt = select(func.count()).select_from(Topic).where(Topic.household_id == household_id)
    return await session.scalar(stmt) or 0


# --- Sprints & backlog ----------------------------------------------------------------

OPEN_SPRINT_STATUSES = (SprintStatus.PLANNED, SprintStatus.ACTIVE)


async def list_backlog(
    session: AsyncSession, household_id: uuid.UUID, limit: int, offset: int
) -> Sequence[Topic]:
    """Topics that still need planning: not in a planned/active sprint, and not
    completed in an earlier sprint."""
    in_open_sprint = (
        select(SprintItem.id)
        .join(Sprint, Sprint.id == SprintItem.sprint_id)
        .where(SprintItem.topic_id == Topic.id, Sprint.status.in_(OPEN_SPRINT_STATUSES))
        .exists()
    )
    done = (
        select(SprintItem.id)
        .where(SprintItem.topic_id == Topic.id, SprintItem.completed.is_(True))
        .exists()
    )
    stmt = (
        select(Topic)
        .where(
            Topic.household_id == household_id,
            Topic.deleted_at.is_(None),
            ~in_open_sprint,
            ~done,
        )
        .order_by(Topic.occurred_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return (await session.scalars(stmt)).all()


async def list_sprints(session: AsyncSession, household_id: uuid.UUID) -> Sequence[Sprint]:
    stmt = (
        select(Sprint).where(Sprint.household_id == household_id).order_by(Sprint.start_date.desc())
    )
    return (await session.scalars(stmt)).all()


async def get_sprint(
    session: AsyncSession, household_id: uuid.UUID, sprint_id: uuid.UUID
) -> Sprint | None:
    stmt = select(Sprint).where(Sprint.id == sprint_id, Sprint.household_id == household_id)
    return await session.scalar(stmt)


async def sprints_with_status(
    session: AsyncSession, household_id: uuid.UUID, status: SprintStatus
) -> Sequence[Sprint]:
    stmt = select(Sprint).where(Sprint.household_id == household_id, Sprint.status == status)
    return (await session.scalars(stmt)).all()


async def list_sprint_items(session: AsyncSession, sprint_id: uuid.UUID) -> Sequence[SprintItem]:
    stmt = (
        select(SprintItem)
        .where(SprintItem.sprint_id == sprint_id)
        .order_by(SprintItem.status, SprintItem.position, SprintItem.created_at)
    )
    return (await session.scalars(stmt)).unique().all()


async def get_sprint_item(
    session: AsyncSession, sprint_id: uuid.UUID, item_id: uuid.UUID
) -> SprintItem | None:
    stmt = select(SprintItem).where(SprintItem.id == item_id, SprintItem.sprint_id == sprint_id)
    return await session.scalar(stmt)


async def topic_is_planned_or_done(session: AsyncSession, topic_id: uuid.UUID) -> bool:
    stmt = (
        select(SprintItem.id)
        .join(Sprint, Sprint.id == SprintItem.sprint_id)
        .where(
            SprintItem.topic_id == topic_id,
            (Sprint.status.in_(OPEN_SPRINT_STATUSES)) | (SprintItem.completed.is_(True)),
        )
        .limit(1)
    )
    return await session.scalar(stmt) is not None


async def get_sprint_review(session: AsyncSession, sprint_id: uuid.UUID) -> SprintReview | None:
    return await session.scalar(select(SprintReview).where(SprintReview.sprint_id == sprint_id))


# --- Simulations ------------------------------------------------------------------------


async def list_simulation_runs(session: AsyncSession) -> Sequence[SimulationRun]:
    stmt = select(SimulationRun).order_by(SimulationRun.created_at.desc())
    return (await session.scalars(stmt)).all()


async def get_simulation_run(session: AsyncSession, run_id: uuid.UUID) -> SimulationRun | None:
    return await session.get(SimulationRun, run_id)


async def model_versions_by_name(
    session: AsyncSession, names: dict[str, str]
) -> dict[Task, ModelVersion]:
    """{task: version name} -> {Task: ModelVersion}; raises on unknown or mismatched names."""
    out = {}
    for task, name in names.items():
        version = await session.scalar(select(ModelVersion).where(ModelVersion.name == name))
        if version is None or str(version.task) != task:
            raise ValueError(f"No {task} model version named {name!r}")
        out[Task(task)] = version
    return out
