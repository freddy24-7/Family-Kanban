"""Data access for domain tables. Tenant data is always queried by household_id:
functions that read tenant data take a household_id and filter on it."""

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain import ModelStatus, Source, Task
from app.models import (
    FamilyProfile,
    GenerationRun,
    Household,
    Invite,
    Membership,
    ModelVersion,
    Prediction,
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
    stmt = select(Topic).where(Topic.id == topic_id, Topic.household_id == household_id)
    return await session.scalar(stmt)


async def list_topics(
    session: AsyncSession, household_id: uuid.UUID, limit: int, offset: int
) -> Sequence[Topic]:
    stmt = (
        select(Topic)
        .where(Topic.household_id == household_id)
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
