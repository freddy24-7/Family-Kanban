"""Household creation, membership and email invites."""

import hashlib
import secrets
import uuid
from datetime import timedelta

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app import config, mailer, messages_nl, repository
from app.clock import Clock
from app.domain import Source
from app.models import Household, Invite, Membership, User


async def create_household(
    session: AsyncSession,
    name: str,
    owner: User | None,
    kind: Source = Source.REAL,
    training_eligible: bool = False,
) -> Household:
    """The owner (if any) becomes planner + reviewer. Simulated households created
    by the platform admin have no owner membership."""
    household = Household(
        id=uuid.uuid4(), name=name, kind=kind, training_eligible=training_eligible
    )
    session.add(household)
    if owner is not None:
        session.add(
            Membership(
                user_id=owner.id, household_id=household.id, is_planner=True, is_reviewer=True
            )
        )
    await session.commit()
    return household


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def create_invite(
    session: AsyncSession,
    household: Household,
    inviter: User,
    email: str,
    is_planner: bool,
    is_reviewer: bool,
    clock: Clock,
    is_child: bool = False,
) -> Invite:
    token = secrets.token_urlsafe(32)
    invite = Invite(
        household_id=household.id,
        email=email.lower(),
        token_hash=hash_token(token),
        is_planner=is_planner,
        is_reviewer=is_reviewer,
        is_child=is_child,
        invited_by=inviter.id,
        expires_at=clock.now() + timedelta(days=config.INVITE_LIFETIME_DAYS),
    )
    session.add(invite)
    await session.commit()
    subject, body = messages_nl.household_invite(
        household.name, inviter.display_name, f"{config.FRONTEND_URL}/invite?token={token}"
    )
    await mailer.send_email(invite.email, subject, body)
    return invite


async def accept_invite(session: AsyncSession, user: User, token: str, clock: Clock) -> Membership:
    invite = await repository.get_invite_by_token_hash(session, hash_token(token))
    if invite is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invite not found")
    if invite.accepted_at is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Invite already used")
    if invite.expires_at <= clock.now():
        raise HTTPException(status.HTTP_410_GONE, "Invite expired")
    if invite.email != user.email.lower():
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invite was sent to a different email")
    if await repository.get_membership(session, user.id, invite.household_id):
        raise HTTPException(status.HTTP_409_CONFLICT, "Already a member of this household")
    membership = Membership(
        user_id=user.id,
        household_id=invite.household_id,
        is_planner=invite.is_planner,
        is_reviewer=invite.is_reviewer,
        is_child=invite.is_child,
    )
    invite.accepted_at = clock.now()
    session.add(membership)
    await session.commit()
    return membership
