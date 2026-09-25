"""Household access control. Every household-scoped route depends on
`household_access`, which proves the caller may act in that household."""

import uuid
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app import repository
from app.auth import current_active_user
from app.db import get_session
from app.models import Household, User


@dataclass(frozen=True)
class HouseholdAccess:
    household: Household
    user: User
    is_planner: bool
    is_reviewer: bool
    is_member: bool


async def household_access(
    household_id: uuid.UUID,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_session),
) -> HouseholdAccess:
    household = await repository.get_household(session, household_id)
    membership = await repository.get_membership(session, user.id, household_id)
    if household is not None and membership is not None:
        return HouseholdAccess(
            household, user, membership.is_planner, membership.is_reviewer, is_member=True
        )
    if household is not None and user.is_superuser:
        # Platform admin may operate any household (e.g. simulated ones).
        return HouseholdAccess(household, user, True, True, is_member=False)
    # 404 rather than 403: don't reveal that another family's household exists.
    raise HTTPException(status.HTTP_404_NOT_FOUND, "Household not found")


async def planner_access(access: HouseholdAccess = Depends(household_access)) -> HouseholdAccess:
    if not access.is_planner:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Planner role required")
    return access
