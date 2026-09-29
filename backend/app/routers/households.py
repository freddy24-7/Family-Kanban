import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app import repository
from app.auth import current_active_user, current_superuser
from app.clock import Clock, get_clock
from app.db import get_session
from app.models import User
from app.schemas import (
    AdminHouseholdCreate,
    AdminHouseholdUpdate,
    HouseholdCreate,
    HouseholdDetail,
    HouseholdRead,
    InviteAccept,
    InviteCreate,
    InviteRead,
    MemberRead,
)
from app.services import households as service
from app.tenancy import HouseholdAccess, household_access, planner_access

router = APIRouter(tags=["households"])


@router.post("/households", response_model=HouseholdRead, status_code=status.HTTP_201_CREATED)
async def create_household(
    body: HouseholdCreate,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_session),
):
    return await service.create_household(session, body.name, owner=user)


@router.get("/households", response_model=list[HouseholdRead])
async def my_households(
    user: User = Depends(current_active_user), session: AsyncSession = Depends(get_session)
):
    return await repository.list_households_for_user(session, user.id)


@router.get("/households/{household_id}", response_model=HouseholdDetail)
async def get_household(
    access: HouseholdAccess = Depends(household_access),
    session: AsyncSession = Depends(get_session),
):
    members = await repository.list_members(session, access.household.id)
    return HouseholdDetail(
        **HouseholdRead.model_validate(access.household).model_dump(),
        members=[
            MemberRead(
                user_id=m.user_id,
                display_name=m.user.display_name,
                email=m.user.email,
                is_planner=m.is_planner,
                is_reviewer=m.is_reviewer,
                is_child=m.is_child,
                is_simulated=m.user.is_simulated,
            )
            for m in members
        ],
    )


@router.post(
    "/households/{household_id}/invites",
    response_model=InviteRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_invite(
    body: InviteCreate,
    access: HouseholdAccess = Depends(planner_access),
    session: AsyncSession = Depends(get_session),
    clock: Clock = Depends(get_clock),
):
    return await service.create_invite(
        session,
        access.household,
        access.user,
        body.email,
        body.is_planner,
        body.is_reviewer,
        clock,
        is_child=body.is_child,
    )


@router.get("/households/{household_id}/invites", response_model=list[InviteRead])
async def list_invites(
    access: HouseholdAccess = Depends(planner_access),
    session: AsyncSession = Depends(get_session),
):
    return await repository.list_pending_invites(session, access.household.id)


@router.post("/invites/accept", response_model=HouseholdRead)
async def accept_invite(
    body: InviteAccept,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_session),
    clock: Clock = Depends(get_clock),
):
    membership = await service.accept_invite(session, user, body.token, clock)
    return await repository.get_household(session, membership.household_id)


# --- Platform admin -------------------------------------------------------------


@router.get("/admin/households", response_model=list[HouseholdRead], tags=["admin"])
async def admin_list_households(
    _: User = Depends(current_superuser), session: AsyncSession = Depends(get_session)
):
    return await repository.list_households(session)


@router.post(
    "/admin/households",
    response_model=HouseholdRead,
    status_code=status.HTTP_201_CREATED,
    tags=["admin"],
)
async def admin_create_household(
    body: AdminHouseholdCreate,
    _: User = Depends(current_superuser),
    session: AsyncSession = Depends(get_session),
):
    return await service.create_household(
        session, body.name, owner=None, kind=body.kind, training_eligible=body.training_eligible
    )


@router.patch("/admin/households/{household_id}", response_model=HouseholdRead, tags=["admin"])
async def admin_update_household(
    household_id: uuid.UUID,
    body: AdminHouseholdUpdate,
    _: User = Depends(current_superuser),
    session: AsyncSession = Depends(get_session),
):
    household = await repository.get_household(session, household_id)
    if household is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Household not found")
    household.training_eligible = body.training_eligible
    await session.commit()
    return household
