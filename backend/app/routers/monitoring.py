"""Model Health Dashboard endpoints (platform admin)."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import repository
from app.auth import current_superuser
from app.db import get_session
from app.domain import Source
from app.models import Household, User
from app.services import monitoring as service

router = APIRouter(prefix="/admin/monitoring", tags=["admin", "monitoring"])


@router.get("/simulations/{run_id}")
async def simulation(
    run_id: uuid.UUID,
    reference: service.ReferenceKind = "holdout",
    window: int = Query(4, ge=1, le=12),
    _: User = Depends(current_superuser),
    session: AsyncSession = Depends(get_session),
):
    run = await repository.get_simulation_run(session, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Simulation not found")
    result = await service.report(
        session,
        [run.household_id],
        reference,
        window,
        run.scenario.get("events", []),
        run.start_date,
    )
    return {
        "run": {
            "id": str(run.id),
            "scenario": run.scenario,
            "start_date": run.start_date.isoformat(),
        },
        **result,
    }


@router.get("/real")
async def real(
    reference: service.ReferenceKind = "holdout",
    window: int = Query(4, ge=1, le=12),
    _: User = Depends(current_superuser),
    session: AsyncSession = Depends(get_session),
):
    """All real households together (a single family has too few tickets per week)."""
    ids = list(
        (await session.scalars(select(Household.id).where(Household.kind == Source.REAL))).all()
    )
    return await service.report(session, ids, reference, window)


@router.get("/models")
async def models(
    _: User = Depends(current_superuser), session: AsyncSession = Depends(get_session)
):
    return await service.model_log(session)
