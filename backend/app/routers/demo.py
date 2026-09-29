"""Platform-admin endpoints behind the demo screen: define a family, generate tickets."""

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app import llm, repository
from app.auth import current_superuser
from app.clock import Clock, get_clock
from app.db import SessionFactory, get_session
from app.domain import Source
from app.models import User
from app.schemas import DemoFamilyCreate, GenerationRequest, GenerationRunRead, HouseholdRead
from app.services import demo as service
from sim.family import PRESETS, FamilySpec

router = APIRouter(prefix="/admin/demo", tags=["admin", "demo"])


@router.get("/presets", response_model=dict[str, FamilySpec])
async def list_presets(_: User = Depends(current_superuser)):
    return PRESETS


@router.post("/families", response_model=HouseholdRead, status_code=status.HTTP_201_CREATED)
async def create_family(
    body: DemoFamilyCreate,
    _: User = Depends(current_superuser),
    session: AsyncSession = Depends(get_session),
):
    if (body.preset is None) == (body.spec is None):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Give exactly one of preset or spec"
        )
    if body.preset is not None and body.preset not in PRESETS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unknown preset {body.preset!r}"
        )
    spec = PRESETS[body.preset] if body.preset else body.spec
    return await service.create_demo_family(
        session, spec, body.name, body.preset, body.training_eligible, body.random_seed
    )


async def _run_generation_in_background(
    household_id: uuid.UUID, run_id: uuid.UUID, clock: Clock
) -> None:
    # Background tasks outlive the request, so they need their own session.
    async with SessionFactory() as session:
        household = await repository.get_household(session, household_id)
        run = await repository.get_generation_run(session, run_id)
        await service.continue_generation(session, household, run, clock)


@router.post(
    "/families/{household_id}/tickets",
    response_model=GenerationRunRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_tickets(
    household_id: uuid.UUID,
    body: GenerationRequest,
    background: BackgroundTasks,
    _: User = Depends(current_superuser),
    session: AsyncSession = Depends(get_session),
    clock: Clock = Depends(get_clock),
):
    """Starts generation and returns immediately; poll the run for progress."""
    if not llm.is_configured():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "GEMINI_API_KEY is not configured")
    household = await repository.get_household(session, household_id)
    if household is None or household.kind != Source.SIMULATED:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Simulated household not found")
    run = await service.start_generation(session, household, body.count, body.random_seed)
    background.add_task(_run_generation_in_background, household.id, run.id, clock)
    return run


@router.get("/generation-runs/{run_id}", response_model=GenerationRunRead)
async def get_generation_run(
    run_id: uuid.UUID,
    _: User = Depends(current_superuser),
    session: AsyncSession = Depends(get_session),
):
    run = await repository.get_generation_run(session, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Generation run not found")
    return run
