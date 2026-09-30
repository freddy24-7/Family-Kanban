"""Platform-admin endpoints for the simulator. Runs take minutes, so they execute as
background tasks; poll GET /admin/simulations/{id} for progress."""

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app import llm, repository
from app.auth import current_superuser
from app.db import SessionFactory, get_session
from app.models import User
from app.schemas import SimulationCreate, SimulationDetail, SimulationRead, SimulationReplay
from app.services import simulation as service
from sim.family import FamilySpec
from sim.pool import build_pool
from sim.runner import run_simulation
from sim.world import SCENARIOS, SIM_FAMILY, Scenario

log = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/simulations", tags=["admin", "simulations"])


async def _execute(run_id: uuid.UUID) -> None:
    """Build the pool if needed, then run. Own session: this outlives the request."""
    async with SessionFactory() as session:
        run = await repository.get_simulation_run(session, run_id)
        try:
            if run.status == "pool_pending":
                profile = await repository.get_family_profile(session, run.household_id)
                await build_pool(
                    session, run, FamilySpec.model_validate(profile.spec), profile.random_seed
                )
            await run_simulation(session, run)
        except Exception as exc:  # recorded on the run; never crash the worker
            log.exception("Simulation %s failed", run_id)
            await session.rollback()
            run = await repository.get_simulation_run(session, run_id)
            run.status, run.error = "failed", f"{type(exc).__name__}: {exc}"[:2000]
            await session.commit()


def pool_complete(run) -> bool:
    """A run's pool is complete once the run got past building it (weeks started), even
    if the run itself failed later."""
    return run.status in ("pool_ready", "running", "completed") or (
        run.status == "failed" and run.current_week > 0
    )


@router.post(
    "/{run_id}/resume", response_model=SimulationRead, status_code=status.HTTP_202_ACCEPTED
)
async def resume_run(
    run_id: uuid.UUID,
    background: BackgroundTasks,
    _: User = Depends(current_superuser),
    session: AsyncSession = Depends(get_session),
):
    """Continue a run that stopped while building its pool (the pool is resumable: weeks
    already written are kept and the RNG replays identically). A run that stopped after
    its weeks began can't be resumed safely mid-week: replay its pool instead."""
    run = await repository.get_simulation_run(session, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Simulation not found")
    if (
        run.pool_run_id is not None
        or run.current_week > 0
        or run.status not in ("failed", "pool_pending")
    ):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Only runs stopped while building their pool can resume"
        )
    if not llm.is_configured():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "GEMINI_API_KEY is not configured")
    run.status, run.error = "pool_pending", None
    await session.commit()
    background.add_task(_execute, run.id)
    return run


@router.get("/scenarios", response_model=dict[str, Scenario])
async def scenarios(_: User = Depends(current_superuser)):
    return SCENARIOS


@router.get("", response_model=list[SimulationRead])
async def list_runs(
    _: User = Depends(current_superuser), session: AsyncSession = Depends(get_session)
):
    return await repository.list_simulation_runs(session)


@router.get("/{run_id}", response_model=SimulationDetail)
async def get_run(
    run_id: uuid.UUID,
    _: User = Depends(current_superuser),
    session: AsyncSession = Depends(get_session),
):
    run = await repository.get_simulation_run(session, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Simulation not found")
    return run


@router.post("", response_model=SimulationRead, status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    body: SimulationCreate,
    background: BackgroundTasks,
    _: User = Depends(current_superuser),
    session: AsyncSession = Depends(get_session),
):
    if body.scenario not in SCENARIOS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"Unknown scenario {body.scenario!r}"
        )
    if not llm.is_configured():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "GEMINI_API_KEY is not configured")
    run = await service.create_run(
        session,
        SIM_FAMILY,
        SCENARIOS[body.scenario],
        body.planner,
        body.start_date,
        body.weeks,
        body.seed,
        body.family_seed,
    )
    background.add_task(_execute, run.id)
    return run


@router.post(
    "/{run_id}/replay", response_model=SimulationRead, status_code=status.HTTP_202_ACCEPTED
)
async def replay_run(
    run_id: uuid.UUID,
    body: SimulationReplay,
    background: BackgroundTasks,
    _: User = Depends(current_superuser),
    session: AsyncSession = Depends(get_session),
):
    """Same world (pool, family, calendar), new behaviour and/or the current models."""
    pool_run = await repository.get_simulation_run(session, run_id)
    if pool_run is None or not pool_complete(pool_run):
        raise HTTPException(status.HTTP_409_CONFLICT, "That run has no finished pool to replay")
    source = (
        pool_run
        if pool_run.pool_run_id is None
        else await repository.get_simulation_run(session, pool_run.pool_run_id)
    )
    run = await service.create_run(
        session, SIM_FAMILY, None, body.planner, None, 0, body.seed, pool_run=source
    )
    background.add_task(_execute, run.id)
    return run
