"""Creating simulation runs: each run lives in its own simulated household (never
training-eligible by default, so experiments can't leak into training data)."""

import secrets
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from app import repository
from app.models import SimulationRun
from app.services import demo
from sim.family import FamilySpec
from sim.world import PlannerBehaviour, Scenario


async def create_run(
    session: AsyncSession,
    spec: FamilySpec,
    scenario: Scenario,
    planner: PlannerBehaviour,
    start: date,
    weeks: int,
    seed: int | None = None,
    family_seed: int | None = None,
    pool_run: SimulationRun | None = None,
) -> SimulationRun:
    """New household + run. With `pool_run`, the run replays that run's pool: same
    family (spec + name seed), same scenario and calendar, own household."""
    if pool_run is not None:
        profile = await repository.get_family_profile(session, pool_run.household_id)
        spec = FamilySpec.model_validate(profile.spec)
        family_seed = profile.random_seed
        scenario = Scenario.model_validate(pool_run.scenario)
        start, weeks = pool_run.start_date, pool_run.weeks
    family_seed = family_seed if family_seed is not None else secrets.randbelow(2**31)
    household = await demo.create_demo_family(
        session,
        spec,
        name=f"Sim {scenario.name} {start.isoformat()}",
        preset_key="simulation",
        training_eligible=False,
        random_seed=family_seed,
    )
    run = SimulationRun(
        household_id=household.id,
        pool_run_id=pool_run.id if pool_run else None,
        scenario=scenario.model_dump(mode="json"),
        planner=planner.model_dump(),
        start_date=start,
        weeks=weeks,
        random_seed=seed if seed is not None else secrets.randbelow(2**31),
        status="pool_ready" if pool_run else "pool_pending",
        weekly_stats=[],
    )
    session.add(run)
    await session.commit()
    return run
