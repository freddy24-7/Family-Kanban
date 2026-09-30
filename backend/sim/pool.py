"""The ticket pool: Gemini writes each simulated week's tickets for a truth the world
model decided first. Built once per world (the only step that calls the LLM), then
replayed by any number of runs. Resumable: weeks already in the pool are skipped, and
the RNG is advanced identically, so a resumed pool equals an uninterrupted one."""

import random
from datetime import UTC, datetime, time

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SimTicket, SimulationRun
from sim.family import FamilySpec, build_world
from sim.generator import TicketRequest, generate_batch
from sim.world import Scenario, plan_week, state_at

POOL_PROMPT_VERSION = "sim-v1"


async def build_pool(
    session: AsyncSession, run: SimulationRun, spec: FamilySpec, family_seed: int
) -> SimulationRun:
    scenario = Scenario.model_validate(run.scenario)
    rng = np.random.default_rng(run.random_seed)
    done_weeks = set(
        (
            await session.scalars(
                select(SimTicket.week).where(SimTicket.simulation_run_id == run.id).distinct()
            )
        ).all()
    )
    seen: set[str] = set()
    for week in range(run.weeks):
        state = state_at(spec, scenario, run.start_date, week)
        world = build_world(state.spec, random.Random(family_seed))
        plans = plan_week(state, world, rng)  # always drawn: keeps the RNG stream identical
        if week in done_weeks or not plans:
            continue
        requests = [
            TicketRequest(
                index=i,
                category=p.category,
                submitter=p.submitter,
                style=p.style,
                effort=p.text_effort,
            )
            for i, p in enumerate(plans)
        ]
        now = datetime.combine(state.monday, time(9), UTC)
        batch = await generate_batch(world, requests, now, seen)
        for ticket in batch.accepted:
            plan = plans[ticket.index]
            session.add(
                SimTicket(
                    simulation_run_id=run.id,
                    week=week,
                    text=ticket.text,
                    submitter=plan.submitter.name,
                    style=plan.style,
                    true_category=plan.category,
                    true_effort=plan.true_effort,
                    text_effort=plan.text_effort,
                    gemini_category=ticket.category,
                    gemini_effort=ticket.effort,
                    events=list(plan.events),
                )
            )
        run.tokens_in += batch.tokens_in
        run.tokens_out += batch.tokens_out
        await session.commit()
    run.status = "pool_ready"
    await session.commit()
    return run


async def pool_size(session: AsyncSession, run_id) -> int:
    stmt = select(func.count()).select_from(SimTicket).where(SimTicket.simulation_run_id == run_id)
    return await session.scalar(stmt) or 0
