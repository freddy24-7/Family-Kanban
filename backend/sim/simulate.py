"""Run simulations from the command line (writes to whatever DATABASE_URL points at).

# new world: build the pool with Gemini, then run it
  uv run python -m sim.simulate create --scenario drift-demo --weeks 26 \
      --start 2026-07-06 --seed 11 --family-seed 5
# same world, other behaviour or model: replay the pool (no LLM calls)
uv run python -m sim.simulate replay <run_id> --rubber-stamp 0.6
uv run python -m sim.simulate resume <run_id>
uv run python -m sim.simulate report <run_id>
"""

import argparse
import asyncio
import logging
import os
import uuid
from datetime import date

os.environ.setdefault("DB_POOLED", "false")  # long batch job: no pooled connections

from app import llm, repository  # noqa: E402
from app.db import SessionFactory, engine  # noqa: E402
from app.models import SimulationRun  # noqa: E402
from app.services import simulation as service  # noqa: E402
from sim.family import FamilySpec  # noqa: E402
from sim.pool import build_pool, pool_size  # noqa: E402
from sim.report import segments, weekly_table  # noqa: E402
from sim.runner import run_simulation  # noqa: E402
from sim.world import SCENARIOS, SIM_FAMILY, PlannerBehaviour  # noqa: E402


def planner_from(args) -> PlannerBehaviour:
    return PlannerBehaviour(
        check_rate=args.check_rate,
        rubber_stamp_rate=args.rubber_stamp,
        label_error_rate=args.label_error,
    )


async def _continue(session, run: SimulationRun) -> SimulationRun:
    if run.status in ("pool_pending",) or (
        run.status == "failed" and run.pool_run_id is None and run.current_week == 0
    ):
        if not llm.is_configured():
            raise SystemExit("GEMINI_API_KEY is needed to build the pool")
        profile = await repository.get_family_profile(session, run.household_id)
        run.status = "pool_pending"
        await build_pool(session, run, FamilySpec.model_validate(profile.spec), profile.random_seed)
        size = await pool_size(session, run.id)
        print(f"pool ready: {size} tickets, tokens {run.tokens_in}+{run.tokens_out}")
    return await run_simulation(session, run)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "replay"):
        p = sub.add_parser(name)
        if name == "create":
            p.add_argument("--scenario", choices=list(SCENARIOS), default="drift-demo")
            p.add_argument("--weeks", type=int, default=26)
            p.add_argument("--start", type=date.fromisoformat, default=date(2026, 7, 6))
            p.add_argument("--family-seed", type=int)
        else:
            p.add_argument("pool_run_id", type=uuid.UUID)
        p.add_argument("--seed", type=int)
        p.add_argument("--check-rate", type=float, default=0.9)
        p.add_argument("--rubber-stamp", type=float, default=0.2)
        p.add_argument("--label-error", type=float, default=0.03)
        if name == "replay":  # shadow evaluation of candidate models
            p.add_argument("--category-model")
            p.add_argument("--effort-model")
    for name in ("resume", "report"):
        sub.add_parser(name).add_argument("run_id", type=uuid.UUID)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    async with SessionFactory() as session:
        if args.command == "create":
            run = await service.create_run(
                session,
                SIM_FAMILY,
                SCENARIOS[args.scenario],
                planner_from(args),
                args.start,
                args.weeks,
                args.seed,
                args.family_seed,
            )
            print(f"run {run.id} (household {run.household_id})")
            run = await _continue(session, run)
        elif args.command == "replay":
            pool_run = await session.get(SimulationRun, args.pool_run_id)
            overrides = {
                k: v
                for k, v in (("category", args.category_model), ("effort", args.effort_model))
                if v
            }
            run = await service.create_run(
                session,
                SIM_FAMILY,
                None,
                planner_from(args),
                None,
                0,
                args.seed,
                pool_run=pool_run,
                model_versions=overrides or None,
            )
            print(f"replay run {run.id} of pool {pool_run.id}")
            run = await run_simulation(session, run)
        else:
            run = await session.get(SimulationRun, args.run_id)
            if args.command == "resume":
                run = await _continue(session, run)
        progress = f"{run.status}, {run.current_week}/{run.weeks} weeks"
        print(f"\n{run.scenario['name']}: {progress}, planner {run.planner}\n")
        print(weekly_table(run))
        print("\nBetween planted events:\n" + segments(run))
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
