"""Create preset demo families and fill them with Gemini tickets.

Usage (runs against whatever DATABASE_URL points to):
  uv run python -m sim.seed_dataset --presets typical-1 --tickets 25
  uv run python -m sim.seed_dataset --presets all --tickets 200 --training-eligible --seed 2026
"""

import argparse
import asyncio
import logging
import random

from app import llm
from app.clock import SystemClock
from app.db import SessionFactory, engine
from app.services import demo
from sim.family import PRESETS

PARALLEL_FAMILIES = 4


async def seed_family(
    key: str, tickets: int, training_eligible: bool, seed: int, gate: asyncio.Semaphore
):
    async with gate, SessionFactory() as session:
        household = await demo.create_demo_family(
            session,
            PRESETS[key],
            preset_key=key,
            training_eligible=training_eligible,
            random_seed=seed,
        )
        run = await demo.generate_tickets(
            session, household, tickets, SystemClock(), random_seed=seed + 1
        )
        print(
            f"{key:15} {household.name:32} {run.status:9} produced={run.produced:4} "
            f"invalid={run.rejected_invalid} dup={run.rejected_duplicate} "
            f"relabelled={run.category_mismatches} "
            f"tokens={run.tokens_in}+{run.tokens_out} {run.models_used}"
        )
        return run


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--presets", default="all", help="'all' or comma-separated preset keys")
    parser.add_argument("--tickets", type=int, default=200, help="tickets per family")
    parser.add_argument("--training-eligible", action="store_true")
    parser.add_argument("--seed", type=int, default=2026, help="base seed (names, ticket plans)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if not llm.is_configured():
        raise SystemExit("GEMINI_API_KEY is not set")
    keys = list(PRESETS) if args.presets == "all" else args.presets.split(",")
    unknown = [k for k in keys if k not in PRESETS]
    if unknown:
        raise SystemExit(f"Unknown presets: {unknown}. Known: {list(PRESETS)}")

    rng = random.Random(args.seed)
    gate = asyncio.Semaphore(PARALLEL_FAMILIES)
    runs = await asyncio.gather(
        *(
            seed_family(k, args.tickets, args.training_eligible, rng.randrange(2**31), gate)
            for k in keys
        )
    )
    total_in = sum(r.tokens_in for r in runs)
    total_out = sum(r.tokens_out for r in runs)
    print(f"\n{sum(r.produced for r in runs)} tickets, tokens in={total_in} out={total_out}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
