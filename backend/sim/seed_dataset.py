"""Create preset demo families and fill them with Gemini tickets.

Resumable: for each preset it reuses the existing training-eligible demo family
(if any) and only generates the tickets still missing to reach --tickets. Re-run
the same command after a failure to top up.

Usage (runs against whatever DATABASE_URL points to):
  uv run python -m sim.seed_dataset --presets typical-1 --tickets 25
  uv run python -m sim.seed_dataset --presets all --tickets 200 --training-eligible --seed 2026
"""

import argparse
import asyncio
import logging
import os
import random

# Long-running batch job: no connection pool (must be set before app.db is imported).
os.environ.setdefault("DB_POOLED", "false")

from app import llm, repository  # noqa: E402
from app.clock import SystemClock  # noqa: E402
from app.db import SessionFactory, engine  # noqa: E402
from app.services import demo  # noqa: E402
from sim.family import PRESETS  # noqa: E402

PARALLEL_FAMILIES = 3


async def seed_family(
    key: str, target: int, training_eligible: bool, seed: int, gate: asyncio.Semaphore
) -> int:
    async with gate, SessionFactory() as session:
        household = (
            await repository.find_demo_household(session, key) if training_eligible else None
        )
        if household is None:
            household = await demo.create_demo_family(
                session,
                PRESETS[key],
                preset_key=key,
                training_eligible=training_eligible,
                random_seed=seed,
            )
        missing = target - await repository.count_topics(session, household.id)
        if missing <= 0:
            print(f"{key:15} {household.name:32} already has {target}+ tickets")
            return 0
        run = await demo.generate_tickets(
            session, household, missing, SystemClock(), random_seed=seed + missing
        )
        print(
            f"{key:15} {household.name:32} {run.status:9} produced={run.produced:4}/{missing} "
            f"invalid={run.rejected_invalid} dup={run.rejected_duplicate} "
            f"relabelled={run.category_mismatches} "
            f"tokens={run.tokens_in}+{run.tokens_out} {run.models_used}"
            + (f" error={run.error}" if run.error else "")
        )
        return run.tokens_in + run.tokens_out


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--presets", default="all", help="'all' or comma-separated preset keys")
    parser.add_argument("--tickets", type=int, default=200, help="target tickets per family")
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
    seeds = {k: rng.randrange(2**31) for k in PRESETS}  # stable per preset across runs
    gate = asyncio.Semaphore(PARALLEL_FAMILIES)
    results = await asyncio.gather(
        *(seed_family(k, args.tickets, args.training_eligible, seeds[k], gate) for k in keys),
        return_exceptions=True,  # one family failing must not cancel the others
    )
    for key, result in zip(keys, results, strict=True):
        if isinstance(result, BaseException):
            print(f"{key:15} CRASHED: {type(result).__name__}: {result}")
    print(f"\ntokens used: {sum(r for r in results if isinstance(r, int))}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
