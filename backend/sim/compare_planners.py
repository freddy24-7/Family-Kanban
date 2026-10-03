"""Compare two planner runs of the same world (Phase 9, lesson 08).

  uv run python -m sim.compare_planners <run A> <run B>

Both runs must replay the same pool with the same seed, settings and models under the
load work model, so they differ only in who planned (rules vs Gemini). Under that model
every random event is keyed by ticket and week, so both planners face the same luck.

Decision rule, fixed before any result was seen: B is better than A only if
  1. B finishes more hours of work per week (paired by week, 95% interval above 0), AND
  2. B's share of planned items that get finished is not clearly lower (interval not
     entirely below 0).
Finished hours alone would reward planning MORE (an overloaded task still sometimes gets
done and failing costs nothing), which is the opposite of planning well.

Also reported: overloaded items, waiting time of finished tasks, the age of the oldest
open task (guards against skipping old tasks), the final backlog, LLM fallbacks and models.
Weeks are not independent (a plan changes next week's backlog), so a moving-block
bootstrap (blocks of 3 weeks) is used; it's still one world, not a law.
"""

import argparse
import asyncio
import uuid

import numpy as np

from app.db import SessionFactory
from app.models import SimulationRun

BLOCK = 3  # weeks per bootstrap block


def _series(stats: list[dict], key: str) -> np.ndarray:
    return np.array([float(w.get(key) or 0) for w in stats])


def summary(run: SimulationRun) -> dict:
    stats = run.weekly_stats
    planned = sum(w["sprint_items"] for w in stats)
    waits = [w["done_wait_days"] for w in stats if w.get("done_wait_days") is not None]
    models = sorted({(w.get("planner") or {}).get("model") for w in stats} - {None})
    return {
        "planner": run.planner.get("policy"),
        "weeks": len(stats),
        "done_hours": round(float(_series(stats, "done_hours").sum()), 1),
        "planned_hours": round(float(_series(stats, "planned_hours").sum()), 1),
        "done_items": int(_series(stats, "sprint_done").sum()),
        "completion_rate": round(sum(w["sprint_done"] for w in stats) / planned, 3)
        if planned
        else None,
        "overloaded_items": int(_series(stats, "overloaded_items").sum()),
        "mean_wait_days": round(float(np.mean(waits)), 1) if waits else None,
        "oldest_open_days_end": stats[-1].get("oldest_open_days") if stats else None,
        "final_backlog": stats[-1].get("backlog_size") if stats else None,
        "llm_fallback_weeks": sum(1 for w in stats if (w.get("planner") or {}).get("fallback")),
        "models": models,
        "tokens": sum((w.get("planner") or {}).get("tokens", 0) for w in stats),
    }


def block_bootstrap_mean(diff: np.ndarray, n: int = 5000, seed: int = 0):
    """Mean of `diff` with a moving-block bootstrap 95% interval (keeps neighbouring
    weeks together, because they influence each other)."""
    rng = np.random.default_rng(seed)
    if len(diff) < BLOCK:  # too short for blocks: plain bootstrap
        means = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
        return (
            float(diff.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        )
    blocks = max(1, int(np.ceil(len(diff) / BLOCK)))
    starts = len(diff) - BLOCK + 1
    means = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in rng.integers(0, starts, blocks)])
        means.append(diff[idx[: len(diff)]].mean())
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def weekly_completion_rate(stats: list[dict]) -> np.ndarray:
    return np.array(
        [w["sprint_done"] / w["sprint_items"] if w["sprint_items"] else 0.0 for w in stats]
    )


def check_comparable(a: SimulationRun, b: SimulationRun) -> None:
    problems = []
    if (a.pool_run_id or a.id) != (b.pool_run_id or b.id):
        problems.append("different pools")
    if a.random_seed != b.random_seed:
        problems.append("different seeds")
    if a.weeks != b.weeks or len(a.weekly_stats) != len(b.weekly_stats):
        problems.append("different numbers of weeks (unfinished run?)")
    if (a.adaptation, a.model_versions) != (b.adaptation, b.model_versions):
        problems.append("different models or adaptation")
    settings = lambda run: {k: v for k, v in run.planner.items() if k != "policy"}  # noqa: E731
    if settings(a) != settings(b):
        problems.append(f"different planner settings: {settings(a)} vs {settings(b)}")
    if a.planner.get("work_model") != "load":
        problems.append("needs --work-model load")
    if problems:
        raise SystemExit("Not comparable: " + "; ".join(problems))


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_a", type=uuid.UUID)
    parser.add_argument("run_b", type=uuid.UUID)
    args = parser.parse_args()
    async with SessionFactory() as session:
        a = await session.get(SimulationRun, args.run_a)
        b = await session.get(SimulationRun, args.run_b)
    if a is None or b is None:
        raise SystemExit("Run not found")
    check_comparable(a, b)
    for name, run in (("A", a), ("B", b)):
        print(f"{name} {str(run.id)[:8]}: {summary(run)}")

    # Weeks where the LLM fell back to rules are partly a rules run: excluded.
    keep = [
        i
        for i, (wa, wb) in enumerate(zip(a.weekly_stats, b.weekly_stats, strict=True))
        if not (wa.get("planner") or {}).get("fallback")
        and not (wb.get("planner") or {}).get("fallback")
    ]
    sa = [a.weekly_stats[i] for i in keep]
    sb = [b.weekly_stats[i] for i in keep]
    hours = block_bootstrap_mean(_series(sb, "done_hours") - _series(sa, "done_hours"))
    rate = block_bootstrap_mean(weekly_completion_rate(sb) - weekly_completion_rate(sa))
    print(f"\n{len(keep)} paired weeks (fallback weeks excluded), B - A per week:")
    print(f"  finished hours   {hours[0]:+.2f}  (95% CI {hours[1]:+.2f}..{hours[2]:+.2f})")
    print(f"  completion rate  {rate[0]:+.3f}  (95% CI {rate[1]:+.3f}..{rate[2]:+.3f})")
    better = hours[1] > 0 and rate[2] >= 0
    worse = hours[2] < 0 and rate[1] <= 0
    verdict = "B better" if better else "A better" if worse else "no clear difference"
    print(f"Decision rule: {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
