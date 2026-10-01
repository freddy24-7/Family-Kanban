"""The retraining loop (lesson 06).

  uv run python -m ml.retrain status
  uv run python -m ml.retrain snapshot --name train-2026-10-02      # freeze today's training data
  uv run python -m ml.retrain run --snapshot train-2026-10-02 --dry-run
  uv run python -m ml.retrain run --snapshot train-2026-10-02 --promote [--real-weight 5]
  uv run python -m ml.retrain rollback --task effort

`run` trains a challenger per task and evaluates it next to the champion on every frozen
holdout; the gate (ml/train.py: promotion_decision) decides. Before promoting in
production, replay an unseen simulated world with the candidate (shadow evaluation):
  uv run python -m sim.simulate replay <run> --category-model category-v2 --effort-model effort-v2
"""

import argparse

import pandas as pd
from sqlalchemy import select

from app.domain import Task
from app.models import ModelVersion
from ml import registry
from ml.datasets import (
    load_household_members,
    load_labelled,
    load_snapshot,
    load_snapshot_members,
    save_snapshot,
)
from ml.train import train_task


def status() -> None:
    session = registry.sync_session()
    versions = session.scalars(select(ModelVersion).order_by(ModelVersion.created_at)).all()
    for v in versions:
        holdouts = (v.metrics or {}).get("holdouts", {})
        scores = ", ".join(f"{h}: {m.get('macro_f1')}" for h, m in holdouts.items())
        print(f"{v.name:18} {str(v.status):9} n={v.training_set_size:<5} {scores}")
    session.close()


def snapshot(name: str) -> None:
    df = load_labelled()
    meta = save_snapshot(df, name, members=load_household_members())
    print(f"snapshot {name}: {meta['rows']} rows, hash {meta['dataset_hash'][:16]}")
    print(f"  by source: {meta['by_source']}")
    print(f"  effort labels from: {df['effort_source'].value_counts().to_dict()}")


def run(
    name: str,
    promote: bool,
    dry_run: bool,
    real_weight: float,
    primary: str | None,
    review_weight: float = 1.0,
) -> None:
    df: pd.DataFrame = load_snapshot(name)
    members = load_snapshot_members(name)
    for task in (Task.CATEGORY, Task.EFFORT):
        result = train_task(
            task, df, name, members, promote, dry_run, real_weight, primary, review_weight
        )
        print(f"\n=== {task.value}: {result['version']} -> {result['outcome']}")
        print(f"gate vs {result['champion']}: {result['gate']}")
        for holdout, h in result["holdouts"].items():
            gain = h["delta_vs_champion"]
            groceries = h.get("accuracy_by_category", {}).get("groceries")
            print(
                f"  {holdout:22} challenger {h['macro_f1']:.3f} {h.get('macro_f1_ci95')} | "
                f"champion {h['champion']['macro_f1']:.3f} | "
                f"gain {gain['delta']:+.3f} {gain['ci95']}"
                + (
                    f" | groceries {groceries['accuracy']:.2f} (n={groceries['n']})"
                    if groceries
                    else ""
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    s = sub.add_parser("snapshot")
    s.add_argument("--name", required=True)
    r = sub.add_parser("run")
    r.add_argument("--snapshot", required=True)
    r.add_argument("--promote", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--real-weight", type=float, default=1.0)
    r.add_argument(
        "--review-weight", type=float, default=1.0, help="effort: weight of sprint-review labels"
    )
    r.add_argument("--primary-holdout")
    b = sub.add_parser("rollback")
    b.add_argument("--task", choices=[t.value for t in Task], required=True)
    args = parser.parse_args()

    if args.command == "status":
        status()
    elif args.command == "snapshot":
        snapshot(args.name)
    elif args.command == "run":
        run(
            args.snapshot,
            args.promote,
            args.dry_run,
            args.real_weight,
            args.primary_holdout,
            args.review_weight,
        )
    else:
        session = registry.sync_session()
        retired, active = registry.rollback(session, Task(args.task))
        print(f"rolled back: {retired.name} retired, {active.name} active again")
        session.close()


if __name__ == "__main__":
    main()
