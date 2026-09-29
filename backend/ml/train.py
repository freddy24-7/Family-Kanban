"""Train, evaluate and register a model version: the reproducible training run.

  uv run python -m ml.train --snapshot seed-v1 --task all            # candidates only
  uv run python -m ml.train --snapshot seed-v1 --task all --promote  # + promotion gate

Steps per task:
 1. training data = snapshot minus holdouts (incl. exact/near duplicates)
 2. role-token masking with each household's members (same function as serving)
 3. group cross-validation -> out-of-fold metrics; temperature fitted on OOF probs
 4. final model fitted on all training data, wrapped with the temperature
 5. ONE evaluation per frozen holdout, next to the current champion
 6. register as candidate; with --promote, activate only if the gate passes
"""

import argparse
import json
import subprocess
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from app.domain import Category, Task
from ml import registry
from ml.calibration import TemperatureScaled, apply_temperature, fit_temperature
from ml.datasets import dataset_hash, load_household_members, load_snapshot
from ml.evaluate import (
    expected_calibration_error,
    group_cv_predict,
    per_class_report,
    summarise,
    threshold_table,
)
from ml.holdout import holdout_frame, load_manifests, training_frame
from ml.models import EFFORT_ORDER, TextModelConfig, build_text_classifier
from ml.text import ROLE_MASK_VERSION, mask_member_names

# Chosen in notebooks/02-model-experiments.ipynb (group CV on training families).
CONFIGS: dict[Task, TextModelConfig] = {
    Task.CATEGORY: TextModelConfig(features="word+char", C=1.0, class_weight="balanced"),
    Task.EFFORT: TextModelConfig(
        features="word+char", C=1.0, class_weight="balanced", ordinal=True
    ),
}
LABEL_COLUMN = {Task.CATEGORY: "category", Task.EFFORT: "effort"}
LABELS = {Task.CATEGORY: [c.value for c in Category], Task.EFFORT: EFFORT_ORDER}
PRIMARY_METRIC = "macro_f1"


def masked_texts(frame: pd.DataFrame, members: dict[str, list[tuple[str, bool]]]) -> np.ndarray:
    return np.array(
        [
            mask_member_names(t, members.get(h, []))
            for t, h in zip(frame["text"], frame["household_id"], strict=True)
        ],
        dtype=object,
    )


def promotion_decision(challenger: dict, champion: dict) -> tuple[bool, str]:
    """Gate: the challenger must beat the champion's primary metric on EVERY frozen
    holdout, and must not be worse on any holdout of real data."""
    reasons = []
    for name, c_metrics in challenger.items():
        ch, cp = c_metrics[PRIMARY_METRIC], champion.get(name, {}).get(PRIMARY_METRIC, 0.0)
        if ch <= cp:
            reasons.append(f"{name}: {PRIMARY_METRIC} {ch:.3f} does not beat champion {cp:.3f}")
    if reasons:
        return False, "; ".join(reasons)
    return True, "beats champion on every holdout"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def train_task(
    task: Task, df: pd.DataFrame, snapshot: str, members, promote: bool, dry_run: bool = False
) -> dict:
    config, label = CONFIGS[task], LABEL_COLUMN[task]
    ordinal = task == Task.EFFORT
    train, train_report = training_frame(df)
    X, y, groups = (
        masked_texts(train, members),
        train[label].to_numpy(),
        train["household_id"].to_numpy(),
    )

    # 3. Cross-validation (out-of-fold) + temperature
    classes = np.array(EFFORT_ORDER) if ordinal else np.array(sorted(set(y)))
    oof_proba = group_cv_predict(
        build_text_classifier(config), X, y, groups, proba=True, classes=classes
    )
    temperature = fit_temperature(oof_proba, y, classes)
    scaled = apply_temperature(oof_proba, temperature)
    oof_pred = classes[scaled.argmax(axis=1)]
    correct = oof_pred == y
    cv_metrics = {
        **summarise(y, oof_pred, ordinal=ordinal),
        "ece_before": round(
            expected_calibration_error(
                oof_proba.max(axis=1), classes[oof_proba.argmax(axis=1)] == y
            ),
            4,
        ),
        "ece_after": round(expected_calibration_error(scaled.max(axis=1), correct), 4),
        "thresholds": threshold_table(scaled.max(axis=1), correct).to_dict(orient="records"),
    }

    # 4. Final model on all training data
    final = TemperatureScaled(build_text_classifier(config).fit(X, y), temperature)
    served = registry.SklearnTextClassifier(final, ROLE_MASK_VERSION)

    # 5. Evaluate challenger and champion on every frozen holdout
    session = registry.sync_session()
    champion_version = registry.active_version(session, task)
    champion = registry.load_model_sync(session, champion_version) if champion_version else None
    challenger_holdouts, champion_holdouts, reports = {}, {}, {}
    for manifest in load_manifests():
        held, held_report = holdout_frame(df, manifest.name)
        y_held = held[label].to_numpy()
        outputs = served.predict(list(masked_texts(held, members)))
        pred = np.array([o.predicted for o in outputs])
        conf = np.array([o.confidence for o in outputs])
        challenger_holdouts[manifest.name] = {
            **summarise(y_held, pred, ordinal=ordinal),
            "ece": round(expected_calibration_error(conf, pred == y_held), 4),
            "per_class": per_class_report(y_held, pred, LABELS[task]).to_dict(orient="index"),
            "by_source": {
                s: summarise(y_held[m], pred[m], ordinal=ordinal, ci=False)
                for s, m in _source_masks(held)
            },
            "report": held_report,
        }
        if champion is not None:
            champ_texts = (
                list(held["text"])
                if champion.preprocessing is None
                else list(masked_texts(held, members))
            )
            champ_pred = np.array([o.predicted for o in champion.predict(champ_texts)])
            champion_holdouts[manifest.name] = summarise(
                y_held, champ_pred, ordinal=ordinal, ci=False
            )
        reports[manifest.name] = held_report

    # 6. Register + gate
    metrics = {
        "primary_metric": PRIMARY_METRIC,
        "cv": cv_metrics,
        "holdouts": challenger_holdouts,
        "champion_at_training": {
            "name": champion_version.name if champion_version else None,
            "holdouts": champion_holdouts,
        },
        "temperature": round(temperature, 4),
    }
    meta = {
        "task": task.value,
        "config": config.as_dict(),
        "preprocessing": ROLE_MASK_VERSION,
        "classes": [str(c) for c in final.classes_],
        "snapshot": snapshot,
        "training_frame": train_report,
        "git_commit": _git_commit(),
        "trained_at": datetime.now(UTC).isoformat(),
    }
    passed, reason = promotion_decision(challenger_holdouts, champion_holdouts)
    if dry_run:
        session.close()
        return {
            "version": "(dry run, not registered)",
            "outcome": f"would {'PROMOTE' if passed else 'REJECT'}" if promote else "candidate",
            "gate": reason,
            "cv": cv_metrics,
            "holdouts": challenger_holdouts,
            "champion": champion_holdouts,
            "temperature": temperature,
        }
    version = registry.save_and_register(
        session,
        task,
        final,
        meta,
        metrics,
        training_set_size=len(train),
        training_data_hash=dataset_hash(train),
        parent=champion_version,
        notes=f"Trained on snapshot {snapshot}; config {config.as_dict()}",
    )
    outcome = "candidate (not promoted)"
    if promote and passed:
        registry.promote(session, version)
        outcome = "PROMOTED to active"
    elif promote:
        registry.reject(session, version, reason)
        outcome = "REJECTED"
    session.close()
    return {
        "version": version.name,
        "outcome": outcome,
        "gate": reason,
        "cv": cv_metrics,
        "holdouts": challenger_holdouts,
        "champion": champion_holdouts,
        "temperature": temperature,
    }


def _source_masks(frame: pd.DataFrame):
    for source in sorted(frame["source"].unique()):
        yield source, (frame["source"] == source).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--task", choices=["category", "effort", "all"], default="all")
    parser.add_argument(
        "--promote", action="store_true", help="activate if the promotion gate passes"
    )
    args = parser.parse_args()

    df = load_snapshot(args.snapshot)
    members = load_household_members()
    tasks = [Task.CATEGORY, Task.EFFORT] if args.task == "all" else [Task(args.task)]
    for task in tasks:
        result = train_task(task, df, args.snapshot, members, args.promote, args.dry_run)
        print(f"\n=== {task.value}: {result['version']} -> {result['outcome']} ({result['gate']})")
        print(
            f"temperature {result['temperature']:.3f} | CV: "
            + json.dumps({k: v for k, v in result["cv"].items() if k != "thresholds"})
        )
        for name, m in result["holdouts"].items():
            slim = {k: v for k, v in m.items() if k not in ("per_class", "by_source", "report")}
            print(f"holdout {name}: {json.dumps(slim)}")
            print(f"  champion on {name}: {json.dumps(result['champion'].get(name))}")
            print(pd.DataFrame(m["per_class"]).T.to_string())


if __name__ == "__main__":
    main()
