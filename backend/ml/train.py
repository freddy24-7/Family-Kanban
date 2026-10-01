"""Train, evaluate and register a model version: the reproducible training run.

  uv run python -m ml.train --snapshot seed-v2 --task all --dry-run  # evaluate only
  uv run python -m ml.train --snapshot seed-v2 --task all            # register candidates
  uv run python -m ml.train --snapshot seed-v2 --task all --promote  # + promotion gate

Steps per task:
 1. training data = snapshot minus holdouts (incl. exact/near duplicates of holdout
    texts, checked on raw AND on role-masked text)
 2. role-token masking with the member lists stored IN the snapshot (reproducible;
    the same function runs at serving time)
 3. group cross-validation -> out-of-fold metrics; temperature fitted on OOF probs
 4. final model fitted on all training data, wrapped with the temperature
 5. ONE evaluation per frozen holdout: challenger, champion and baselines
 6. register as candidate; with --promote, activate only if the gate passes

Known limitation: near-duplicate tickets *between training families* can land on both
sides of a CV fold, so CV scores and the CV-fitted temperature are slightly optimistic.
The holdout (with duplicate exclusion) is the number to trust.
"""

import argparse
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from app.domain import Category, Task
from ml import monitoring, registry
from ml.calibration import TemperatureScaled, apply_temperature, fit_temperature
from ml.datasets import dataset_hash, load_snapshot, load_snapshot_members, load_topics_by_ids
from ml.evaluate import (
    expected_calibration_error,
    group_cv_predict,
    macro_f1,
    paired_bootstrap_delta,
    per_class_report,
    summarise,
    threshold_table,
)
from ml.holdout import exclude_similar, holdout_frame, load_manifests, training_frame
from ml.models import (
    EFFORT_ORDER,
    KeywordBaseline,
    TextModelConfig,
    build_text_classifier,
    majority_baseline,
)
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
# Feature types whose runtime dependency isn't installed in production (lesson 07).
EXPERIMENT_ONLY_FEATURES = {"embedding"}


def masked_texts(frame: pd.DataFrame, members: dict[str, list[tuple[str, bool]]]) -> np.ndarray:
    missing = set(frame["household_id"]) - set(members)
    if missing:
        # Silently unmasked training text + masked serving text = training/serving skew.
        raise ValueError(
            f"No member list for {len(missing)} household(s); refusing to mask partially"
        )
    return np.array(
        [
            mask_member_names(t, members[h])
            for t, h in zip(frame["text"], frame["household_id"], strict=True)
        ],
        dtype=object,
    )


def promotion_decision(holdouts: dict[str, dict], primary: str) -> tuple[bool, str]:
    """Gate (paired bootstrap of the primary metric, per frozen holdout):
    - no frozen topics may be missing;
    - on the PRIMARY holdout (today's world) the challenger must be clearly better:
      the 95% interval of the gain lies above zero;
    - on every other holdout it must not be clearly worse: the interval must not lie
      entirely below zero (an old world may legitimately score a new model lower, but not
      significantly so without a deliberate decision);
    - on real-data rows (once they exist) it must not be worse."""
    reasons = []
    for name, h in holdouts.items():
        if h["report"]["missing"]:
            reasons.append(f"{name}: {h['report']['missing']} frozen topics missing")
            continue
        delta = h["delta_vs_champion"]["delta"]
        low, high = h["delta_vs_champion"]["ci95"]
        if name == primary and low <= 0:
            reasons.append(
                f"{name} (primary): {PRIMARY_METRIC} gain {delta:+.3f} "
                f"(95% CI {low:+.3f}..{high:+.3f}) is not clearly above zero"
            )
        elif name != primary and high < 0:
            reasons.append(f"{name}: clearly worse ({delta:+.3f}, 95% CI {low:+.3f}..{high:+.3f})")
        real = h.get("delta_vs_champion_real")
        if real is not None and real["delta"] < 0:
            reasons.append(f"{name}: worse than champion on real data ({real['delta']:+.3f})")
    if primary not in holdouts:
        reasons.append(f"primary holdout {primary!r} not found")
    if reasons:
        return False, "; ".join(reasons)
    return True, f"clearly better on {primary}, not clearly worse elsewhere"


def frozen_holdout(df: pd.DataFrame, manifest, manifests) -> tuple[pd.DataFrame, dict]:
    """A frozen holdout's rows with its FROZEN labels. Rows come from the snapshot when
    present, else from the database by id (holdouts may live in households that are not
    training-eligible, e.g. a simulated world)."""
    held, report = holdout_frame(df, manifest.name, manifests)
    missing_ids = sorted(manifest.topic_ids - set(held["topic_id"]))
    if missing_ids:
        extra = load_topics_by_ids(missing_ids)
        extra["category"] = extra["topic_id"].map(lambda t: manifest.labels[t][0])
        extra["effort"] = extra["topic_id"].map(lambda t: manifest.labels[t][1])
        held = pd.concat([held, extra], ignore_index=True)
        report = {**report, "found": len(held), "missing": len(manifest.topic_ids) - len(held)}
    return held, report


def task_profile(task: Task, texts: list[str], outputs, labels) -> dict:
    """The monitoring reference for one task (see ml/monitoring.py)."""
    if task == Task.CATEGORY:
        rows = [
            {
                "text": t,
                "pred_category": o.predicted,
                "category_conf": o.confidence,
                "label_category": y,
            }
            for t, o, y in zip(texts, outputs, labels, strict=True)
        ]
        prof = monitoring.profile(rows)
        keep = ("n", "category_mix", "confidence", "text_length", "category_accuracy")
        return {k: prof[k] for k in keep}
    return {
        "n": len(texts),
        "effort_confidence": monitoring.distribution(
            [monitoring.conf_bin(o.confidence) for o in outputs], monitoring.CONF_BINS
        ),
        "effort_accuracy": float(
            np.mean([o.predicted == y for o, y in zip(outputs, labels, strict=True)])
        ),
    }


def backfill_reference(df: pd.DataFrame, members) -> None:
    """Add reference profiles to the ACTIVE model versions (trained before Phase 6).
    Only derived monitoring data is added; the model and its evaluation are untouched."""
    session = registry.sync_session()
    for task in (Task.CATEGORY, Task.EFFORT):
        version = registry.active_version(session, task)
        if version is None:
            continue
        model = registry.load_model_sync(session, version)
        profiles = {}
        manifests = load_manifests()
        all_members = {
            **members,
            **{h: ms for m in manifests for h, ms in (m.members or {}).items()},
        }
        for manifest in manifests:
            held, _ = frozen_holdout(df, manifest, manifests)
            outputs = model.predict(list(masked_texts(held, all_members)))
            profiles[manifest.name] = task_profile(
                task, list(held["text"]), outputs, held[LABEL_COLUMN[task]].to_numpy()
            )
        version.metrics = {**version.metrics, "reference_profile": profiles}
        session.commit()
        print(f"{version.name}: reference profile stored for {list(profiles)}")
    session.close()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _champion_predictions(champion, held: pd.DataFrame, members) -> np.ndarray:
    if champion.preprocessing is None:
        texts = list(held["text"])
    elif champion.preprocessing == ROLE_MASK_VERSION:
        texts = list(masked_texts(held, members))
    else:
        raise ValueError(f"Champion needs unknown preprocessing {champion.preprocessing!r}")
    return np.array([o.predicted for o in champion.predict(texts)])


def train_task(
    task: Task,
    df: pd.DataFrame,
    snapshot: str,
    members,
    promote: bool,
    dry_run: bool = False,
    real_weight: float = 1.0,
    primary: str | None = None,
    review_weight: float = 1.0,
    overrides: dict | None = None,
    cv_only: bool = False,
) -> dict:
    # `overrides` (e.g. {"features": "embedding", "C": 10}) is for experiments against
    # the same data, holdouts and gate as the production config.
    config, label = replace(CONFIGS[task], **(overrides or {})), LABEL_COLUMN[task]
    if config.features in EXPERIMENT_ONLY_FEATURES and not (dry_run or cv_only):
        # fastembed is a dev dependency: such a model would load in production and then
        # fail on its first prediction. Experiments are evaluated, never registered.
        raise SystemExit(f"features={config.features!r} is experiment-only: use --dry-run")
    ordinal = task == Task.EFFORT
    manifests = load_manifests()
    primary = primary or latest_holdout(manifests)
    members = {**members, **{h: ms for m in manifests for h, ms in (m.members or {}).items()}}
    held_frames = {m.name: frozen_holdout(df, m, manifests) for m in manifests}
    holdout_texts = pd.concat([h["text"] for h, _ in held_frames.values()]) if held_frames else None

    # 1-2. Training frame (raw-text exclusion), masking, then exclusion on masked text
    train, train_report = training_frame(df, manifests, holdout_texts=holdout_texts)
    X = masked_texts(train, members)
    holdout_masked = np.concatenate([masked_texts(h, members) for h, _ in held_frames.values()])
    keep = ~exclude_similar(X, holdout_masked)
    train_report["excluded_similar_after_masking"] = int((~keep).sum())
    train, X = train[keep], X[keep]
    train_report["rows_out"] = len(train)
    train_report["by_source"] = train["source"].value_counts().to_dict()
    if "effort_source" in train:
        train_report["effort_source"] = train["effort_source"].value_counts().to_dict()
    train_report["real_weight"] = real_weight
    y, groups = train[label].to_numpy(), train["household_id"].to_numpy()
    # Real tickets can count more than synthetic ones, so they aren't drowned out.
    weights = np.where(train["source"].to_numpy() == "real", real_weight, 1.0)
    # Effort only: a sprint-review label is the MEASURED effort; generator and planner
    # labels are estimates. Under concept drift the old estimates outvote the few new
    # measurements unless measurements count more.
    if task == Task.EFFORT and "effort_source" in train:
        weights = weights * np.where(
            train["effort_source"].to_numpy() == "review", review_weight, 1.0
        )
    train_report["review_weight"] = review_weight if task == Task.EFFORT else None

    # 3. Cross-validation (out-of-fold) + temperature
    classes = np.array(EFFORT_ORDER) if ordinal else np.array(sorted(set(y)))
    oof_proba = group_cv_predict(
        build_text_classifier(config),
        X,
        y,
        groups,
        proba=True,
        classes=classes,
        sample_weight=weights,
    )
    temperature = fit_temperature(oof_proba, y, classes)
    scaled = apply_temperature(oof_proba, temperature)
    oof_pred = classes[scaled.argmax(axis=1)]
    correct = oof_pred == y
    raw_correct = classes[oof_proba.argmax(axis=1)] == y
    cv_metrics = {
        **summarise(y, oof_pred, ordinal=ordinal),
        "ece_before": round(expected_calibration_error(oof_proba.max(axis=1), raw_correct), 4),
        "ece_after": round(expected_calibration_error(scaled.max(axis=1), correct), 4),
        "thresholds": threshold_table(scaled.max(axis=1), correct).to_dict(orient="records"),
    }

    if cv_only:
        # Model selection (e.g. choosing C): training families only, no holdout is scored.
        return {"version": "(cv only)", "outcome": "no holdout evaluated", "cv": cv_metrics}

    # 4. Final model on all training data (+ baselines fitted on the same data)
    final = TemperatureScaled(
        build_text_classifier(config).fit(X, y, classifier__sample_weight=weights), temperature
    )
    served = registry.SklearnTextClassifier(final, ROLE_MASK_VERSION)
    baselines = {"majority": majority_baseline().fit(X, y)}
    if task == Task.CATEGORY:
        baselines["keyword rules"] = KeywordBaseline().fit(X, y)

    # 5. One evaluation per frozen holdout: challenger, champion, baselines
    session = registry.sync_session()
    champion_version = registry.active_version(session, task)
    champion = registry.load_model_sync(session, champion_version) if champion_version else None
    holdouts, reference_profiles = {}, {}
    for manifest in manifests:
        held, held_report = held_frames[manifest.name]
        y_held, X_held = held[label].to_numpy(), masked_texts(held, members)
        outputs = served.predict(list(X_held))
        pred = np.array([o.predicted for o in outputs])
        conf = np.array([o.confidence for o in outputs])
        # No active model yet: the challenger must at least beat the majority baseline.
        champ_pred = (
            _champion_predictions(champion, held, members)
            if champion
            else baselines["majority"].predict(X_held)
        )
        delta, low, high = paired_bootstrap_delta(y_held, pred, champ_pred, macro_f1)
        entry = {
            **summarise(y_held, pred, ordinal=ordinal),
            "ece": round(expected_calibration_error(conf, pred == y_held), 4),
            "per_class": per_class_report(y_held, pred, LABELS[task]).to_dict(orient="index"),
            "by_source": {
                s: summarise(y_held[m], pred[m], ordinal=ordinal, ci=False)
                for s, m in _source_masks(held)
            },
            "champion": summarise(y_held, champ_pred, ordinal=ordinal, ci=False),
            "delta_vs_champion": {
                "delta": round(delta, 4),
                "ci95": (round(low, 4), round(high, 4)),
            },
            "baselines": {
                n: summarise(y_held, b.predict(X_held), ordinal=ordinal, ci=False)
                for n, b in baselines.items()
            },
            "report": held_report,
            # Per true category: a change in one category (concept drift) hides in the
            # overall score. (Effort-v2 first looked fine overall while groceries didn't move.)
            "accuracy_by_category": {
                c: {"n": int(m.sum()), "accuracy": round(float((pred[m] == y_held[m]).mean()), 3)}
                for c in sorted(set(held["category"]))
                for m in [(held["category"] == c).to_numpy()]
            },
        }
        real = (held["source"] == "real").to_numpy()
        if real.any():
            entry["delta_vs_champion_real"] = {
                "delta": round(
                    macro_f1(y_held[real], pred[real]) - macro_f1(y_held[real], champ_pred[real]), 4
                )
            }
        holdouts[manifest.name] = entry
        reference_profiles[manifest.name] = task_profile(task, list(held["text"]), outputs, y_held)

    # 6. Register + gate
    passed, reason = promotion_decision(holdouts, primary)
    result = {
        "gate": reason,
        "cv": cv_metrics,
        "holdouts": holdouts,
        "temperature": temperature,
        "champion": _champion_name(champion_version),
    }
    if dry_run:
        session.close()
        would = f"would {'PROMOTE' if passed else 'REJECT'}" if promote else "candidate"
        return {**result, "version": "(dry run, not registered)", "outcome": would}

    metrics = {
        "primary_metric": PRIMARY_METRIC,
        "cv": cv_metrics,
        "holdouts": holdouts,
        "champion_at_training": _champion_name(champion_version),
        "temperature": round(temperature, 4),
        # What "normal" looks like for monitoring (Phase 6): distributions and accuracy
        # of this model's predictions on each frozen holdout.
        "reference_profile": reference_profiles,
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
    return {**result, "version": version.name, "outcome": outcome}


def _champion_name(champion_version) -> str:
    return champion_version.name if champion_version else "majority baseline (no active model)"


def latest_holdout(manifests) -> str:
    """Default primary holdout: the most recently frozen one (today's world)."""
    return max(manifests, key=lambda m: m.raw.get("frozen_at", "")).name


def _source_masks(frame: pd.DataFrame):
    for source in sorted(frame["source"].unique()):
        yield source, (frame["source"] == source).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--task", choices=["category", "effort", "all"], default="all")
    parser.add_argument("--promote", action="store_true", help="activate if the gate passes")
    parser.add_argument("--dry-run", action="store_true", help="evaluate, register nothing")
    parser.add_argument(
        "--backfill-reference",
        action="store_true",
        help="add monitoring references to active models",
    )
    parser.add_argument(
        "--real-weight", type=float, default=1.0, help="weight of real vs synthetic rows"
    )
    parser.add_argument(
        "--review-weight", type=float, default=1.0, help="effort: weight of review labels"
    )
    parser.add_argument("--primary-holdout", help="holdout the challenger must clearly win")
    parser.add_argument(
        "--features",
        choices=["word", "char", "word+char", "embedding"],
        help="experiment: other text features than the production config",
    )
    parser.add_argument("--C", type=float, help="experiment: other regularisation strength")
    parser.add_argument(
        "--cv-only",
        action="store_true",
        help="model selection: group-CV on training families only, no holdout scored",
    )
    args = parser.parse_args()
    overrides = {k: v for k, v in {"features": args.features, "C": args.C}.items() if v}

    df = load_snapshot(args.snapshot)
    members = load_snapshot_members(args.snapshot)
    if args.backfill_reference:
        backfill_reference(df, members)
        return
    tasks = [Task.CATEGORY, Task.EFFORT] if args.task == "all" else [Task(args.task)]
    for task in tasks:
        result = train_task(
            task,
            df,
            args.snapshot,
            members,
            args.promote,
            args.dry_run,
            args.real_weight,
            args.primary_holdout,
            args.review_weight,
            overrides,
            args.cv_only,
        )
        print(f"\n=== {task.value}: {result['version']} -> {result['outcome']}")
        if args.cv_only:
            cv = {k: v for k, v in result["cv"].items() if k != "thresholds"}
            print(f"CV: {json.dumps(cv)}")
            continue
        print(f"gate vs {result['champion']}: {result['gate']}")
        cv = {k: v for k, v in result["cv"].items() if k != "thresholds"}
        print(f"temperature {result['temperature']:.3f} | CV: {json.dumps(cv)}")
        for name, h in result["holdouts"].items():
            keep = ("n", "accuracy", "macro_f1", "macro_f1_ci95", "ordinal_mae", "recall_L", "ece")
            print(f"holdout {name}: {json.dumps({k: h[k] for k in keep if k in h})}")
            print(
                f"  vs champion: {json.dumps(h['delta_vs_champion'])} "
                f"| champion: {json.dumps(h['champion'])}"
            )
            for b, m in h["baselines"].items():
                print(f"  baseline {b}: {json.dumps(m)}")
            print(pd.DataFrame(h["per_class"]).T.to_string())


if __name__ == "__main__":
    main()
