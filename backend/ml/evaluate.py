"""Evaluation: metrics with uncertainty, group cross-validation, calibration.
See docs/learning/02-text-classification-and-evaluation.md."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, cross_val_predict

from ml.models import EFFORT_ORDER


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_report(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=labels).round(3)


def ordinal_mae(y_true, y_pred, order: list[str] = EFFORT_ORDER) -> float:
    """Mean number of steps wrong on the S<M<L scale (S->L counts 2)."""
    rank = {v: i for i, v in enumerate(order)}
    return float(np.mean([abs(rank[t] - rank[p]) for t, p in zip(y_true, y_pred, strict=True)]))


def bootstrap_ci(
    y_true, y_pred, metric: Callable = macro_f1, n: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Resample the test set with replacement and recompute the metric: the spread
    shows how much the score depends on which tickets happened to be in the test set."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(metric(y_true[idx], y_pred[idx]))
    alpha = (1 - level) / 2
    return float(np.quantile(scores, alpha)), float(np.quantile(scores, 1 - alpha))


def paired_bootstrap_delta(
    y_true, pred_new, pred_old, metric: Callable = macro_f1, n: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Difference metric(new) - metric(old) on the SAME resampled rows each time.
    Pairing removes the noise both models share (which tickets happened to be in the
    test set), so it detects real improvements that two separate intervals would miss.
    Returns (delta, ci95_low, ci95_high)."""
    y_true, pred_new, pred_old = np.asarray(y_true), np.asarray(pred_new), np.asarray(pred_old)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        deltas.append(metric(y_true[idx], pred_new[idx]) - metric(y_true[idx], pred_old[idx]))
    delta = metric(y_true, pred_new) - metric(y_true, pred_old)
    return float(delta), float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def summarise(y_true, y_pred, ordinal: bool = False, ci: bool = True) -> dict:
    out = {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(macro_f1(y_true, y_pred), 3),
    }
    if ci:
        low, high = bootstrap_ci(y_true, y_pred)
        out["macro_f1_ci95"] = (round(low, 3), round(high, 3))
    if ordinal:
        out["ordinal_mae"] = round(ordinal_mae(y_true, y_pred), 3)
        _, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=["L"], zero_division=0
        )
        out["recall_L"] = round(float(recall[0]), 3)
    return out


# --- Cross-validation on training families only ------------------------------------------


def group_cv_predict(
    estimator, texts, y, groups, n_splits: int = 3, proba: bool = False, classes=None
):
    """Out-of-fold predictions with whole households per fold (mirrors the holdout
    design). Every ticket is predicted by a model that never saw its family.

    With proba=True the columns follow `classes` (default: sorted labels). We loop
    ourselves instead of cross_val_predict, which silently re-encodes labels as
    integers for probabilities and would break the S < M < L order of the ordinal model."""
    X, y = np.asarray(texts, dtype=object), np.asarray(y)
    cv = GroupKFold(n_splits=n_splits)
    if not proba:
        return cross_val_predict(clone(estimator), X, y, groups=groups, cv=cv)
    classes = list(classes) if classes is not None else sorted(set(y))
    out = np.zeros((len(y), len(classes)))
    for train_idx, test_idx in cv.split(X, y, groups):
        model = clone(estimator).fit(X[train_idx], y[train_idx])
        fold = model.predict_proba(X[test_idx])
        for j, cls in enumerate(model.classes_):
            out[test_idx, classes.index(str(cls))] = fold[:, j]
    return out


# --- Calibration -------------------------------------------------------------------------


def reliability_table(confidence, correct, bins: int = 10) -> pd.DataFrame:
    """Per confidence bin: how confident the model claimed to be vs how often it was right."""
    confidence, correct = np.asarray(confidence), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            rows.append(
                {
                    "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                    "n": int(mask.sum()),
                    "mean_confidence": confidence[mask].mean(),
                    "accuracy": correct[mask].mean(),
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(confidence, correct, bins: int = 10) -> float:
    """Weighted average gap between claimed confidence and actual accuracy (0 = perfect)."""
    table = reliability_table(confidence, correct, bins)
    return float(
        (table["n"] * (table["mean_confidence"] - table["accuracy"]).abs()).sum() / table["n"].sum()
    )


def threshold_table(
    confidence, correct, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
) -> pd.DataFrame:
    """For each 'needs review below X' threshold: how many tickets are auto-accepted
    (coverage) and how accurate those auto-accepted predictions are."""
    confidence, correct = np.asarray(confidence), np.asarray(correct, dtype=float)
    rows = []
    for t in thresholds:
        accepted = confidence >= t
        rows.append(
            {
                "threshold": t,
                "auto_accepted": accepted.mean(),
                "accuracy_of_accepted": correct[accepted].mean() if accepted.any() else np.nan,
                "flagged_for_review": (~accepted).mean(),
            }
        )
    return pd.DataFrame(rows).round(3)
