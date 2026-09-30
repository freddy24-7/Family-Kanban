"""Text report of a simulation run: the weekly statistics next to the planted events."""

import numpy as np

from app.models import SimulationRun

COLUMNS = [
    ("n", "n", "{:>3}"),
    ("true_category_acc", "cat acc", "{:>7.2f}"),
    ("measured_category_acc", "cat acc*", "{:>8.2f}"),
    ("true_effort_acc", "eff acc", "{:>7.2f}"),
    ("mean_category_conf", "cat conf", "{:>8.2f}"),
    ("mean_effort_conf", "eff conf", "{:>8.2f}"),
    ("flagged_rate", "flagged", "{:>7.2f}"),
    ("mean_words", "words", "{:>5.1f}"),
]


def _fmt(value, pattern: str) -> str:
    width = int("".join(ch for ch in pattern if ch.isdigit()).split(".")[0][:1] or 5)
    return "-".rjust(width) if value is None else pattern.format(value)


def weekly_table(run: SimulationRun) -> str:
    header = "week  monday      " + " ".join(
        label.rjust(len(fmt.format(0)) if "f" in fmt else 3) for _, label, fmt in COLUMNS
    )
    lines = [header, "-" * len(header)]
    for s in run.weekly_stats:
        cells = " ".join(_fmt(s.get(key), fmt) for key, _, fmt in COLUMNS)
        mark = f"   <- {'; '.join(s['events'])}" if s.get("events") else ""
        lines.append(f"{s['week'] + 1:>4}  {s['monday']}  {cells}{mark}")
    lines.append(
        "\ncat acc = vs hidden truth; cat acc* = vs the planner's labels (what you'd measure)"
    )
    return "\n".join(lines)


def _mean(rows: list[dict], key: str) -> float:
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else float("nan")


def segments(run: SimulationRun) -> str:
    """Mean metrics between planted events: the before/after the drift detectors must see."""
    events = sorted({e["week"] for e in run.scenario.get("events", [])})
    bounds = [0, *events, run.weeks]
    keys = [
        ("true_category_acc", 8),
        ("measured_category_acc", 9),
        ("true_effort_acc", 8),
        ("effort_mae", 8),
        ("mean_category_conf", 9),
        ("flagged_rate", 8),
    ]
    out = ["weeks     cat acc  cat acc*  eff acc  eff mae  cat conf  flagged  words"]
    for lo, hi in zip(bounds, bounds[1:], strict=False):
        rows = [s for s in run.weekly_stats if lo <= s["week"] < hi]
        if rows:
            cells = " ".join(f"{_mean(rows, k):>{w}.2f}" for k, w in keys)
            out.append(f"{lo + 1:>2}-{hi:<5} {cells} {_mean(rows, 'mean_words'):>6.1f}")
    return "\n".join(out)
