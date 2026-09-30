"""Drift detection and performance monitoring: pure functions over a stream of
predictions (no database). See docs/learning/05-monitoring-and-drift-detection.md.

A *reference profile* describes how the data looked when the model was validated
(its predictions on the frozen holdout, or the first weeks of a stream). Every week a
rolling window of recent tickets is compared with it:

  label-free (immediate): PSI + chi-square on the predicted category mix, PSI on the
                          confidence and text-length distributions
  labelled (delayed):     category accuracy vs planner labels, effort accuracy vs the
                          sprint review, each with a Wilson interval
"""

import math
from collections import Counter
from datetime import date, timedelta
from typing import Any

import numpy as np
from scipy import stats

CATEGORIES = ["chores", "groceries", "kids", "home_maintenance", "finance", "social", "other"]
EFFORTS = ["S", "M", "L"]
CONF_EDGES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
CONF_BINS = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9+"]
LENGTH_BINS = ["1", "2", "3", "4-5", "6-8", "9-12", "13+"]

PSI_ALARM = 0.25  # "significant shift" rule of thumb
P_ALARM = 0.01
MIN_WINDOW = 30  # tickets in a window before label-free tests may alarm
MIN_LABELLED = 15  # labelled tickets before accuracy may alarm
# Segment tests (effort accuracy per predicted category): window vs all earlier weeks.
MIN_SEGMENT_WINDOW, MIN_SEGMENT_PRIOR = 10, 20


# --- Binning and basic statistics ------------------------------------------------------------


def length_bin(text: str) -> str:
    n = len(str(text).split())
    for label, upper in zip(LENGTH_BINS, [1, 2, 3, 5, 8, 12, math.inf], strict=True):
        if n <= upper:
            return label
    return LENGTH_BINS[-1]


def conf_bin(confidence: float) -> str:
    return CONF_BINS[int(np.digitize(confidence, CONF_EDGES[1:-1]))]


def distribution(values: list[str], bins: list[str]) -> dict[str, float]:
    counts = Counter(values)
    total = sum(counts[b] for b in bins) or 1
    return {b: counts[b] / total for b in bins}


def psi(reference: dict[str, float], current: dict[str, float], eps: float = 1e-3) -> float:
    """Population Stability Index. Epsilon keeps empty bins finite (and makes PSI on tiny
    windows biased upward: one reason for a minimum window size)."""
    total = 0.0
    for b in reference:
        r, c = max(reference[b], eps), max(current.get(b, 0.0), eps)
        total += (c - r) * math.log(c / r)
    return float(total)


def smooth(reference: dict[str, float], eps: float = 0.005) -> dict[str, float]:
    """Give every bin a small minimum share (then renormalise). Without it, a category
    the reference never saw would have an expected count of 0, and the chi-square test
    would be undefined at exactly the moment it should fire loudest."""
    raw = {b: max(p, eps) for b, p in reference.items()}
    total = sum(raw.values())
    return {b: p / total for b, p in raw.items()}


def chi_square(
    reference: dict[str, float], counts: dict[str, int], min_expected: float = 5.0
) -> float | None:
    """Goodness-of-fit p-value of observed counts vs (smoothed) reference proportions.
    Bins with an expected count below `min_expected` are merged into one 'rest' bin,
    because the test is unreliable for small expected counts. None when fewer than two
    bins remain (e.g. an empty window)."""
    n = sum(counts.values())
    if n == 0:
        return None
    expected = {b: p * n for b, p in smooth(reference).items()}
    keep = [b for b in expected if expected[b] >= min_expected]
    rest = [b for b in expected if b not in keep]
    obs = [counts.get(b, 0) for b in keep]
    exp = [expected[b] for b in keep]
    if rest:
        obs.append(sum(counts.get(b, 0) for b in rest))
        exp.append(sum(expected[b] for b in rest))
    if len(obs) < 2:
        return None
    return float(stats.chisquare(obs, f_exp=exp).pvalue)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson interval for a proportion: sane for small n and values near 0 or 1."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def worse_than_before(k_prior: int, n_prior: int, k_now: int, n_now: int) -> float | None:
    """One-sided two-proportion z-test: p-value for 'the current rate is lower than the
    earlier rate'. None when either side has no data."""
    if n_prior == 0 or n_now == 0:
        return None
    p1, p2 = k_prior / n_prior, k_now / n_now
    pooled = (k_prior + k_now) / (n_prior + n_now)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_prior + 1 / n_now))
    if se == 0:
        return None if p2 >= p1 else 0.0
    return float(stats.norm.cdf((p2 - p1) / se))


# --- Reference profile -------------------------------------------------------------------------


def profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Distributions + accuracies of a set of rows (holdout predictions, or a stream's
    first weeks). Row keys: text, pred_category, category_conf, pred_effort, effort_conf,
    and optionally label_category / label_effort (for reference accuracy)."""
    cat_rows = [r for r in rows if r.get("pred_category")]
    eff_rows = [r for r in rows if r.get("pred_effort")]
    lab_c = [r for r in cat_rows if r.get("label_category")]
    lab_e = [r for r in eff_rows if r.get("label_effort")]
    return {
        "n": len(rows),
        "category_mix": distribution([r["pred_category"] for r in cat_rows], CATEGORIES),
        "confidence": distribution([conf_bin(r["category_conf"]) for r in cat_rows], CONF_BINS),
        "effort_confidence": distribution(
            [conf_bin(r["effort_conf"]) for r in eff_rows], CONF_BINS
        ),
        "text_length": distribution([length_bin(r["text"]) for r in rows], LENGTH_BINS),
        "category_accuracy": (
            sum(r["pred_category"] == r["label_category"] for r in lab_c) / len(lab_c)
            if lab_c
            else None
        ),
        "effort_accuracy": sum(r["pred_effort"] == r["label_effort"] for r in lab_e) / len(lab_e)
        if lab_e
        else None,
    }


# --- Rolling monitoring ---------------------------------------------------------------


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _accuracy(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    k, n = sum(a == b for a, b in pairs), len(pairs)
    interval = wilson(k, n)
    return {"n": n, "value": k / n if n else None, "ci95": interval}


def monitor(
    rows: list[dict[str, Any]], reference: dict[str, Any], window_weeks: int = 4
) -> list[dict[str, Any]]:
    """One entry per calendar week, computed over the window ending that week.

    Row keys (besides the profile keys): week (a Monday), and optionally
      label_category        planner's category label (delayed, may be rubber-stamped)
      actual_effort         effort from the sprint review (delayed further)
      truth_category/_effort  hidden truth (simulations only)
      category_changed      planner changed the predicted category
    """
    if not rows:
        return []
    weeks = sorted({r["week"] for r in rows})
    first, last = weeks[0], weeks[-1]
    out = []
    current = first
    while current <= last:
        lo = current - timedelta(weeks=window_weeks - 1)
        window = [r for r in rows if lo <= r["week"] <= current]
        this_week = [r for r in rows if r["week"] == current]
        prior = [r for r in rows if r["week"] < lo]
        out.append(_window_stats(current, window, this_week, reference, prior))
        current += timedelta(weeks=1)
    return out


def _effort_segments(window: list[dict], prior: list[dict]) -> dict[str, dict[str, Any]]:
    """Effort accuracy per predicted category, now vs all earlier weeks. A change that hits
    one category (e.g. groceries take longer) is diluted in the overall average but
    obvious within its segment."""
    out = {}
    for cat in CATEGORIES:
        now = [
            r["pred_effort"] == r["actual_effort"]
            for r in window
            if r.get("pred_category") == cat and r.get("actual_effort")
        ]
        before = [
            r["pred_effort"] == r["actual_effort"]
            for r in prior
            if r.get("pred_category") == cat and r.get("actual_effort")
        ]
        if not now:
            continue
        out[cat] = {
            "n": len(now),
            "value": sum(now) / len(now),
            "prior_n": len(before),
            "prior_value": sum(before) / len(before) if before else None,
            "p_worse": worse_than_before(sum(before), len(before), sum(now), len(now)),
        }
    return out


def _window_stats(
    week: date,
    window: list[dict],
    this_week: list[dict],
    ref: dict,
    prior: list[dict] | None = None,
) -> dict[str, Any]:
    n = len(window)
    cat = [r for r in window if r.get("pred_category")]
    mix_counts = Counter(r["pred_category"] for r in cat)
    mix = distribution([r["pred_category"] for r in cat], CATEGORIES)
    conf = distribution([conf_bin(r["category_conf"]) for r in cat], CONF_BINS)
    length = distribution([length_bin(r["text"]) for r in window], LENGTH_BINS)
    labelled = [(r["pred_category"], r["label_category"]) for r in cat if r.get("label_category")]
    reviewed = [
        (r["pred_effort"], r["actual_effort"])
        for r in window
        if r.get("actual_effort") and r.get("pred_effort")
    ]
    changed = [r["category_changed"] for r in window if r.get("category_changed") is not None]
    entry: dict[str, Any] = {
        "week": week.isoformat(),
        "n_week": len(this_week),
        "n_window": n,
        "mean_confidence": float(np.mean([r["category_conf"] for r in cat])) if cat else None,
        "flagged_rate": (
            float(np.mean([min(r["category_conf"], r.get("effort_conf") or 1) < 0.8 for r in cat]))
            if cat
            else None
        ),
        "psi_category": psi(ref["category_mix"], mix) if cat else None,
        "p_category": chi_square(ref["category_mix"], dict(mix_counts)),
        "psi_confidence": psi(ref["confidence"], conf) if cat else None,
        "p_confidence": chi_square(
            ref["confidence"], Counter(conf_bin(r["category_conf"]) for r in cat)
        ),
        "psi_length": psi(ref["text_length"], length) if window else None,
        "p_length": chi_square(ref["text_length"], Counter(length_bin(r["text"]) for r in window)),
        "category_mix": {k: round(v, 3) for k, v in mix.items()},
        "category_accuracy": _accuracy(labelled),
        "effort_accuracy": _accuracy(reviewed),
        "correction_rate": float(np.mean(changed)) if changed else None,
        "effort_by_category": _effort_segments(window, prior or []),
    }
    if any(r.get("truth_category") for r in window):
        entry["true_category_accuracy"] = _accuracy(
            [(r["pred_category"], r["truth_category"]) for r in cat]
        )
        entry["true_effort_accuracy"] = _accuracy(
            [(r["pred_effort"], r["truth_effort"]) for r in window if r.get("pred_effort")]
        )
    entry["alarms"] = _alarms(entry, ref)
    return entry


def _alarms(e: dict[str, Any], ref: dict[str, Any]) -> list[str]:
    alarms = []
    if e["n_window"] >= MIN_WINDOW:
        # A label-free alarm needs a big shift (PSI) AND one that is unlikely to be chance
        # (chi-square). PSI alone fires on noise in windows of ~60 tickets.
        for name, psi_key, p_key in (
            ("category_mix", "psi_category", "p_category"),
            ("confidence", "psi_confidence", "p_confidence"),
            ("text_length", "psi_length", "p_length"),
        ):
            p_value = e[p_key]
            if (e[psi_key] or 0) > PSI_ALARM and p_value is not None and p_value < P_ALARM:
                alarms.append(name)
    for key, ref_key in (
        ("category_accuracy", "category_accuracy"),
        ("effort_accuracy", "effort_accuracy"),
    ):
        acc, target = e[key], ref.get(ref_key)
        # The whole interval below the reference = significantly worse. The tolerance
        # avoids a float artefact: Wilson's upper bound for 60/60 is 0.9999999..., not 1.0.
        below = acc["ci95"] is not None and acc["ci95"][1] < (target or 0) - 1e-9
        if target is not None and acc["n"] >= MIN_LABELLED and below:
            alarms.append(key)
    # Segments: 7 tests every week, so Bonferroni: each uses P_ALARM / number of segments.
    segments = e.get("effort_by_category", {})
    for cat, seg in segments.items():
        enough = seg["n"] >= MIN_SEGMENT_WINDOW and seg["prior_n"] >= MIN_SEGMENT_PRIOR
        if enough and seg["p_worse"] is not None and seg["p_worse"] < P_ALARM / len(CATEGORIES):
            alarms.append(f"effort_accuracy:{cat}")
    return alarms


def onsets(weekly: list[dict]) -> dict[date, list[str]]:
    """Alarm types that START in a week (absent the week before). A detector that keeps
    firing is one condition, not a new detection each week."""
    out, previous = {}, set()
    for w in weekly:
        current = set(w["alarms"])
        new = sorted(current - previous)
        if new:
            out[date.fromisoformat(w["week"])] = new
        previous = current
    return out


def detection_summary(weekly: list[dict], events: list[dict], start: date) -> dict[str, Any]:
    """Per planted event: the first alarm ONSET between this event and the next one, and
    its delay in weeks. Onsets before the first event are false alarms (or seasons).
    Alarms still on from before an event don't count as detecting it."""
    starts = onsets(weekly)
    event_weeks = sorted((week_start(start) + timedelta(weeks=e["week"]), e) for e in events)
    first_event = event_weeks[0][0] if event_weeks else None
    false_alarms = [
        {"week": w.isoformat(), "alarms": a}
        for w, a in sorted(starts.items())
        if first_event is None or w < first_event
    ]
    detections = []
    for i, (ew, event) in enumerate(event_weeks):
        until = event_weeks[i + 1][0] if i + 1 < len(event_weeks) else None
        hits = [
            (w, a) for w, a in sorted(starts.items()) if w >= ew and (until is None or w < until)
        ]
        detections.append(
            {
                "event": f"{event['kind']} (week {event['week'] + 1})",
                "note": event.get("note", ""),
                "first_alarm_week": hits[0][0].isoformat() if hits else None,
                "delay_weeks": (hits[0][0] - ew).days // 7 if hits else None,
                "detectors": [f"{x} (+{(w - ew).days // 7} wk)" for w, a in hits for x in a],
            }
        )
    return {"false_alarms": false_alarms, "detections": detections}
