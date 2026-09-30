"""Drift detectors on synthetic streams where we know exactly what changed."""

import random
from datetime import date, timedelta

import pytest

from ml.monitoring import (
    CATEGORIES,
    chi_square,
    conf_bin,
    detection_summary,
    distribution,
    length_bin,
    monitor,
    profile,
    psi,
    wilson,
)

START = date(2026, 7, 6)  # a Monday


def test_binning():
    assert [length_bin(t) for t in ["gras", "gras maaien", "a b c d e", "x " * 20]] == [
        "1",
        "2",
        "4-5",
        "13+",
    ]
    assert [conf_bin(c) for c in [0.3, 0.55, 0.8, 0.95, 1.0]] == [
        "<0.5",
        "0.5-0.6",
        "0.8-0.9",
        "0.9+",
        "0.9+",
    ]


def test_psi():
    ref = {"a": 0.5, "b": 0.5}
    assert psi(ref, {"a": 0.5, "b": 0.5}) == pytest.approx(0.0)
    assert 0.0 < psi(ref, {"a": 0.55, "b": 0.45}) < 0.1  # small move: stable
    assert psi(ref, {"a": 0.9, "b": 0.1}) > 0.25  # big move: significant


def test_chi_square_and_small_bin_merging():
    ref = {"a": 0.5, "b": 0.49, "c": 0.01}
    assert chi_square(ref, {"a": 50, "b": 49, "c": 1}) > 0.5  # as expected
    assert chi_square(ref, {"a": 80, "b": 19, "c": 1}) < 0.001  # clearly shifted
    # 'c' expects 1 ticket (<5): merged with nothing else -> test on remaining bins still works
    assert chi_square(ref, {"a": 5, "b": 5, "c": 0}) is not None
    assert chi_square(ref, {}) is None


def test_wilson():
    low, high = wilson(9, 10)
    assert low < 0.9 < high and high <= 1.0
    assert wilson(0, 0) is None
    narrow, wide = wilson(90, 100), wilson(9, 10)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]  # more data, narrower interval


def _stream(shift_week: int, weeks: int = 16, per_week: int = 20, seed: int = 0) -> list[dict]:
    """Before shift_week: balanced chores/groceries, correct predictions.
    From shift_week: mostly 'social' tickets the model gets wrong (label = chores)."""
    rng = random.Random(seed)
    rows = []
    for w in range(weeks):
        for _ in range(per_week):
            shifted = w >= shift_week
            pred = (
                "social" if shifted and rng.random() < 0.8 else rng.choice(["chores", "groceries"])
            )
            label = "chores" if shifted else pred
            rows.append(
                {
                    "week": START + timedelta(weeks=w),
                    "text": "een twee drie vier",
                    "pred_category": pred,
                    "category_conf": 0.95,
                    "pred_effort": "S",
                    "effort_conf": 0.9,
                    "label_category": label,
                    "label_effort": "S",
                }
            )
    return rows


def test_monitor_detects_planted_shift_without_false_alarms():
    rows = _stream(shift_week=8)
    reference = profile([r for r in rows if r["week"] < START + timedelta(weeks=4)])
    weekly = monitor(rows, reference, window_weeks=4)
    assert len(weekly) == 16
    before = [w for w in weekly if date.fromisoformat(w["week"]) < START + timedelta(weeks=8)]
    assert all(not w["alarms"] for w in before)
    summary = detection_summary(weekly, [{"week": 8, "kind": "category_boost", "note": "x"}], START)
    assert summary["false_alarms"] == []
    hit = summary["detections"][0]
    assert hit["delay_weeks"] is not None and hit["delay_weeks"] <= 2
    fired = " ".join(hit["detectors"])
    assert "category_mix" in fired and "category_accuracy" in fired


def test_small_windows_do_not_alarm():
    rows = _stream(shift_week=2, weeks=4, per_week=3)  # 12 tickets in total
    reference = profile(_stream(shift_week=99, weeks=4))
    assert all(not w["alarms"] for w in monitor(rows, reference))


def test_distribution_sums_to_one():
    d = distribution(["chores", "kids", "kids"], CATEGORIES)
    assert sum(d.values()) == pytest.approx(1.0) and d["kids"] == pytest.approx(2 / 3)


def test_regression_perfect_accuracy_is_not_an_alarm():
    """Wilson's upper bound for 60/60 is 0.9999999..., which once read as 'below 1.0'."""
    rows = _stream(shift_week=99, weeks=4, per_week=15)
    reference = profile(rows)
    assert reference["category_accuracy"] == 1.0
    assert all("category_accuracy" not in w["alarms"] for w in monitor(rows, reference))


def test_regression_unseen_category_makes_chi_square_significant():
    """A category the reference never saw used to give an undefined test (p=None)."""
    reference = {c: 0.0 for c in CATEGORIES} | {"chores": 0.5, "groceries": 0.5}
    p = chi_square(reference, {"chores": 10, "groceries": 10, "social": 40})
    assert p is not None and p < 1e-6


def test_ongoing_alarm_does_not_count_as_detecting_a_later_event():
    from ml.monitoring import detection_summary as summary

    week = lambda i: (START + timedelta(weeks=i)).isoformat()  # noqa: E731
    weekly = [{"week": week(i), "alarms": ["effort_accuracy"] if i >= 1 else []} for i in range(8)]
    weekly[6]["alarms"] = ["effort_accuracy", "category_mix"]
    result = summary(weekly, [{"week": 4, "kind": "x"}], START)
    assert result["false_alarms"] == [{"week": week(1), "alarms": ["effort_accuracy"]}]
    assert result["detections"][0]["delay_weeks"] == 2  # the category_mix onset, not the old alarm


def test_two_proportion_test():
    from ml.monitoring import worse_than_before

    assert worse_than_before(40, 50, 3, 25) < 1e-6  # 80% -> 12%: clearly worse
    assert worse_than_before(40, 50, 20, 25) > 0.3  # 80% -> 80%: no evidence
    assert worse_than_before(0, 0, 3, 5) is None


def test_segment_alarm_catches_drift_diluted_in_the_average():
    """One category's effort estimates break (concept drift) while six others are fine:
    the overall accuracy barely moves, the segment alarm fires."""
    rng = random.Random(1)
    rows = []
    for w in range(16):
        for i in range(40):
            cat = CATEGORIES[i % 7]
            broken = cat == "groceries" and w >= 10
            right = rng.random() < (0.1 if broken else 0.8)
            rows.append(
                {
                    "week": START + timedelta(weeks=w),
                    "text": "a b c",
                    "pred_category": cat,
                    "category_conf": 0.9,
                    "pred_effort": "S",
                    "effort_conf": 0.9,
                    "actual_effort": "S" if right else "M",
                }
            )
    weekly = monitor(rows, profile(rows[:160]), window_weeks=4)
    alarm_weeks = [w["week"] for w in weekly if "effort_accuracy:groceries" in w["alarms"]]
    assert alarm_weeks and min(alarm_weeks) >= (START + timedelta(weeks=10)).isoformat()
    assert all(
        not any(
            a.startswith("effort_accuracy:") and a != "effort_accuracy:groceries"
            for a in w["alarms"]
        )
        for w in weekly
    )
