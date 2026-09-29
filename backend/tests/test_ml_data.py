"""Dataset hashing and holdout rules (pure pandas, no database)."""

import json

import pandas as pd
import pytest

from ml.datasets import dataset_hash, load_snapshot, save_snapshot
from ml.holdout import freeze, holdout_frame, load_manifests, training_frame
from ml.text import text_hash


def frame(rows):
    df = pd.DataFrame(
        rows,
        columns=[
            "topic_id",
            "household_id",
            "household_name",
            "preset_key",
            "text",
            "category",
            "effort",
        ],
    )
    df["source"] = "simulated"
    df["text_hash"] = df["text"].map(text_hash)
    return df


@pytest.fixture
def df():
    return frame(
        [
            ("t1", "h1", "A", "typical-1", "Gras maaien", "home_maintenance", "M"),
            ("t2", "h1", "A", "typical-1", "Melk halen", "groceries", "S"),
            ("t3", "h2", "B", "typical-2", "gras maaien!!", "home_maintenance", "M"),  # dup of t1
            ("t4", "h2", "B", "typical-2", "Belasting doen", "finance", "L"),
            ("t5", "h3", "C", "typical-3", "Hond uitlaten", "chores", "S"),
        ]
    )


def test_dataset_hash_ignores_order_but_not_content(df):
    assert dataset_hash(df) == dataset_hash(df.iloc[::-1])
    changed = df.copy()
    changed.loc[0, "effort"] = "L"
    assert dataset_hash(changed) != dataset_hash(df)


def test_snapshot_roundtrip_and_no_overwrite(df, tmp_path):
    meta = save_snapshot(df, "snap", tmp_path)
    assert meta["rows"] == 5 and meta["dataset_hash"] == dataset_hash(df)
    assert dataset_hash(load_snapshot("snap", tmp_path)) == meta["dataset_hash"]
    with pytest.raises(FileExistsError):
        save_snapshot(df, "snap", tmp_path)


def test_freeze_is_group_split_and_never_overwrites(df, tmp_path):
    manifest = freeze(df, "hv1", ["h1"], "test", tmp_path)
    assert manifest["counts"]["topics"] == 2
    assert [h["household_id"] for h in manifest["households"]] == ["h1"]
    with pytest.raises(FileExistsError):
        freeze(df, "hv1", ["h2"], "again", tmp_path)


def test_training_excludes_holdout_and_its_duplicates(df, tmp_path):
    freeze(df, "hv1", ["h1"], "test", tmp_path)
    train, report = training_frame(df, load_manifests(tmp_path))
    assert set(train["topic_id"]) == {"t4", "t5"}  # t3 duplicates holdout text t1
    assert report == {
        "rows_in": 5,
        "excluded_holdout": 2,
        "excluded_duplicate_of_holdout": 1,
        "rows_out": 2,
    }
    assert set(train["household_id"]).isdisjoint({"h1"})


def test_holdout_uses_frozen_labels_and_reports_changes(df, tmp_path):
    freeze(df, "hv1", ["h1"], "test", tmp_path)
    later = df.copy()
    later.loc[later["topic_id"] == "t1", "category"] = "chores"  # a human relabelled it
    later = later[later["topic_id"] != "t2"]  # and a topic was deleted
    held, report = holdout_frame(later, "hv1", load_manifests(tmp_path))
    assert report == {"expected": 2, "found": 1, "missing": 1, "labels_changed_since_freeze": 1}
    assert held.set_index("topic_id").loc["t1", "category"] == "home_maintenance"


def test_manifest_is_committable_json(df, tmp_path):
    freeze(df, "hv1", ["h2"], "note", tmp_path)
    raw = json.loads((tmp_path / "hv1.json").read_text())
    assert raw["split"] == "group (whole households)"
    assert {t["topic_id"] for t in raw["topics"]} == {"t3", "t4"}
