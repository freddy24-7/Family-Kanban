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
    train, report = training_frame(df, load_manifests(tmp_path), near_duplicate_threshold=None)
    assert set(train["topic_id"]) == {"t4", "t5"}  # t3 duplicates holdout text t1
    assert report == {
        "rows_in": 5,
        "excluded_holdout": 2,
        "excluded_duplicate_of_holdout": 1,
        "excluded_near_duplicate_of_holdout": 0,
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


def test_near_duplicates_of_holdout_are_excluded(tmp_path):
    df = frame(
        [
            ("h1", "a", "A", "p", "zorgverzekering declaratie indienen", "finance", "S"),
            (
                "t1",
                "b",
                "B",
                "q",
                "declaratie zorgverzekering indienen",
                "finance",
                "S",
            ),  # reordered
            ("t2", "b", "B", "q", "boodschappen doen bij de supermarkt", "groceries", "S"),
        ]
    )
    freeze(df, "hv1", ["a"], "test", tmp_path)
    train, report = training_frame(df, load_manifests(tmp_path))
    assert list(train["topic_id"]) == ["t2"]
    assert report["excluded_near_duplicate_of_holdout"] == 1


def test_snapshot_members_roundtrip_and_tamper_check(df, tmp_path):
    from ml.datasets import load_snapshot_members

    members = {"h1": [("Lieke", True)], "h2": [("Jeroen", False)], "other": [("X", False)]}
    save_snapshot(df, "snap", tmp_path, members=members)
    loaded = load_snapshot_members("snap", tmp_path)
    assert loaded == {
        "h1": [("Lieke", True)],
        "h2": [("Jeroen", False)],
    }  # only snapshot households
    meta_path = tmp_path / "snap.json"
    meta_path.write_text(meta_path.read_text().replace("Lieke", "Sem"))
    with pytest.raises(ValueError, match="hash"):
        load_snapshot_members("snap", tmp_path)


def test_exclude_similar_on_masked_text():
    from ml.holdout import exclude_similar

    mask = exclude_similar(["naamkind zwemles", "gras maaien", "melk halen"], ["Naamkind zwemles!"])
    assert list(mask) == [True, False, False]
