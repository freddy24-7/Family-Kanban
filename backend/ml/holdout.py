"""Frozen holdout sets.

A holdout is frozen ONCE, before any modelling, and never edited: that is what
makes metrics comparable across model versions. We hold out whole households
(a group split), because the model must work for families it has never seen.

A manifest records, per topic: id, text hash and the labels at freeze time. The
training set excludes every holdout topic id AND every text whose normalised hash
appears in a holdout (so a duplicate in another family can't leak the answer).

CLI:
  uv run python -m ml.holdout suggest
  uv run python -m ml.holdout freeze --name holdout-sim-v1 \
      --presets typical-2,typical-5,city-apartment
"""

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml.datasets import dataset_hash, load_labelled
from ml.text import text_hash

HOLDOUT_DIR = Path(__file__).resolve().parent / "holdouts"
# Cosine similarity (character n-grams) above which a training text counts as a
# near-duplicate of a holdout text. Chosen from the EDA examples (reordered words ~0.9).
NEAR_DUPLICATE_THRESHOLD = 0.9


@dataclass(frozen=True)
class Manifest:
    name: str
    household_ids: frozenset[str]
    topic_ids: frozenset[str]
    text_hashes: frozenset[str]
    labels: dict[str, tuple[str, str]]  # topic_id -> (category, effort) at freeze time
    raw: dict


def manifest_path(name: str, directory: Path = HOLDOUT_DIR) -> Path:
    return directory / f"{name}.json"


def freeze(
    df: pd.DataFrame, name: str, household_ids: list[str], note: str, directory: Path = HOLDOUT_DIR
) -> dict:
    """Write a manifest holding out all labelled topics of the given households."""
    path = manifest_path(name, directory)
    if path.exists():
        raise FileExistsError(f"Holdout {name!r} is frozen; freeze a new version instead")
    unknown = set(household_ids) - set(df["household_id"])
    if unknown:
        raise ValueError(f"Households not in the dataset: {sorted(unknown)}")
    held = df[df["household_id"].isin(household_ids)].sort_values("topic_id")
    manifest = {
        "name": name,
        "frozen_at": datetime.now(UTC).isoformat(),
        "note": note,
        "split": "group (whole households)",
        "dataset_hash_at_freeze": dataset_hash(df),
        "households": [
            {
                "household_id": hid,
                "name": g["household_name"].iloc[0],
                "preset_key": g["preset_key"].iloc[0],
            }
            for hid, g in held.groupby("household_id")
        ],
        "counts": {
            "topics": len(held),
            "by_source": held["source"].value_counts().to_dict(),
            "by_category": held["category"].value_counts().to_dict(),
            "by_effort": held["effort"].value_counts().to_dict(),
        },
        "topics": [
            {
                "topic_id": r.topic_id,
                "text_hash": r.text_hash,
                "category": r.category,
                "effort": r.effort,
            }
            for r in held.itertuples()
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return manifest


def load_manifests(directory: Path = HOLDOUT_DIR) -> list[Manifest]:
    manifests = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text())
        manifests.append(
            Manifest(
                name=raw["name"],
                household_ids=frozenset(h["household_id"] for h in raw["households"]),
                topic_ids=frozenset(t["topic_id"] for t in raw["topics"]),
                text_hashes=frozenset(t["text_hash"] for t in raw["topics"]),
                labels={t["topic_id"]: (t["category"], t["effort"]) for t in raw["topics"]},
                raw=raw,
            )
        )
    return manifests


def training_frame(
    df: pd.DataFrame,
    manifests: list[Manifest] | None = None,
    near_duplicate_threshold: float | None = NEAR_DUPLICATE_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """Everything that may be trained on: drops holdout topics, anything from a
    holdout household, any text duplicating a holdout text, and (by default) any
    text that is a *near*-duplicate of a holdout text (character n-gram cosine
    similarity >= threshold, e.g. the same words in a different order)."""
    manifests = load_manifests() if manifests is None else manifests
    ids = frozenset().union(*(m.topic_ids for m in manifests)) if manifests else frozenset()
    households = (
        frozenset().union(*(m.household_ids for m in manifests)) if manifests else frozenset()
    )
    hashes = frozenset().union(*(m.text_hashes for m in manifests)) if manifests else frozenset()
    in_holdout = df["topic_id"].isin(ids) | df["household_id"].isin(households)
    duplicate = ~in_holdout & df["text_hash"].isin(hashes)
    report = {
        "rows_in": len(df),
        "excluded_holdout": int(in_holdout.sum()),
        "excluded_duplicate_of_holdout": int(duplicate.sum()),
    }
    train = df[~in_holdout & ~duplicate]
    report["excluded_near_duplicate_of_holdout"] = 0
    if near_duplicate_threshold is not None and len(train) and in_holdout.any():
        near = _near_duplicates(train["text"], df.loc[in_holdout, "text"], near_duplicate_threshold)
        report["excluded_near_duplicate_of_holdout"] = int(near.sum())
        train = train[~near]
    report["rows_out"] = len(train)
    return train, report


def _near_duplicates(candidates: pd.Series, reference: pd.Series, threshold: float) -> np.ndarray:
    """Boolean mask over `candidates`: max cosine similarity to any reference text
    >= threshold. The vectorizer here is a similarity tool, not a model feature."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), strip_accents="unicode")
    vectorizer.fit(pd.concat([candidates, reference]).str.lower())
    sims = cosine_similarity(
        vectorizer.transform(candidates.str.lower()), vectorizer.transform(reference.str.lower())
    )
    return sims.max(axis=1) >= threshold


def holdout_frame(
    df: pd.DataFrame, name: str, manifests: list[Manifest] | None = None
) -> tuple[pd.DataFrame, dict]:
    """The frozen holdout's topics with their labels AS FROZEN. Reports topics that
    have disappeared and labels that changed since freezing (e.g. a human corrected
    a weak label): the frozen label stays the evaluation truth for comparability."""
    manifests = load_manifests() if manifests is None else manifests
    manifest = next((m for m in manifests if m.name == name), None)
    if manifest is None:
        raise KeyError(f"No holdout named {name!r}")
    held = df[df["topic_id"].isin(manifest.topic_ids)].copy()
    frozen = held["topic_id"].map(manifest.labels)
    changed = (held["category"] != frozen.str[0]) | (held["effort"] != frozen.str[1])
    held["category"], held["effort"] = frozen.str[0], frozen.str[1]
    report = {
        "expected": len(manifest.topic_ids),
        "found": len(held),
        "missing": len(manifest.topic_ids) - len(held),
        "labels_changed_since_freeze": int(changed.sum()),
    }
    return held, report


def _suggest(df: pd.DataFrame) -> None:
    summary = (
        df.groupby(["preset_key", "household_id", "household_name"])
        .agg(topics=("topic_id", "size"), categories=("category", "nunique"))
        .reset_index()
        .sort_values("preset_key")
    )
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("suggest", help="list households available for a holdout")
    f = sub.add_parser("freeze", help="freeze a new holdout manifest")
    f.add_argument("--name", required=True)
    f.add_argument("--presets", required=True, help="comma-separated preset keys to hold out")
    f.add_argument("--note", default="")
    args = parser.parse_args()

    df = load_labelled()
    if args.command == "suggest":
        _suggest(df)
        return
    keys = args.presets.split(",")
    chosen = df[df["preset_key"].isin(keys)].groupby("preset_key")["household_id"].unique()
    if set(chosen.index) != set(keys) or any(len(v) != 1 for v in chosen):
        raise SystemExit(f"Each preset must map to exactly one household; found {chosen.to_dict()}")
    manifest = freeze(df, args.name, [v[0] for v in chosen], args.note)
    print(json.dumps({k: manifest[k] for k in ("name", "households", "counts")}, indent=1))


if __name__ == "__main__":
    main()


def exclude_similar(
    candidates, reference, threshold: float | None = NEAR_DUPLICATE_THRESHOLD
) -> np.ndarray:
    """Mask over `candidates`: exact (normalised) or near duplicate of any reference
    text. Used on role-masked text, where "Liekes zwemles" and "Sems zwemles" both
    become "naamkind zwemles" and would otherwise leak across the holdout boundary."""
    candidates, reference = pd.Series(list(candidates)), pd.Series(list(reference))
    exact = candidates.map(text_hash).isin(set(reference.map(text_hash))).to_numpy()
    if threshold is None or not len(candidates) or not len(reference):
        return exact
    return exact | _near_duplicates(candidates, reference, threshold)
