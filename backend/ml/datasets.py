"""Loading labelled data for ML, and versioned dataset snapshots.

ML code reads the database synchronously (batch jobs, notebooks) through this
module only. Snapshots are the reproducible unit: LLM generation can't be re-run
identically, but a snapshot file + its content hash can.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from app import config
from ml.text import text_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = REPO_ROOT / "data" / "datasets"

# Only confirmed/generator labels from training-eligible households. Metadata
# columns (household, source, submitter, run) are for splitting and analysis;
# they are never model features.
LABELLED_TOPICS_SQL = text(
    """
    SELECT t.id::text AS topic_id,
           t.household_id::text AS household_id,
           h.name AS household_name,
           fp.preset_key,
           t.source,
           t.text,
           t.category_label AS category,
           t.effort_label AS effort,
           t.label_source,
           t.occurred_at,
           t.generation_run_id::text AS generation_run_id,
           m.profile ->> 'role' AS submitter_role,
           (m.profile ->> 'age')::int AS submitter_age
    FROM topic t
    JOIN household h ON h.id = t.household_id
    LEFT JOIN family_profile fp ON fp.household_id = t.household_id
    LEFT JOIN membership m ON m.user_id = t.created_by AND m.household_id = t.household_id
    WHERE h.training_eligible
      AND t.category_label IS NOT NULL
      AND t.effort_label IS NOT NULL
    ORDER BY t.id
    """
)


def load_labelled(database_url: str | None = None) -> pd.DataFrame:
    url = database_url or config.DATABASE_URL
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(LABELLED_TOPICS_SQL, conn)
    finally:
        engine.dispose()
    df["text_hash"] = df["text"].map(text_hash)
    return df


def dataset_hash(df: pd.DataFrame) -> str:
    """Content hash over exactly what a model learns from: id, text and labels.
    Row order doesn't matter; any change to a text or label changes the hash."""
    rows = sorted(zip(df["topic_id"], df["text"], df["category"], df["effort"], strict=True))
    digest = hashlib.sha256()
    for row in rows:
        digest.update("\x1f".join(row).encode())
        digest.update(b"\x1e")
    return digest.hexdigest()


def save_snapshot(df: pd.DataFrame, name: str, directory: Path = SNAPSHOT_DIR) -> dict:
    """Write <name>.parquet + <name>.json (metadata). Never overwrites."""
    directory.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = directory / f"{name}.parquet", directory / f"{name}.json"
    if data_path.exists() or meta_path.exists():
        raise FileExistsError(f"Snapshot {name!r} already exists; pick a new name")
    meta = {
        "name": name,
        "created_at": datetime.now(UTC).isoformat(),
        "rows": len(df),
        "dataset_hash": dataset_hash(df),
        "households": int(df["household_id"].nunique()),
        "by_source": df["source"].value_counts().to_dict(),
        "by_category": df["category"].value_counts().to_dict(),
        "by_effort": df["effort"].value_counts().to_dict(),
    }
    df.to_parquet(data_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return meta


def load_snapshot(name: str, directory: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    df = pd.read_parquet(directory / f"{name}.parquet")
    meta = json.loads((directory / f"{name}.json").read_text())
    if dataset_hash(df) != meta["dataset_hash"]:
        raise ValueError(f"Snapshot {name!r} does not match its recorded hash")
    return df
