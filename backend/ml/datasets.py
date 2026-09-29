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


def save_snapshot(
    df: pd.DataFrame,
    name: str,
    directory: Path = SNAPSHOT_DIR,
    members: dict[str, list[tuple[str, bool]]] | None = None,
) -> dict:
    """Write <name>.parquet + <name>.json (metadata). Never overwrites.

    `members` (household -> [(name, is_child)]) is stored in the metadata so that
    role-token masking is reproducible from the snapshot alone, independent of which
    database happens to be configured."""
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
    if members is not None:
        households = set(df["household_id"])
        kept = {h: [list(m) for m in ms] for h, ms in sorted(members.items()) if h in households}
        meta["members"] = kept
        meta["members_hash"] = hashlib.sha256(json.dumps(kept, sort_keys=True).encode()).hexdigest()
    df.to_parquet(data_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return meta


def load_snapshot(name: str, directory: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    df = pd.read_parquet(directory / f"{name}.parquet")
    meta = json.loads((directory / f"{name}.json").read_text())
    if dataset_hash(df) != meta["dataset_hash"]:
        raise ValueError(f"Snapshot {name!r} does not match its recorded hash")
    return df


GENERATION_RUNS_SQL = text(
    """
    SELECT gr.id::text AS run_id, fp.preset_key, gr.status, gr.prompt_version,
           gr.requested, gr.produced, gr.rejected_invalid, gr.rejected_duplicate,
           gr.category_mismatches, gr.models_used, gr.tokens_in, gr.tokens_out, gr.error
    FROM generation_run gr
    JOIN household h ON h.id = gr.household_id
    LEFT JOIN family_profile fp ON fp.household_id = gr.household_id
    WHERE h.training_eligible
    ORDER BY gr.created_at
    """
)


def load_generation_runs(database_url: str | None = None) -> pd.DataFrame:
    """Generator lineage/quality per run (for EDA; not training data)."""
    engine = create_engine(database_url or config.DATABASE_URL)
    try:
        with engine.connect() as conn:
            return pd.read_sql(GENERATION_RUNS_SQL, conn)
    finally:
        engine.dispose()


HOUSEHOLD_MEMBERS_SQL = text(
    """
    SELECT m.household_id::text AS household_id, u.display_name AS name, m.is_child
    FROM membership m JOIN "user" u ON u.id = m.user_id
    """
)


def load_household_members(database_url: str | None = None) -> dict[str, list[tuple[str, bool]]]:
    """household_id -> [(display name, is_child)], for role-token masking."""
    engine = create_engine(database_url or config.DATABASE_URL)
    try:
        with engine.connect() as conn:
            rows = conn.execute(HOUSEHOLD_MEMBERS_SQL).all()
    finally:
        engine.dispose()
    members: dict[str, list[tuple[str, bool]]] = {}
    for household_id, name, is_child in rows:
        members.setdefault(household_id, []).append((name, bool(is_child)))
    return members


def load_snapshot_members(
    name: str, directory: Path = SNAPSHOT_DIR
) -> dict[str, list[tuple[str, bool]]]:
    meta = json.loads((directory / f"{name}.json").read_text())
    if "members" not in meta:
        raise ValueError(f"Snapshot {name!r} has no member lists; create a newer snapshot")
    members = meta["members"]
    digest = hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest()
    if digest != meta["members_hash"]:
        raise ValueError(f"Snapshot {name!r} member lists do not match their hash")
    return {h: [(n, bool(c)) for n, c in ms] for h, ms in members.items()}
