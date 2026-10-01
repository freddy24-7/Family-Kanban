"""Offline evaluation of similar-task retrieval (Phase 8, lesson 07).

  uv run python -m ml.retrieval_eval --household <id>      # one household
  uv run python -m ml.retrieval_eval --simulated           # every simulated run

Replays a household's history in time order. For every reviewed ticket q, the
candidates are the tickets whose review happened BEFORE q was created (what the
planner could have seen), within the same household. Then:

1. Retrieval quality, TF-IDF vs embeddings: precision@k, relevant = same TRUE category
   (the simulator's hidden truth when available, else the confirmed label).
2. kNN effort (the planning suggestion): the similarity-weighted vote of the neighbours'
   reviewed effort, vs the effort model's logged prediction for the SAME tickets
   (paired bootstrap), overall, per category, and before/after a scenario event week.
3. A threshold sweep for MIN_SIMILARITY: coverage (share of the evaluated tickets,
   i.e. reviewed tickets with at least one earlier review, that get any suggestion) vs
   how precise the shown neighbours are.

TF-IDF is fitted per query on the query plus its candidates, exactly as in serving
(`retrieval.tfidf_similarities`): no future vocabulary. Embeddings use the raw text.
Production serves TF-IDF: it won this comparison. Households holding frozen-holdout
topics are never evaluated (the threshold must not be tuned on a holdout).
"""

import argparse
import json
from collections.abc import Callable

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from app import config
from ml import retrieval
from ml.evaluate import paired_bootstrap_delta
from ml.holdout import load_manifests

EFFORT_ORDER = ["S", "M", "L"]

HISTORY_SQL = text("""
    WITH latest_review AS (
        SELECT DISTINCT ON (topic_id) topic_id, reviewed_at, effort_actual, completed
        FROM sprint_item
        WHERE household_id = :h AND reviewed_at IS NOT NULL
        ORDER BY topic_id, reviewed_at DESC
    ), effort_prediction AS (
        SELECT DISTINCT ON (topic_id) topic_id, predicted
        FROM prediction
        WHERE household_id = :h AND task = 'effort'
        ORDER BY topic_id, created_at DESC
    )
    SELECT t.id::text AS topic_id, t.text, t.occurred_at,
           COALESCE(st.true_category, t.category_label) AS category,
           COALESCE(st.true_effort, r.effort_actual) AS effort_true,
           st.week, r.reviewed_at, r.effort_actual, r.completed,
           p.predicted AS model_effort
    FROM topic t
    LEFT JOIN sim_ticket st ON st.id = t.sim_ticket_id
    LEFT JOIN latest_review r ON r.topic_id = t.id
    LEFT JOIN effort_prediction p ON p.topic_id = t.id
    WHERE t.household_id = :h AND t.deleted_at IS NULL
    ORDER BY t.occurred_at
""")

SIMULATED_SQL = text("""
    SELECT sr.household_id::text AS household_id, sr.id::text AS run,
           sr.scenario->>'name' AS scenario,
           (SELECT min((e->>'week')::int) FROM jsonb_array_elements(sr.scenario->'events') e
            WHERE e->>'kind' = 'effort_shift') AS shift_week
    FROM simulation_run sr
    WHERE EXISTS (SELECT 1 FROM sprint_item si
                  WHERE si.household_id = sr.household_id AND si.reviewed_at IS NOT NULL)
      AND NOT EXISTS (SELECT 1 FROM topic t
                      WHERE t.household_id = sr.household_id
                        AND t.id::text = ANY(:holdout_ids))
    ORDER BY sr.created_at
""")


def holdout_topic_ids() -> list[str]:
    return sorted(set().union(*(m.topic_ids for m in load_manifests())))


def load_history(engine, household_id: str) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(HISTORY_SQL, conn, params={"h": household_id})
    if df["topic_id"].isin(holdout_topic_ids()).any():
        raise SystemExit(f"Household {household_id} holds frozen-holdout topics; not evaluated")
    return df


Scorer = Callable[[int, np.ndarray], np.ndarray]  # (query row, candidate rows) -> sims


def scorers(df: pd.DataFrame) -> dict[str, Scorer]:
    from ml import embeddings  # fastembed is a dev dependency (not in production)

    texts = df["text"].tolist()
    emb = embeddings.embed(texts)

    def tfidf(q: int, rows: np.ndarray) -> np.ndarray:
        return retrieval.tfidf_similarities(texts[q], [texts[i] for i in rows])

    def embedding(q: int, rows: np.ndarray) -> np.ndarray:
        return emb[rows] @ emb[q]

    def hybrid(q: int, rows: np.ndarray) -> np.ndarray:
        return (tfidf(q, rows) + embedding(q, rows)) / 2  # mean of the two cosines

    return {"tfidf": tfidf, "embedding": embedding, "hybrid": hybrid}


def neighbours(
    df: pd.DataFrame, scorer: Scorer, k: int, window_weeks: int | None = None
) -> list[tuple[int, list]]:
    """(query row, [(candidate row, similarity), ...]) for every reviewed ticket that has
    at least one earlier-reviewed candidate. Rows are positions in `df`.
    `window_weeks`: only reviews from the last N weeks (recency, see lesson 06)."""
    reviewed_at = df["reviewed_at"].to_numpy()
    has_review = df["reviewed_at"].notna().to_numpy()
    out = []
    for q in np.flatnonzero(has_review & df["effort_true"].notna().to_numpy()):
        created = df["occurred_at"].iloc[q]
        allowed = has_review & (reviewed_at < created)
        if window_weeks is not None:
            allowed &= reviewed_at >= created - pd.Timedelta(weeks=window_weeks)
        allowed[q] = False
        rows = np.flatnonzero(allowed)
        if len(rows):
            hits = retrieval.top_k(scorer(q, rows), k)
            out.append((q, [(int(rows[i]), sim) for i, sim in hits]))
    return out


def precision(df: pd.DataFrame, result, k: int, min_similarity: float = -1.0) -> dict:
    category = df["category"].to_numpy()
    scores, shown = [], 0
    for q, hits in result:
        hits = [(i, s) for i, s in hits if s >= min_similarity]
        if hits:
            shown += 1
            scores.append(
                retrieval.precision_at_k([category[i] == category[q] for i, _ in hits], k)
            )
    return {
        "precision": round(float(np.mean(scores)), 3) if scores else None,
        "coverage": round(shown / len(result), 3) if result else None,
    }


def knn_effort(df: pd.DataFrame, result, min_similarity: float) -> pd.DataFrame:
    """Per query: the kNN effort suggestion (None when no neighbour is similar enough)."""
    effort = df["effort_actual"].to_numpy()
    rows = []
    for q, hits in result:
        votes = retrieval.weighted_vote(
            [effort[i] for i, _ in hits], [s for _, s in hits], EFFORT_ORDER, min_similarity
        )
        rows.append({"row": q, "knn": max(votes, key=votes.get) if votes else None})
    out = pd.DataFrame(rows).set_index("row")
    return df.iloc[out.index].assign(knn=out["knn"].to_numpy())


def _accuracy(y, p) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def compare_effort(frame: pd.DataFrame) -> dict | None:
    """kNN vs the logged model prediction on the tickets where kNN made a suggestion."""
    f = frame[frame["knn"].notna() & frame["model_effort"].notna()]
    if len(f) == 0:
        return None
    delta, low, high = paired_bootstrap_delta(
        f["effort_true"], f["knn"], f["model_effort"], _accuracy
    )
    return {
        "n": len(f),
        "knn": round(_accuracy(f["effort_true"], f["knn"]), 3),
        "model": round(_accuracy(f["effort_true"], f["model_effort"]), 3),
        "delta": round(delta, 3),
        "ci95": (round(low, 3), round(high, 3)),
    }


def evaluate(
    engine,
    household_id: str,
    k: int,
    shift_week: int | None,
    retriever: str = "tfidf",
    window_weeks: int | None = None,
    min_similarity: float = retrieval.MIN_SIMILARITY,
) -> dict:
    df = load_history(engine, household_id).reset_index(drop=True)
    by_name = scorers(df)
    report: dict = {"tickets": len(df), "reviewed": int(df["reviewed_at"].notna().sum())}

    for name, scorer in by_name.items():
        result = neighbours(df, scorer, k)
        report[f"precision@{k} {name}"] = precision(df, result, k)
    result = neighbours(df, by_name[retriever], k, window_weeks)
    report[f"threshold sweep ({retriever})"] = {
        t: precision(df, result, k, t) for t in (0.3, 0.4, 0.5, 0.6, 0.7)
    }

    frame = knn_effort(df, result, min_similarity)
    report["coverage of kNN effort"] = round(float(frame["knn"].notna().mean()), 3)
    report["kNN vs model (all)"] = compare_effort(frame)
    report["kNN vs model per category"] = {
        c: compare_effort(frame[frame["category"] == c]) for c in sorted(frame["category"].unique())
    }
    if shift_week is not None and frame["week"].notna().any():
        for label, part in (
            (f"before week {shift_week}", frame[frame["week"] < shift_week]),
            (f"from week {shift_week}", frame[frame["week"] >= shift_week]),
        ):
            report[f"kNN vs model {label}"] = compare_effort(part)
            report[f"groceries {label}"] = compare_effort(part[part["category"] == "groceries"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--household")
    target.add_argument("--simulated", action="store_true", help="every simulated run")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--shift-week", type=int, help="scenario event week (if --household)")
    parser.add_argument(
        "--retriever",
        choices=["tfidf", "embedding", "hybrid"],
        default="tfidf",
        help="retriever for the threshold sweep and kNN effort",
    )
    parser.add_argument("--window-weeks", type=int, help="kNN: only reviews of the last N weeks")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=retrieval.MIN_SIMILARITY,
        help="kNN: neighbours below this similarity don't vote (scale differs per retriever)",
    )
    args = parser.parse_args()

    engine = create_engine(config.DATABASE_URL)
    try:
        if args.household:
            targets = [(args.household, args.household[:8], args.shift_week)]
        else:
            with engine.connect() as conn:
                runs = (
                    conn.execute(SIMULATED_SQL, {"holdout_ids": holdout_topic_ids()})
                    .mappings()
                    .all()
                )
            targets = [
                (r["household_id"], f"run {r['run'][:8]} ({r['scenario']})", r["shift_week"])
                for r in runs
            ]
        for household_id, label, shift_week in targets:
            print(f"\n=== {label}")
            report = evaluate(
                engine,
                household_id,
                args.k,
                shift_week,
                args.retriever,
                args.window_weeks,
                args.min_similarity,
            )
            print(json.dumps(report, indent=1, default=str))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
