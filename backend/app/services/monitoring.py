"""Assembles the monitoring stream from the database and runs the detectors
(ml/monitoring.py): one row per ticket with its intake prediction, the planner's
label, the sprint-review effort and, for simulations, the hidden truth."""

import uuid
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import repository
from app.domain import Task
from app.models import ModelVersion, Prediction, SimTicket, SprintItem, Topic
from ml import monitoring

ReferenceKind = Literal["holdout", "first_weeks"]
HOLDOUT = "holdout-sim-v1"


async def stream_rows(
    session: AsyncSession, household_ids: list[uuid.UUID]
) -> list[dict[str, Any]]:
    topics = (
        await session.scalars(
            select(Topic).where(Topic.household_id.in_(household_ids), Topic.deleted_at.is_(None))
        )
    ).all()
    if not topics:
        return []
    ids = [t.id for t in topics]
    # The FIRST prediction per task is the one the family saw at intake (later ones
    # come from edits or re-scoring); that's the one to monitor.
    intake: dict[tuple[uuid.UUID, Task], Prediction] = {}
    for p in (
        await session.scalars(
            select(Prediction).where(Prediction.topic_id.in_(ids)).order_by(Prediction.created_at)
        )
    ).all():
        intake.setdefault((p.topic_id, p.task), p)
    actual = {
        i.topic_id: i.effort_actual
        for i in (
            await session.scalars(
                select(SprintItem)
                .where(SprintItem.topic_id.in_(ids), SprintItem.completed.is_(True))
                .order_by(SprintItem.reviewed_at)
            )
        ).unique()
    }
    sim_ids = [t.sim_ticket_id for t in topics if t.sim_ticket_id]
    truth = (
        {
            s.id: s
            for s in (
                await session.scalars(select(SimTicket).where(SimTicket.id.in_(sim_ids)))
            ).all()
        }
        if sim_ids
        else {}
    )
    rows = []
    for t in topics:
        cat, eff = intake.get((t.id, Task.CATEGORY)), intake.get((t.id, Task.EFFORT))
        sim = truth.get(t.sim_ticket_id)
        rows.append(
            {
                "week": monitoring.week_start(t.occurred_at.date()),
                "text": t.text,
                "pred_category": cat.predicted if cat else None,
                "category_conf": cat.confidence if cat else None,
                "pred_effort": eff.predicted if eff else None,
                "effort_conf": eff.confidence if eff else None,
                "model": cat.model_version_id if cat else None,
                "label_category": str(t.category_label) if t.category_label else None,
                "category_changed": t.category_prediction_changed,
                "actual_effort": str(actual[t.id]) if actual.get(t.id) else None,
                "truth_category": str(sim.true_category) if sim else None,
                "truth_effort": str(sim.true_effort) if sim else None,
            }
        )
    return rows


async def holdout_reference(session: AsyncSession) -> dict[str, Any] | None:
    """Merge the active category and effort models' stored holdout profiles."""
    active = await repository.active_model_versions(session)
    merged: dict[str, Any] = {}
    for version in active.values():
        merged |= (version.metrics.get("reference_profile") or {}).get(HOLDOUT, {})
    return merged if "category_mix" in merged else None


def first_weeks_reference(rows: list[dict[str, Any]], weeks: int = 4) -> dict[str, Any] | None:
    """The stream's own first weeks as 'normal' (reviewed effort as effort labels)."""
    if not rows:
        return None
    first = min(r["week"] for r in rows)
    early = [
        {**r, "label_effort": r["actual_effort"]}
        for r in rows
        if (r["week"] - first).days < 7 * weeks and r["pred_category"]
    ]
    return monitoring.profile(early) if early else None


async def report(
    session: AsyncSession,
    household_ids: list[uuid.UUID],
    reference_kind: ReferenceKind,
    window_weeks: int,
    events: list[dict] | None = None,
    start=None,
) -> dict[str, Any]:
    rows = await stream_rows(session, household_ids)
    reference = (
        await holdout_reference(session)
        if reference_kind == "holdout"
        else first_weeks_reference(rows)
    )
    if reference is None:
        return {
            "reference_kind": reference_kind,
            "reference": None,
            "weekly": [],
            "detection": None,
        }
    weekly = monitoring.monitor(rows, reference, window_weeks)
    detection = (
        monitoring.detection_summary(weekly, events or [], start)
        if events is not None and start
        else None
    )
    return {
        "reference_kind": reference_kind,
        "reference": reference,
        "n_tickets": len(rows),
        "weekly": weekly,
        "detection": detection,
    }


async def model_log(session: AsyncSession) -> list[dict[str, Any]]:
    """Every model version with its evaluation: the retraining log."""
    versions = (await session.scalars(select(ModelVersion).order_by(ModelVersion.created_at))).all()
    log = []
    for v in versions:
        holdout = (v.metrics.get("holdouts") or {}).get(HOLDOUT, {})
        log.append(
            {
                "name": v.name,
                "task": str(v.task),
                "status": str(v.status),
                "created_at": v.created_at.isoformat(),
                "training_set_size": v.training_set_size,
                "holdout_macro_f1": holdout.get("macro_f1"),
                "holdout_macro_f1_ci95": holdout.get("macro_f1_ci95"),
                "cv_macro_f1": (v.metrics.get("cv") or {}).get("macro_f1"),
                "notes": v.notes,
            }
        )
    return log
