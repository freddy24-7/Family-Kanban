"""ORM models.

Tenancy: every tenant-owned table carries household_id.
Provenance: topics and predictions carry `source` (real | simulated).
Predictions are append-only. Frozen holdouts are
committed manifest files (backend/ml/holdouts/), not tables.
"""

import uuid
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from fastapi_users_db_sqlalchemy import (
    SQLAlchemyBaseOAuthAccountTableUUID,
    SQLAlchemyBaseUserTableUUID,
)
from fastapi_users_db_sqlalchemy.access_token import SQLAlchemyBaseAccessTokenTableUUID
from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base
from app.domain import (
    Category,
    Effort,
    ItemStatus,
    LabelSource,
    ModelStatus,
    Source,
    SprintStatus,
    Task,
)


def str_enum(enum_cls: type[StrEnum], name: str | None = None) -> Enum:
    """Store enum values as VARCHAR + CHECK constraint (easy to extend via migrations).
    `name` becomes the CHECK constraint's name; it must be unique per table, so pass it
    when a table has several columns of the same enum."""
    return Enum(
        enum_cls,
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda e: [m.value for m in e],
        name=name or enum_cls.__name__.lower(),
    )


def uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(Uuid, primary_key=True, default=uuid.uuid4)


def created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


# --- Identity (fastapi-users) -------------------------------------------------


class OAuthAccount(SQLAlchemyBaseOAuthAccountTableUUID, Base):
    pass


class User(SQLAlchemyBaseUserTableUUID, Base):
    # is_superuser (from fastapi-users) = platform admin.
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    # Simulated family members are users with login disabled (is_active=False).
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = created_at()

    oauth_accounts: Mapped[list[OAuthAccount]] = relationship(lazy="joined")


class AccessToken(SQLAlchemyBaseAccessTokenTableUUID, Base):
    pass


# --- Tenancy ------------------------------------------------------------------


class Household(Base):
    __tablename__ = "household"

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    # Simulated households get source=simulated on everything they produce.
    kind: Mapped[Source] = mapped_column(str_enum(Source), nullable=False)
    # Only training-eligible households feed model training (set by platform admin).
    training_eligible: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = created_at()


class Membership(Base):
    __tablename__ = "membership"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), primary_key=True
    )
    household_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("household.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    is_planner: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_reviewer: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Used by the classifier's role tokens: a child's name becomes "naamkind".
    is_child: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false"), nullable=False
    )
    # Simulator-only person attributes (gender, age, world-model traits).
    profile: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    joined_at: Mapped[datetime] = created_at()

    user: Mapped[User] = relationship(lazy="joined")


class Invite(Base):
    __tablename__ = "invite"

    id: Mapped[uuid.UUID] = uuid_pk()
    household_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("household.id", ondelete="CASCADE"), index=True, nullable=False
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    # Only a SHA-256 hash is stored; the raw token exists only in the emailed link.
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    is_planner: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_reviewer: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_child: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false"), nullable=False
    )
    invited_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("user.id", ondelete="SET NULL"))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = created_at()


# --- Topics, models, predictions ----------------------------------------------


class Topic(Base):
    __tablename__ = "topic"
    __table_args__ = (Index("ix_topic_household_occurred", "household_id", "occurred_at"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    household_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("household.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    due_by: Mapped[date | None] = mapped_column(Date)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("user.id", ondelete="SET NULL"))
    # Domain time (from the injectable clock) vs. database insert time.
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = created_at()
    source: Mapped[Source] = mapped_column(str_enum(Source), nullable=False)

    # Ground-truth labels (null until confirmed). These are training targets,
    # never features.
    category_label: Mapped[Category | None] = mapped_column(str_enum(Category))
    effort_label: Mapped[Effort | None] = mapped_column(str_enum(Effort))
    label_source: Mapped[LabelSource | None] = mapped_column(str_enum(LabelSource))
    labeled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    labeled_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("user.id", ondelete="SET NULL"))
    # Did the human change what the model predicted? Guards against automation
    # bias: if people rubber-stamp predictions, measured accuracy is inflated.
    category_prediction_changed: Mapped[bool | None] = mapped_column(Boolean)
    effort_prediction_changed: Mapped[bool | None] = mapped_column(Boolean)
    # Edits re-run the classifier (new predictions are appended). Deletion is soft:
    # hidden from the app and excluded from training, but kept so frozen holdouts
    # and history stay intact.
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Data lineage: which generator run produced this (synthetic) topic.
    generation_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("generation_run.id", ondelete="SET NULL"), index=True
    )
    # Simulator lineage: the pool ticket (with hidden ground truth) this topic replays.
    sim_ticket_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sim_ticket.id", ondelete="SET NULL"), index=True
    )


class ModelVersion(Base):
    __tablename__ = "model_version"
    __table_args__ = (
        # At most one active model per task.
        Index(
            "uq_model_version_active_per_task",
            "task",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    task: Mapped[Task] = mapped_column(str_enum(Task), nullable=False)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    status: Mapped[ModelStatus] = mapped_column(str_enum(ModelStatus), nullable=False)
    # Resolved only by ml/registry.py. "stub://" = built-in placeholder model.
    artifact_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    training_set_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    training_data_hash: Mapped[str | None] = mapped_column(String(64))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("model_version.id"))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at()


class ModelArtifact(Base):
    """Serialized model bytes (joblib, compressed). Stored in Postgres because the
    database is the one thing the training machine and the API share; only
    ml/registry.py reads or writes this table."""

    __tablename__ = "model_artifact"

    id: Mapped[uuid.UUID] = uuid_pk()
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = created_at()


class Prediction(Base):
    """Append-only: never updated or deleted by application code."""

    __tablename__ = "prediction"
    __table_args__ = (Index("ix_prediction_household_occurred", "household_id", "occurred_at"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    topic_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("topic.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # Denormalised from the topic so monitoring can query per household cheaply.
    household_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("household.id", ondelete="CASCADE"), nullable=False
    )
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_version.id"), index=True, nullable=False
    )
    task: Mapped[Task] = mapped_column(str_enum(Task), nullable=False)
    predicted: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    probabilities: Mapped[dict[str, float]] = mapped_column(JSONB, default=dict, nullable=False)
    source: Mapped[Source] = mapped_column(str_enum(Source), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = created_at()


# --- Demo / simulation ------------------------------------------------------------


class FamilyProfile(Base):
    """Definition of a simulated family (members, home, pets, vehicles)."""

    __tablename__ = "family_profile"

    household_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("household.id", ondelete="CASCADE"), primary_key=True
    )
    spec: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    preset_key: Mapped[str | None] = mapped_column(String(50))
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = created_at()


class GenerationRun(Base):
    """One call of the ticket generator for a household: what was asked, what came
    back, what was rejected, and what it cost. LLM output is not reproducible, so
    this record (plus the prompt version) is the lineage of every synthetic topic."""

    __tablename__ = "generation_run"

    id: Mapped[uuid.UUID] = uuid_pk()
    household_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("household.id", ondelete="CASCADE"), index=True, nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # running|completed|failed
    prompt_version: Mapped[str] = mapped_column(String(20), nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    requested: Mapped[int] = mapped_column(Integer, nullable=False)
    produced: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rejected_invalid: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rejected_duplicate: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Gemini labelled the text differently from the category we asked for:
    # a direct measure of how ambiguous the categories are.
    category_mismatches: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    models_used: Mapped[dict[str, int]] = mapped_column(JSONB, default=dict, nullable=False)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# --- Sprints ----------------------------------------------------------------------


class Sprint(Base):
    __tablename__ = "sprint"
    __table_args__ = (Index("ix_sprint_household_status", "household_id", "status"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    household_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("household.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[SprintStatus] = mapped_column(str_enum(SprintStatus), nullable=False)
    created_at: Mapped[datetime] = created_at()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SprintItem(Base):
    """A backlog topic planned into a sprint, with its board state and its review.
    The review's `effort_actual` is kept next to the planner's estimate on the topic
    (topic.effort_label): estimate vs actual are both useful signals."""

    __tablename__ = "sprint_item"
    __table_args__ = (UniqueConstraint("sprint_id", "topic_id", name="uq_sprint_item_topic"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    sprint_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sprint.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # Denormalised for household-scoped queries.
    household_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("household.id", ondelete="CASCADE"), index=True, nullable=False
    )
    topic_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("topic.id", ondelete="CASCADE"), index=True, nullable=False
    )
    assignee_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL")
    )
    status: Mapped[ItemStatus] = mapped_column(str_enum(ItemStatus), nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = created_at()
    status_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Sprint review (labelling point #2)
    completed: Mapped[bool | None] = mapped_column(Boolean)
    effort_actual: Mapped[Effort | None] = mapped_column(str_enum(Effort))
    review_note: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL")
    )

    topic: Mapped[Topic] = relationship(lazy="joined")


class SprintReview(Base):
    """Sprint-level review, written when the sprint is completed."""

    __tablename__ = "sprint_review"

    id: Mapped[uuid.UUID] = uuid_pk()
    sprint_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sprint.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    household_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("household.id", ondelete="CASCADE"), index=True, nullable=False
    )
    notes: Mapped[str | None] = mapped_column(Text)
    completed_count: Mapped[int] = mapped_column(Integer, nullable=False)
    total_count: Mapped[int] = mapped_column(Integer, nullable=False)
    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = created_at()


# --- Simulation (Phase 5) -------------------------------------------------------------


class SimulationRun(Base):
    """One simulated household living through `weeks` sprints under a scenario.

    Two stages: the *pool* (Gemini writes each week's tickets for their hidden truth;
    stored in sim_ticket) and the *run* (replays the pool through the real app
    services with a seeded RNG and a simulated clock). A replay run reuses another
    run's pool (`pool_run_id`) with its own household, so the same world can be
    re-lived with, say, a sloppier planner or a newer model."""

    __tablename__ = "simulation_run"

    id: Mapped[uuid.UUID] = uuid_pk()
    household_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("household.id", ondelete="CASCADE"), index=True, nullable=False
    )
    pool_run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("simulation_run.id"))
    scenario: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    planner: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    # pool_pending | pool_ready | running | completed | failed
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    current_week: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    weekly_stats: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SimTicket(Base):
    """A pool ticket: text written by Gemini for a truth decided by the world model."""

    __tablename__ = "sim_ticket"
    __table_args__ = (Index("ix_sim_ticket_run_week", "simulation_run_id", "week"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    simulation_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("simulation_run.id", ondelete="CASCADE"), nullable=False
    )
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    submitter: Mapped[str] = mapped_column(String(100), nullable=False)
    style: Mapped[str] = mapped_column(String(20), nullable=False)
    # Hidden ground truth (never shown in the app, never a feature).
    true_category: Mapped[Category] = mapped_column(
        str_enum(Category, "true_category"), nullable=False
    )
    true_effort: Mapped[Effort] = mapped_column(str_enum(Effort, "true_effort"), nullable=False)
    # The effort the TEXT was written for. Differs from true_effort only under
    # planted concept drift (same kind of text, different real effort).
    text_effort: Mapped[Effort] = mapped_column(str_enum(Effort, "text_effort"), nullable=False)
    # Gemini's own (weak) labels for the text it wrote, for comparison.
    gemini_category: Mapped[Category | None] = mapped_column(str_enum(Category, "gemini_category"))
    gemini_effort: Mapped[Effort | None] = mapped_column(str_enum(Effort, "gemini_effort"))
    events: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
