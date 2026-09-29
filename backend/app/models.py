"""ORM models.

Tenancy: every tenant-owned table carries household_id.
Provenance: topics and predictions carry `source` (real | simulated).
Predictions are append-only. Sprint tables arrive in Phase 4, SimulationRun in
Phase 5. Frozen holdouts are committed manifest files (backend/ml/holdouts/), not tables.
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
    String,
    Text,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base
from app.domain import Category, Effort, LabelSource, ModelStatus, Source, Task


def str_enum(enum_cls: type[StrEnum]) -> Enum:
    """Store enum values as VARCHAR + CHECK constraint (easy to extend via migrations)."""
    return Enum(
        enum_cls,
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda e: [m.value for m in e],
        name=enum_cls.__name__.lower(),
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
    # Data lineage: which generator run produced this (synthetic) topic.
    generation_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("generation_run.id", ondelete="SET NULL"), index=True
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
