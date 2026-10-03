"""Request/response models for the domain API (auth schemas live in auth.py)."""

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.domain import Category, Effort, ItemStatus, LabelSource, Source, SprintStatus
from sim.family import FamilySpec
from sim.world import PlannerBehaviour


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- Households -------------------------------------------------------------------


class HouseholdCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class AdminHouseholdCreate(HouseholdCreate):
    kind: Source = Source.SIMULATED
    training_eligible: bool = False


class AdminHouseholdUpdate(BaseModel):
    training_eligible: bool


class HouseholdRead(ORMModel):
    id: uuid.UUID
    name: str
    kind: Source
    training_eligible: bool
    created_at: datetime


class MemberRead(BaseModel):
    user_id: uuid.UUID
    display_name: str
    email: str
    is_planner: bool
    is_reviewer: bool
    is_child: bool
    is_simulated: bool


class HouseholdDetail(HouseholdRead):
    members: list[MemberRead]


# --- Invites ------------------------------------------------------------------------


class InviteCreate(BaseModel):
    email: EmailStr
    is_planner: bool = False
    is_reviewer: bool = False
    is_child: bool = False


class InviteRead(ORMModel):
    id: uuid.UUID
    household_id: uuid.UUID
    email: str
    is_planner: bool
    is_reviewer: bool
    is_child: bool
    expires_at: datetime
    accepted_at: datetime | None


class InviteAccept(BaseModel):
    token: str = Field(min_length=10, max_length=200)


# --- Topics -------------------------------------------------------------------------


class TopicCreate(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    due_by: date | None = None


class TopicUpdate(BaseModel):
    """Only the fields that are sent are changed (due_by: null clears the date)."""

    text: str | None = Field(default=None, min_length=1, max_length=2000)
    due_by: date | None = None


class LabelsUpdate(BaseModel):
    category: Category
    effort: Effort


class PredictionRead(BaseModel):
    category: str | None
    category_confidence: float | None
    effort: str | None
    effort_confidence: float | None
    needs_review: bool


class LabelsRead(BaseModel):
    category: Category | None
    effort: Effort | None
    source: LabelSource | None
    labeled_at: datetime | None


class TopicRead(BaseModel):
    id: uuid.UUID
    household_id: uuid.UUID
    text: str
    due_by: date | None
    created_by: uuid.UUID | None
    occurred_at: datetime
    source: Source
    prediction: PredictionRead
    labels: LabelsRead


# --- Demo families (platform admin) --------------------------------------------------


class DemoFamilyCreate(BaseModel):
    """Either a preset key or a full spec."""

    preset: str | None = None
    spec: FamilySpec | None = None
    name: str | None = Field(default=None, max_length=100)
    training_eligible: bool = False
    random_seed: int | None = None


class GenerationRequest(BaseModel):
    count: int = Field(ge=1, le=500)
    random_seed: int | None = None


class GenerationRunRead(ORMModel):
    id: uuid.UUID
    household_id: uuid.UUID
    status: str
    prompt_version: str
    requested: int
    produced: int
    rejected_invalid: int
    rejected_duplicate: int
    category_mismatches: int
    models_used: dict[str, int]
    tokens_in: int
    tokens_out: int
    error: str | None
    created_at: datetime
    finished_at: datetime | None


# --- Sprints ------------------------------------------------------------------------


class SprintCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    start_date: date
    end_date: date


class SprintRead(BaseModel):
    id: uuid.UUID
    name: str
    start_date: date
    end_date: date
    status: SprintStatus
    started_at: datetime | None
    completed_at: datetime | None
    item_count: int
    done_count: int


class SprintItemCreate(BaseModel):
    topic_id: uuid.UUID
    assignee_id: uuid.UUID | None = None


class SprintItemUpdate(BaseModel):
    assignee_id: uuid.UUID | None


class SprintItemMove(BaseModel):
    status: ItemStatus
    position: int = Field(default=0, ge=0)


class SprintItemReviewIn(BaseModel):
    completed: bool
    effort_actual: Effort | None = None
    note: str | None = Field(default=None, max_length=1000)


class SprintItemRead(BaseModel):
    id: uuid.UUID
    topic: TopicRead
    assignee_id: uuid.UUID | None
    assignee_name: str | None
    status: ItemStatus
    position: int
    completed: bool | None
    effort_actual: Effort | None
    review_note: str | None
    reviewed_at: datetime | None


class SimilarTaskRead(BaseModel):
    """An earlier, reviewed task of the same household that resembles a backlog topic."""

    topic_id: uuid.UUID
    text: str
    similarity: float
    assignee_name: str | None
    completed: bool | None
    effort_actual: Effort | None
    reviewed_at: datetime | None


class TopicSuggestionRead(BaseModel):
    topic_id: uuid.UUID
    similar: list[SimilarTaskRead]  # most similar first; empty: nothing comparable done


class ProposalRequest(BaseModel):
    max_items: int = Field(10, ge=1, le=50)
    method: Literal["auto", "llm", "rules"] = "auto"


class ProposalItemRead(BaseModel):
    topic_id: uuid.UUID
    text: str
    assignee_id: uuid.UUID
    assignee_name: str
    reason: str


class ProposalRead(BaseModel):
    """A draft sprint from the Planner Assistant; nothing is planned until accepted."""

    method: Literal["llm", "rules"]
    # "llm_not_configured" | "llm_failed": rules were used instead of Gemini.
    note: str | None
    items: list[ProposalItemRead]


class SprintCompleteIn(BaseModel):
    notes: str | None = Field(default=None, max_length=2000)


class SprintReviewRead(ORMModel):
    notes: str | None
    completed_count: int
    total_count: int
    created_at: datetime


class SprintDetail(SprintRead):
    items: list[SprintItemRead]
    review: SprintReviewRead | None


# --- Simulations (platform admin) -------------------------------------------------------


class SimulationCreate(BaseModel):
    scenario: str = "drift-demo"
    weeks: int = Field(26, ge=1, le=104)
    start_date: date = date(2026, 7, 6)
    seed: int | None = None
    family_seed: int | None = None
    planner: PlannerBehaviour = Field(default_factory=PlannerBehaviour)


class SimulationReplay(BaseModel):
    seed: int | None = None
    planner: PlannerBehaviour = Field(default_factory=PlannerBehaviour)
    # Shadow evaluation: {"category": "category-v2", ...} instead of the active models.
    model_versions: dict[str, str] | None = None
    # Per-household effort adjustment (Phase 7b) on for this replay.
    adaptation: bool = False


class SimulationRead(ORMModel):
    id: uuid.UUID
    household_id: uuid.UUID
    pool_run_id: uuid.UUID | None
    scenario: dict
    planner: dict
    start_date: date
    weeks: int
    random_seed: int
    model_versions: dict[str, str] | None = None
    adaptation: bool = False
    status: str
    current_week: int
    tokens_in: int
    tokens_out: int
    error: str | None
    created_at: datetime
    finished_at: datetime | None


class SimulationDetail(SimulationRead):
    weekly_stats: list[dict]
