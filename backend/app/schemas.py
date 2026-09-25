"""Request/response models for the domain API (auth schemas live in auth.py)."""

import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.domain import Category, Effort, LabelSource, Source


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
    is_simulated: bool


class HouseholdDetail(HouseholdRead):
    members: list[MemberRead]


# --- Invites ------------------------------------------------------------------------


class InviteCreate(BaseModel):
    email: EmailStr
    is_planner: bool = False
    is_reviewer: bool = False


class InviteRead(ORMModel):
    id: uuid.UUID
    household_id: uuid.UUID
    email: str
    is_planner: bool
    is_reviewer: bool
    expires_at: datetime
    accepted_at: datetime | None


class InviteAccept(BaseModel):
    token: str = Field(min_length=10, max_length=200)


# --- Topics -------------------------------------------------------------------------


class TopicCreate(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
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
