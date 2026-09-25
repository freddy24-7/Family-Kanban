"""Fixed vocabularies shared by the API, the ML code and the simulator.

Keys are English; Dutch display labels live in the frontend.
"""

from enum import StrEnum


class Category(StrEnum):
    CHORES = "chores"
    GROCERIES = "groceries"
    KIDS = "kids"
    HOME_MAINTENANCE = "home_maintenance"
    FINANCE = "finance"
    SOCIAL = "social"
    OTHER = "other"


class Effort(StrEnum):
    S = "S"
    M = "M"
    L = "L"


class Task(StrEnum):
    """What a model predicts."""

    CATEGORY = "category"
    EFFORT = "effort"


class Source(StrEnum):
    """Data provenance. Metrics are always reported per source."""

    REAL = "real"
    SIMULATED = "simulated"


class LabelSource(StrEnum):
    PLANNER = "planner"  # confirmed/corrected at backlog review
    REVIEW = "review"  # set at sprint review
    GENERATOR = "generator"  # weak label from the LLM ticket generator


class ModelStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    RETIRED = "retired"
    REJECTED = "rejected"
