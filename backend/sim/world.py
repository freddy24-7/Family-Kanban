"""The world model: the simulator's hidden ground truth.

Per week it decides how many tickets arrive, each ticket's TRUE category and effort,
who submits it and in which style; and during the sprint who does it and whether it
gets done. Scenario events change the world at a known week, so drift detectors and
retraining can be tested against a truth we planted.
See docs/learning/04-simulation-and-drift.md.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from app.domain import Category, Effort
from sim.family import FamilySpec, FamilyWorld, KidSpec, Person, category_weights
from sim.generator import STYLES

EFFORTS = [Effort.S, Effort.M, Effort.L]

# --- Scenario definition ----------------------------------------------------------------


class Event(BaseModel):
    """A planted change of the world, from `week` onwards."""

    week: int = Field(ge=0)
    kind: Literal["add_pet", "category_boost", "effort_shift", "new_typist"]
    note: str = ""
    # add_pet (prior + covariate drift: more chores, a new name in the texts)
    species: Literal["dog", "cat"] | None = None
    # category_boost (prior drift) / effort_shift (concept drift)
    category: Category | None = None
    factor: float | None = None
    # effort_shift: the TRUE effort distribution of `category` changes, the texts don't
    effort: dict[Effort, float] | None = None


class Scenario(BaseModel):
    name: str
    description: str = ""
    events: list[Event] = Field(default_factory=list)


class PlannerBehaviour(BaseModel):
    """How carefully the simulated planner labels the backlog each week."""

    check_rate: float = Field(0.9, ge=0, le=1)  # share of new tickets labelled at all
    rubber_stamp_rate: float = Field(0.2, ge=0, le=1)  # accept the model's guess unchecked
    label_error_rate: float = Field(0.03, ge=0, le=1)  # honest mistakes when checking
    capacity_per_person: int = Field(6, ge=1)  # sprint items per member per week
    # Phase 9 planner experiment (defaults keep earlier runs and replays unchanged):
    # who plans the sprint, and whether completion depends on each person's load.
    policy: Literal["sim", "rules", "llm"] = "sim"
    work_model: Literal["simple", "load"] = "simple"
    max_items: int = Field(14, ge=1)  # rules/llm: sprint size limit given to the planner


# --- Hidden rules -----------------------------------------------------------------------

# Seasonal multipliers on the family's category weights (month -> category -> factor).
SEASONS: dict[int, dict[Category, float]] = {
    6: {Category.HOME_MAINTENANCE: 1.4, Category.SOCIAL: 1.3},
    7: {Category.HOME_MAINTENANCE: 1.5, Category.SOCIAL: 1.4, Category.KIDS: 0.7},
    8: {Category.HOME_MAINTENANCE: 1.4, Category.SOCIAL: 1.3, Category.KIDS: 0.8},
    9: {Category.KIDS: 1.7, Category.FINANCE: 1.1},
    10: {Category.HOME_MAINTENANCE: 1.3, Category.KIDS: 1.2},
    11: {Category.HOME_MAINTENANCE: 1.2, Category.SOCIAL: 1.2},
    12: {Category.SOCIAL: 1.9, Category.FINANCE: 1.4, Category.GROCERIES: 1.2},
    1: {Category.FINANCE: 1.5, Category.HOME_MAINTENANCE: 0.8},
}

# True effort per category (S, M, L). More large tasks than the seed dataset's 1.4% L:
# Gemini under-estimated effort, and the simulator shouldn't inherit that bias.
EFFORT_BASE: dict[Category, tuple[float, float, float]] = {
    Category.CHORES: (0.60, 0.30, 0.10),
    Category.GROCERIES: (0.70, 0.28, 0.02),
    Category.KIDS: (0.50, 0.40, 0.10),
    Category.HOME_MAINTENANCE: (0.30, 0.40, 0.30),
    Category.FINANCE: (0.50, 0.40, 0.10),
    Category.SOCIAL: (0.50, 0.35, 0.15),
    Category.OTHER: (0.60, 0.35, 0.05),
}

COMPLETION = {Effort.S: 0.95, Effort.M: 0.85, Effort.L: 0.65}


@dataclass(frozen=True)
class TicketPlan:
    week: int
    category: Category
    text_effort: Effort  # what the text is written for
    true_effort: Effort  # the truth (differs only under concept drift)
    submitter: Person
    style: str
    events: tuple[str, ...] = ()


@dataclass
class WeekState:
    week: int
    monday: date
    spec: FamilySpec
    active: list[Event] = field(default_factory=list)

    @property
    def month(self) -> int:
        return (self.monday + timedelta(days=3)).month  # the week's Thursday decides


def state_at(spec: FamilySpec, scenario: Scenario, start: date, week: int) -> WeekState:
    """The world in a given week: the family spec after pets arrived, active events."""
    active = [e for e in scenario.events if e.week <= week]
    spec = spec.model_copy(deep=True)
    for event in active:
        if event.kind == "add_pet" and event.species == "dog":
            spec.dogs += 1
        elif event.kind == "add_pet" and event.species == "cat":
            spec.cats += 1
    return WeekState(week=week, monday=start + timedelta(weeks=week), spec=spec, active=active)


def category_weights_at(state: WeekState) -> dict[Category, float]:
    weights = category_weights(state.spec)
    for category, factor in SEASONS.get(state.month, {}).items():
        weights[category] *= factor
    for event in state.active:
        if event.kind == "category_boost" and event.category and event.factor:
            weights[event.category] *= event.factor
    return weights


def effort_shift(state: WeekState, category: Category) -> tuple[float, float, float] | None:
    """The shifted TRUE effort distribution for `category`, or None when no concept
    drift is active (then the truth is simply the effort the text was written for).
    Returning None instead of comparing float tuples: 0.6 + 0.3 + 0.1 != 1.0 in floating
    point, so an equality check would silently decouple truth from text every week."""
    dist = None
    for event in state.active:
        if event.kind == "effort_shift" and event.category == category and event.effort:
            dist = tuple(event.effort.get(e, 0.0) for e in EFFORTS)
    if dist is None:
        return None
    total = sum(dist)
    return tuple(p / total for p in dist)


def _typists(state: WeekState, world: FamilyWorld) -> tuple[list[Person], list[float]]:
    """Who types tickets this week. A `new_typist` event lets the oldest child who
    didn't type yet start (short, sloppy tickets: covariate drift)."""
    people = list(world.submitters)
    weights = [3.0 if p.role == "adult" else 1.0 for p in people]
    if any(e.kind == "new_typist" for e in state.active):
        newcomer = max(
            (p for p in world.people if p not in people), key=lambda p: p.age or 0, default=None
        )
        if newcomer is not None:
            people.append(newcomer)
            weights.append(2.5)  # an enthusiastic new typist
    return people, weights


def _style(person: Person, state: WeekState, rng: np.random.Generator) -> str:
    names = list(STYLES)
    weights = np.array([STYLES[n][1] for n in names])
    if person.role == "kid":
        weights = weights * np.array([1.5 if n in ("terse", "sloppy") else 0.6 for n in names])
        if any(e.kind == "new_typist" for e in state.active) and (person.age or 99) < 9:
            weights = weights * np.array([3.0 if n == "sloppy" else 1.0 for n in names])
    style = str(rng.choice(names, p=weights / weights.sum()))
    return "terse" if person.role == "kid" and style == "detailed" else style


def plan_week(state: WeekState, world: FamilyWorld, rng: np.random.Generator) -> list[TicketPlan]:
    """The week's tickets with their hidden truth."""
    members = len(state.spec.adults) + len(state.spec.kids)
    n = int(rng.poisson(8 + 1.5 * members))
    weights = category_weights_at(state)
    categories = [c for c, w in weights.items() if w > 0]
    probs = np.array([weights[c] for c in categories])
    typists, typist_weights = _typists(state, world)
    tw = np.array(typist_weights) / sum(typist_weights)
    tags = tuple(f"{e.kind}@{e.week}" for e in state.active)
    plans = []
    for _ in range(n):
        category = categories[rng.choice(len(categories), p=probs / probs.sum())]
        text_effort = EFFORTS[rng.choice(3, p=EFFORT_BASE[category])]
        shifted = effort_shift(state, category)
        true_effort = text_effort if shifted is None else EFFORTS[rng.choice(3, p=shifted)]
        submitter = typists[rng.choice(len(typists), p=tw)]
        plans.append(
            TicketPlan(
                state.week,
                category,
                text_effort,
                true_effort,
                submitter,
                _style(submitter, state, rng),
                tags,
            )
        )
    return plans


# --- Behaviour during the sprint -------------------------------------------------------------


def pick_assignee(category: Category, people: list[Person], rng: np.random.Generator) -> Person:
    """Adults do finance/maintenance/social; children help with chores and groceries."""

    def weight(p: Person) -> float:
        if p.role == "adult":
            return 3.0
        age = p.age or 0
        if age < 8:
            return 0.0
        if category in (Category.CHORES, Category.GROCERIES):
            return 1.5 if age >= 12 else 0.8
        if category == Category.KIDS:
            return 0.5 if age >= 12 else 0.0
        return 0.0

    weights = np.array([weight(p) for p in people])
    return people[rng.choice(len(people), p=weights / weights.sum())]


def completes(true_effort: Effort, assignee: Person, rng: np.random.Generator) -> bool:
    p = COMPLETION[true_effort] * (0.85 if assignee.role == "kid" else 1.0)
    return bool(rng.random() < p)


# --- Load work model (Phase 9: planner evaluation) -----------------------------------------
# Hidden from every planner: they only see what the app sees (history of who finished what).

HOURS = {Effort.S: 0.5, Effort.M: 1.5, Effort.L: 3.0}
OVERLOAD_FACTOR = 0.3  # a task that doesn't fit in the remaining week rarely gets done
UNSUITED_FACTOR = 0.4  # e.g. a 9-year-old doing the tax return


def weekly_hours(person: Person) -> float:
    """Time a person has for household tasks in a week."""
    if person.role == "adult":
        return 5.0
    age = person.age or 0
    return 2.5 if age >= 12 else 1.0 if age >= 8 else 0.0


def suited(category: Category, person: Person) -> bool:
    """The same rules as `pick_assignee`: who can sensibly do this kind of task."""
    if person.role == "adult":
        return True
    age = person.age or 0
    if age < 8:
        return False
    if category in (Category.CHORES, Category.GROCERIES):
        return True
    return category == Category.KIDS and age >= 12


def completion_probability(
    true_effort: Effort, category: Category, assignee: Person, hours_before: float
) -> float:
    """Load model: a person works through their tasks in board order; tasks beyond
    their weekly hours, or unsuited to them, are much less likely to get done."""
    budget = weekly_hours(assignee)
    if budget == 0:
        return 0.0
    p = COMPLETION[true_effort] * (0.85 if assignee.role == "kid" else 1.0)
    if hours_before + HOURS[true_effort] > budget:
        p *= OVERLOAD_FACTOR
    if not suited(category, assignee):
        p *= UNSUITED_FACTOR
    return p


def reported_effort(true_effort: Effort, rng: np.random.Generator) -> Effort:
    """What the reviewer reports: the truth, off by one step 10% of the time."""
    if rng.random() >= 0.1:
        return true_effort
    i = EFFORTS.index(true_effort)
    return EFFORTS[min(2, i + 1)] if i == 0 or (i == 1 and rng.random() < 0.5) else EFFORTS[i - 1]


# --- Preset scenarios and family ---------------------------------------------------------------

SIM_FAMILY = FamilySpec(
    adults=[{"gender": "male"}, {"gender": "female"}],
    kids=[
        KidSpec(gender="female", age=5),
        KidSpec(gender="male", age=8),
        KidSpec(gender="female", age=12),
    ],
    housing="house",
    garden=True,
    cars=1,
)

SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario(
        name="baseline",
        description="Seasons only: summer garden work, school in September, December gifts.",
    ),
    "drift-demo": Scenario(
        name="drift-demo",
        description="Seasons plus three planted changes, one per drift type.",
        events=[
            Event(
                week=8,
                kind="add_pet",
                species="dog",
                note="The family gets a dog (prior + covariate drift)",
            ),
            Event(
                week=12,
                kind="effort_shift",
                category=Category.GROCERIES,
                effort={Effort.S: 0.15, Effort.M: 0.7, Effort.L: 0.15},
                note="Moved house: the shop is far away, groceries take longer (concept drift)",
            ),
            Event(
                week=16,
                kind="new_typist",
                note="The 8-year-old starts typing tickets (covariate drift)",
            ),
        ],
    ),
}
