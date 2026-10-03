"""Planner Assistant (Phase 9): proposes a draft sprint, which tasks and who does them,
with a short reason. The planner accepts or ignores it; nothing is changed here.

Two planners get exactly the same context (`build_context`), so they can be compared
fairly in simulation (docs/learning/08-llm-planner.md):
- `rule_plan`: a transparent baseline, what a careful person would do by hand;
- `llm_plan`: Gemini, given the same facts as JSON.

Privacy: Gemini never sees members' names. Members are "lid 1", "lid 2", ...; their
names inside task texts (full display name and first name) are replaced by role tokens
(the classifier's masking function, ml/text.py). Other names in a text ("oma Riet")
are not recognised and do go to Gemini.
The reasons are mapped back to names before they're shown.
"""

import json
import logging
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app import llm, repository
from app.clock import Clock
from app.domain import Effort, Task
from app.models import Household
from ml.text import mask_member_names

log = logging.getLogger(__name__)

HOURS = {"S": 0.5, "M": 1.5, "L": 3.0}  # the effort scale as shown in the app
CANDIDATES = 40  # backlog tasks offered to a planner (oldest / most urgent first)
HISTORY_WEEKS = 4
# Rule planner's prior capacity for members without history (hours per week).
DEFAULT_HOURS = {False: 3.0, True: 1.0}  # key: is_child
MIN_CATEGORY_RATE = 0.5  # rule planner: skip members who usually don't finish a category


@dataclass
class MemberContext:
    ref: str  # "lid 1": what Gemini sees instead of the name
    user_id: uuid.UUID
    name: str
    is_child: bool
    planned: int = 0  # items in reviewed sprints of the last HISTORY_WEEKS
    done: int = 0
    done_hours_per_week: float = 0.0
    by_category: dict[str, list[int]] = field(default_factory=dict)  # category -> [done, total]

    def completion_rate(self, category: str) -> float | None:
        done, total = self.by_category.get(category, (0, 0))
        return done / total if total else None


@dataclass
class TaskContext:
    ref: str  # "t1"
    topic_id: uuid.UUID
    text: str  # role-masked
    category: str | None
    effort: str
    age_days: int
    due_by: str | None


@dataclass
class PlanningContext:
    tasks: list[TaskContext]
    members: list[MemberContext]
    max_items: int


@dataclass
class ProposedItem:
    topic_id: uuid.UUID
    user_id: uuid.UUID
    reason: str


@dataclass
class Proposal:
    method: str  # "rules" | "llm"
    items: list[ProposedItem]
    note: str | None = None
    model: str | None = None
    tokens: int = 0


async def build_context(
    session: AsyncSession, household: Household, clock: Clock, max_items: int
) -> PlanningContext:
    now = clock.now()
    # A stable order (adults first, then by name): members of a demo family join in one
    # transaction, so their join time can't break ties, and refs must not depend on luck.
    members = sorted(
        await repository.list_members(session, household.id),
        key=lambda m: (m.is_child, m.user.display_name),
    )
    # Full display names and first names ("Freddy Austli" -> also "Freddy").
    names = [(m.user.display_name, m.is_child) for m in members] + [
        (m.user.display_name.split()[0], m.is_child)
        for m in members
        if len(m.user.display_name.split()) > 1
    ]
    context_members = [
        MemberContext(f"lid {i}", m.user_id, m.user.display_name, m.is_child)
        for i, m in enumerate(members, start=1)
    ]
    by_user = {m.user_id: m for m in context_members}

    # Recent history: who planned what, and did it get done?
    reviews = await repository.reviews_since(
        session, household.id, now - timedelta(weeks=HISTORY_WEEKS)
    )
    backlog = list(await repository.list_backlog(session, household.id, 500, 0))
    backlog.sort(key=lambda t: (t.due_by is None, t.due_by or now.date(), t.occurred_at, t.text))
    candidates = backlog[:CANDIDATES]
    predictions = await repository.latest_predictions(
        session, household.id, [t.id for t in candidates] + [r.topic_id for r in reviews]
    )

    def category(topic) -> str | None:
        if topic.category_label:
            return str(topic.category_label)
        prediction = predictions.get(topic.id, {}).get(Task.CATEGORY)
        return prediction.predicted if prediction else None

    weeks = max(1, len({r.sprint_id for r in reviews}))
    hours_done: dict[uuid.UUID, float] = defaultdict(float)
    for review in reviews:
        member = by_user.get(review.assignee_id)
        if member is None:
            continue
        counts = member.by_category.setdefault(category(review.topic) or "other", [0, 0])
        counts[1] += 1
        member.planned += 1
        if review.completed:
            counts[0] += 1
            member.done += 1
            hours_done[member.user_id] += HOURS[str(review.effort_actual or Effort.M)]
    for member in context_members:
        member.done_hours_per_week = round(hours_done[member.user_id] / weeks, 2)

    tasks = []
    for i, topic in enumerate(candidates, start=1):
        effort_prediction = predictions.get(topic.id, {}).get(Task.EFFORT)
        effort = str(
            topic.effort_label or (effort_prediction.predicted if effort_prediction else "M")
        )
        tasks.append(
            TaskContext(
                ref=f"t{i}",
                topic_id=topic.id,
                text=mask_member_names(topic.text, names),
                category=category(topic),
                effort=effort,
                age_days=(now - topic.occurred_at).days,
                due_by=topic.due_by.isoformat() if topic.due_by else None,
            )
        )
    return PlanningContext(tasks, context_members, max_items)


# --- Baseline: transparent rules ------------------------------------------------------------


def estimated_capacity(member: MemberContext) -> float:
    """Hours per week the member can probably handle, from their own history: what they
    finished recently, plus a little extra if they finished nearly everything (otherwise
    an under-planned member never gets more)."""
    if member.planned == 0:
        return DEFAULT_HOURS[member.is_child]
    rate = member.done / member.planned
    return member.done_hours_per_week + (0.5 if rate >= 0.8 else 0.0)


def _may_do(member: MemberContext, category: str) -> bool:
    """No history in this category: allowed (that's how anyone starts). Otherwise only if
    they finished at least half of it. (A first version wrote `rate or 1.0`, which turned a
    0% rate into 100%: exactly the member who always fails got the task.)"""
    rate = member.completion_rate(category)
    return rate is None or rate >= MIN_CATEGORY_RATE


def rule_plan(ctx: PlanningContext) -> list[ProposedItem]:
    """Most urgent / oldest first; each task goes to the member with the most capacity
    left who fits it and doesn't usually fail this kind of task."""
    left = {m.user_id: estimated_capacity(m) for m in ctx.members}
    items: list[ProposedItem] = []
    for task in ctx.tasks:
        if len(items) >= ctx.max_items:
            break
        hours = HOURS[task.effort]
        options = [
            m
            for m in ctx.members
            if left[m.user_id] >= hours and _may_do(m, task.category or "other")
        ]
        if not options:
            continue
        member = max(options, key=lambda m: left[m.user_id])
        left[member.user_id] -= hours
        items.append(
            ProposedItem(
                task.topic_id,
                member.user_id,
                f"{member.ref} heeft deze week nog ruimte"
                + (
                    f"; deadline {task.due_by}"
                    if task.due_by
                    else f"; wacht al {task.age_days} dagen"
                ),
            )
        )
    return items


# --- Gemini ------------------------------------------------------------------------------

SYSTEM = """You plan one week of household tasks for a family (Dutch household app).
Choose which tasks go into this week's sprint and who does each one. Goals, in order:
1. tasks that get FINISHED this week (don't overload anyone; children have less time and
   can't do every kind of task; look at what each member actually finished recently),
2. urgent and long-waiting tasks first,
3. a fair spread of work.
Use only the task refs (t1, t2, ...) and member refs (lid 1, lid 2, ...) given. Plan at most
`max_items` tasks; fewer is fine if more wouldn't get done. For each task, give a short reason
in Dutch (max 12 words), referring to members only by their ref (e.g. "lid 2"). The reason
is shown to the family: describe kinds of tasks in plain Dutch ("boodschappen", "klussen"),
never as the category keys from the data (no "groceries", "home_maintenance")."""

ITEM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "task": {"type": "STRING"},
        "member": {"type": "STRING"},
        "reason": {"type": "STRING"},
    },
    "required": ["task", "member", "reason"],
}


def llm_prompt(ctx: PlanningContext) -> str:
    payload = {
        "max_items": ctx.max_items,
        "effort_hours": HOURS,
        "members": [
            {
                "ref": m.ref,
                "child": m.is_child,
                f"last_{HISTORY_WEEKS}_weeks": {
                    "planned": m.planned,
                    "finished": m.done,
                    "finished_hours_per_week": m.done_hours_per_week,
                    "by_category_finished_of_planned": m.by_category,
                },
            }
            for m in ctx.members
        ],
        "tasks": [
            {
                "ref": t.ref,
                "text": t.text,
                "category": t.category,
                "effort": t.effort,
                "waiting_days": t.age_days,
                "due": t.due_by,
            }
            for t in ctx.tasks
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def validate(raw_items: list, ctx: PlanningContext) -> list[ProposedItem]:
    """An LLM can invent refs, repeat tasks or ignore the limit: keep only what's valid."""
    tasks = {t.ref: t for t in ctx.tasks}
    members = {m.ref.lower(): m for m in ctx.members}
    items, seen = [], set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        task = tasks.get(str(raw.get("task", "")).strip())
        member = members.get(str(raw.get("member", "")).strip().lower())
        if task is None or member is None or task.ref in seen:
            continue
        seen.add(task.ref)
        items.append(ProposedItem(task.topic_id, member.user_id, str(raw.get("reason", ""))[:200]))
        if len(items) >= ctx.max_items:
            break
    return items


async def llm_plan(ctx: PlanningContext) -> tuple[list[ProposedItem], str, int]:
    result = await llm.generate_json_list(SYSTEM, llm_prompt(ctx), ITEM_SCHEMA, temperature=0.2)
    return validate(result.items, ctx), result.model, result.tokens_in + result.tokens_out


# --- Entry point ---------------------------------------------------------------------------


def with_names(reason: str, ctx: PlanningContext) -> str:
    """'lid 2 deed dit vorige week' -> 'Sem deed dit vorige week' (shown to the family only)."""
    refs = {m.ref.lower(): m.name for m in ctx.members}
    return re.sub(
        r"\blid \d+\b", lambda m: refs.get(m.group(0).lower(), m.group(0)), reason, flags=re.I
    )


async def propose(
    session: AsyncSession, household: Household, clock: Clock, max_items: int, method: str = "auto"
) -> tuple[Proposal, PlanningContext]:
    """method: "auto" (Gemini when configured, else rules), "llm" or "rules". If Gemini
    fails, the rule proposal is returned with a note: planning never depends on it."""
    ctx = await build_context(session, household, clock, max_items)
    if not ctx.tasks:
        return Proposal(method="rules", items=[]), ctx
    if method in ("auto", "llm") and llm.is_configured():
        try:
            items, model, tokens = await llm_plan(ctx)
            return Proposal("llm", items, model=model, tokens=tokens), ctx
        except Exception:
            log.exception("Planner assistant: Gemini failed; falling back to rules")
            return Proposal("rules", rule_plan(ctx), note="llm_failed"), ctx
    note = "llm_not_configured" if method in ("auto", "llm") else None
    return Proposal("rules", rule_plan(ctx), note=note), ctx
