"""Replays a simulation week by week through the REAL app services (the same code paths
a family uses), on a simulated clock. Deterministic: same pool + same seed = same run.

Each week:
  Mon 08:00  plan a sprint from the backlog (oldest first, up to capacity) and start it
  Mon-Sun    the week's pool tickets arrive -> the active models classify them
  Sat 12:00  sprint work happens (completion depends on true effort and assignee)
  Sun 20:00  sprint review (actual effort, slightly noisy) and completion
  Sun 21:00  the planner labels new backlog tickets (sometimes rubber-stamping)
  then the week's statistics are recorded against the hidden truth.
"""

import logging
import random
from collections import Counter
from datetime import UTC, date, datetime, time, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import config, repository
from app.clock import SimulatedClock
from app.domain import Effort, ItemStatus, LabelSource, Task
from app.models import Household, SimTicket, SimulationRun, Topic
from app.services import planner_assistant
from app.services import sprints as sprint_service
from app.services import topics as topic_service
from app.tenancy import HouseholdAccess
from sim.family import FamilySpec, Person, build_world
from sim.world import (
    EFFORTS,
    HOURS,
    PlannerBehaviour,
    Scenario,
    completes,
    completion_probability,
    pick_assignee,
    reported_effort,
    state_at,
    weekly_hours,
)

log = logging.getLogger(__name__)


class Purpose:
    """Keys of the independent random streams in the load work model."""

    INTAKE, STARTED, REPORTED, LABEL = 1, 2, 3, 4


def _at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour, minute), UTC)


class Simulation:
    def __init__(
        self,
        session: AsyncSession,
        run: SimulationRun,
        household: Household,
        spec: FamilySpec,
        family_seed: int,
    ):
        self.session, self.run, self.household = session, run, household
        self.scenario = Scenario.model_validate(run.scenario)
        self.planner_cfg = PlannerBehaviour.model_validate(run.planner)
        self.spec, self.family_seed = spec, family_seed
        # A separate stream from the pool's, derived from the run's own seed.
        self.rng = np.random.default_rng([run.random_seed, 1])
        self.clock = SimulatedClock(_at(run.start_date, 8))
        self.truth: dict = {}  # topic_id -> SimTicket
        self.planning: dict = {}  # this week's planner outcome, for the stats
        self.overloaded = 0

    async def setup(self) -> None:
        members = await repository.list_members(self.session, self.household.id)
        self.users = {m.user.display_name: m.user for m in members}
        planner = next(m for m in members if m.is_planner)
        self.access = HouseholdAccess(self.household, planner.user, True, True, is_member=True)
        # Shadow evaluation: the run may pin model versions instead of the active ones.
        self.versions = (
            await repository.model_versions_by_name(self.session, self.run.model_versions)
            if self.run.model_versions
            else None
        )
        pool_id = self.run.pool_run_id or self.run.id
        self.pool = (
            await self.session.scalars(
                select(SimTicket)
                .where(SimTicket.simulation_run_id == pool_id)
                .order_by(SimTicket.week, SimTicket.id)
            )
        ).all()
        # Resume support: topics already created by this run.
        rows = await self.session.execute(
            select(Topic.id, Topic.sim_ticket_id).where(
                Topic.household_id == self.household.id, Topic.sim_ticket_id.is_not(None)
            )
        )
        by_id = {t.id: t for t in self.pool}
        self.truth = {
            topic_id: by_id[ticket_id] for topic_id, ticket_id in rows if ticket_id in by_id
        }

    def people(self, week: int) -> list[Person]:
        state = state_at(self.spec, self.scenario, self.run.start_date, week)
        return build_world(state.spec, random.Random(self.family_seed)).people

    async def run_week(self, week: int) -> dict:
        monday = self.run.start_date + timedelta(weeks=week)
        sprint = await self._plan_sprint(week, monday)
        week_topics = await self._intake(week, monday)
        if sprint is not None:
            await self._work_and_review(sprint, monday)
        await self._label_backlog(monday)
        return await self._stats(week, week_topics, sprint)

    async def _plan_sprint(self, week: int, monday: date):
        self.clock.set(_at(monday, 8))
        backlog = list(
            reversed(await repository.list_backlog(self.session, self.household.id, 500, 0))
        )
        people = [p for p in self.people(week) if p.role == "adult" or (p.age or 0) >= 8]
        backlog = [t for t in backlog if t.id in self.truth][
            : self.planner_cfg.capacity_per_person * len(people)
        ]
        if self.planner_cfg.policy != "sim":
            return await self._plan_with_assistant(week, monday)
        if not backlog:
            return None
        sprint = await sprint_service.create_sprint(
            self.session, self.household, f"Sim week {week + 1}", monday, monday + timedelta(days=6)
        )
        for topic in backlog:
            assignee = pick_assignee(self.truth[topic.id].true_category, people, self.rng)
            await sprint_service.add_item(
                self.session, sprint, topic.id, self.users[assignee.name].id
            )
        return await sprint_service.start_sprint(self.session, sprint, self.clock)

    async def _plan_with_assistant(self, week: int, monday: date):
        """Phase 9: the Planner Assistant (rules or Gemini) plans, through the same
        service the app's "Voorstel maken" button uses."""
        method = self.planner_cfg.policy
        proposal, _ = await planner_assistant.propose(
            self.session, self.household, self.clock, self.planner_cfg.max_items, method
        )
        self.planning = {
            "method": proposal.method,
            "model": proposal.model,
            "fallback": proposal.method != method,
            "tokens": proposal.tokens,
        }
        if not proposal.items:
            return None
        sprint = await sprint_service.create_sprint(
            self.session, self.household, f"Sim week {week + 1}", monday, monday + timedelta(days=6)
        )
        for item in proposal.items:
            await sprint_service.add_item(self.session, sprint, item.topic_id, item.user_id)
        return await sprint_service.start_sprint(self.session, sprint, self.clock)

    async def _intake(self, week: int, monday: date) -> list[Topic]:
        replayed = {ticket.id for ticket in self.truth.values()}  # resume: skip done ones
        tickets = [t for t in self.pool if t.week == week and t.id not in replayed]
        rng = self._rng(Purpose.INTAKE, None, monday)
        minutes = np.sort(rng.integers(60, 6 * 24 * 60, size=len(tickets)))  # Mon 09:00 .. Sun
        created = []
        for ticket, offset in zip(tickets, minutes, strict=True):
            self.clock.set(_at(monday, 8) + timedelta(minutes=int(offset)))
            submitter = self.users.get(ticket.submitter)
            topic = await topic_service.create_topic(
                self.session,
                self.household,
                submitter.id if submitter else None,
                ticket.text,
                None,
                self.clock,
                self.versions,
                self.run.adaptation,
            )
            topic.sim_ticket_id = ticket.id
            await self.session.commit()
            self.truth[topic.id] = ticket
            created.append(topic)
        return created

    async def _work_and_review(self, sprint, monday: date) -> None:
        items = await repository.list_sprint_items(self.session, sprint.id)
        people = {p.name: p for p in self.people((monday - self.run.start_date).days // 7)}
        names = {u.id: name for name, u in self.users.items()}
        outcome = {}
        self.clock.set(_at(monday + timedelta(days=5), 12))
        hours_before: dict = {}
        self.overloaded = 0
        for item in items:  # board order: the planner's priority
            truth = self.truth[item.topic_id]
            assignee = people.get(names.get(item.assignee_id), next(iter(people.values())))
            if self.planner_cfg.work_model == "load":
                before = hours_before.get(assignee.name, 0.0)
                hours_before[assignee.name] = before + HOURS[truth.true_effort]
                self.overloaded += before + HOURS[truth.true_effort] > weekly_hours(assignee)
                p = completion_probability(truth.true_effort, truth.true_category, assignee, before)
                done = self._draw(truth, monday) < p
            else:
                done = completes(truth.true_effort, assignee, self.rng)
            outcome[item.id] = done
            if done or self._rng(Purpose.STARTED, truth, monday).random() < 0.5:
                await sprint_service.move_item(
                    self.session, self.access, sprint, item, ItemStatus.IN_PROGRESS, 0, self.clock
                )
            if done:
                await sprint_service.move_item(
                    self.session, self.access, sprint, item, ItemStatus.DONE, 0, self.clock
                )
        self.clock.set(_at(monday + timedelta(days=6), 20))
        for item in items:
            done = outcome[item.id]
            effort = (
                reported_effort(
                    self.truth[item.topic_id].true_effort,
                    self._rng(Purpose.REPORTED, self.truth[item.topic_id], monday),
                )
                if done
                else None
            )
            await sprint_service.review_item(
                self.session, self.access, sprint, item, done, effort, None, self.clock
            )
        await sprint_service.complete_sprint(self.session, self.access, sprint, None, self.clock)

    def _rng(self, purpose: int, ticket, monday: date) -> np.random.Generator:
        """Randomness for one event. Default work model: the run's single stream, exactly
        as before (earlier runs stay bit-identical). Load model (planner experiments): a
        stream keyed by ticket, week and purpose, so no random event depends on what a
        planner did earlier: two planners differ only by their decisions. (A first version
        keyed only the completion roll; labels and arrival times still drifted apart.)"""
        if self.planner_cfg.work_model != "load":
            return self.rng
        ticket_key = ticket.id.int % 2**63 if ticket is not None else 0
        return np.random.default_rng(
            [self.run.random_seed, ticket_key, monday.toordinal(), purpose]
        )

    def _draw(self, ticket, monday: date) -> float:
        """Common random numbers: a ticket's luck in a given week is fixed by the run
        seed, whoever planned it. Two planners then differ only by their decisions, not
        by the dice (far less noise in the comparison)."""
        seed = [self.run.random_seed, ticket.id.int % 2**63, monday.toordinal()]
        return float(np.random.default_rng(seed).random())

    async def _label_backlog(self, monday: date) -> None:
        """The planner confirms or corrects new tickets. Our UI pre-fills only CONFIDENT
        predictions, so rubber-stamping can only happen on those; uncertain ones are
        always judged (correctly, apart from occasional honest mistakes)."""
        self.clock.set(_at(monday + timedelta(days=6), 21))
        cfg = self.planner_cfg
        backlog = await repository.list_backlog(self.session, self.household.id, 500, 0)
        todo = [t for t in backlog if t.category_label is None and t.id in self.truth]
        predictions = await repository.latest_predictions(
            self.session, self.household.id, [t.id for t in todo]
        )
        threshold = config.LOW_CONFIDENCE_THRESHOLD
        for topic in todo:
            rng = self._rng(Purpose.LABEL, self.truth[topic.id], monday)
            if rng.random() >= cfg.check_rate:
                continue  # not looked at this week
            truth, preds = self.truth[topic.id], predictions.get(topic.id, {})
            cat, eff = preds.get(Task.CATEGORY), preds.get(Task.EFFORT)
            confident = cat and eff and min(cat.confidence, eff.confidence) >= threshold
            if confident and rng.random() < cfg.rubber_stamp_rate:
                category, effort = cat.predicted, eff.predicted  # accepted unchecked
            else:
                category, effort = truth.true_category, truth.text_effort
                if rng.random() < cfg.label_error_rate:
                    others = [c for c in type(truth.true_category) if c != truth.true_category]
                    category = others[rng.integers(len(others))]
            await topic_service.set_labels(
                self.session,
                topic,
                category,
                Effort(effort),
                self.access.user.id,
                LabelSource.PLANNER,
                self.clock,
            )

    async def _stats(self, week: int, topics: list[Topic], sprint) -> dict:
        ids = [t.id for t in topics]
        preds = await repository.latest_predictions(self.session, self.household.id, ids)
        rows = []
        for topic in topics:
            await self.session.refresh(topic)
            p, truth = preds.get(topic.id, {}), self.truth[topic.id]
            cat, eff = p.get(Task.CATEGORY), p.get(Task.EFFORT)
            rows.append((topic, truth, cat, eff))
        n = len(rows)
        backlog = await repository.list_backlog(self.session, self.household.id, 500, 0)
        mean = lambda xs: round(float(np.mean(xs)), 4) if xs else None  # noqa: E731
        labelled = [(t, c, e) for t, _, c, e in rows if t.category_label is not None and c and e]
        items = await repository.list_sprint_items(self.session, sprint.id) if sprint else []
        return {
            "week": week,
            "monday": (self.run.start_date + timedelta(weeks=week)).isoformat(),
            "events": [f"{e.kind}: {e.note}" for e in self.scenario.events if e.week == week],
            "n": n,
            "true_category_acc": mean(
                [c is not None and c.predicted == s.true_category for _, s, c, _ in rows]
            ),
            "true_effort_acc": mean(
                [e is not None and e.predicted == s.true_effort for _, s, _, e in rows]
            ),
            "effort_mae": mean(
                [
                    abs(EFFORTS.index(Effort(e.predicted)) - EFFORTS.index(s.true_effort))
                    for _, s, _, e in rows
                    if e
                ]
            ),
            "measured_category_acc": mean(
                [c.predicted == t.category_label for t, c, _ in labelled]
            ),
            "mean_category_conf": mean([c.confidence for _, _, c, _ in rows if c]),
            "mean_effort_conf": mean([e.confidence for _, _, _, e in rows if e]),
            "flagged_rate": mean(
                [
                    not (
                        c
                        and e
                        and min(c.confidence, e.confidence) >= config.LOW_CONFIDENCE_THRESHOLD
                    )
                    for _, _, c, e in rows
                ]
            ),
            "mean_words": mean([len(t.text.split()) for t, *_ in rows]),
            "true_mix": dict(Counter(str(s.true_category) for _, s, _, _ in rows)),
            "predicted_mix": dict(Counter(c.predicted for _, _, c, _ in rows if c)),
            "labelled": len(labelled),
            "sprint_items": len(items),
            "sprint_done": sum(1 for i in items if i.completed),
            # Planner experiment (Phase 9): true hours of finished work, waiting time.
            "planned_hours": sum(HOURS[self.truth[i.topic_id].true_effort] for i in items),
            "done_hours": sum(
                HOURS[self.truth[i.topic_id].true_effort] for i in items if i.completed
            ),
            "overloaded_items": self.overloaded if items else 0,
            "done_wait_days": mean(
                [
                    (i.reviewed_at - i.topic.occurred_at).days
                    for i in items
                    if i.completed and i.reviewed_at
                ]
            ),
            "backlog_size": len(backlog),
            # Survivorship guard for done_wait_days: a planner that keeps skipping old
            # tasks shows a short wait for finished ones but an old open backlog.
            "oldest_open_days": max(
                ((self.clock.now() - t.occurred_at).days for t in backlog), default=0
            ),
            "planner": self.planning or None,
        }


async def run_simulation(session: AsyncSession, run: SimulationRun) -> SimulationRun:
    household = await repository.get_household(session, run.household_id)
    profile = await repository.get_family_profile(session, household.id)
    sim = Simulation(
        session, run, household, FamilySpec.model_validate(profile.spec), profile.random_seed
    )
    await sim.setup()
    run.status = "running"
    await session.commit()
    try:
        for week in range(run.current_week, run.weeks):
            stats = await sim.run_week(week)
            run.weekly_stats = [*run.weekly_stats, stats]
            run.current_week = week + 1
            await session.commit()
            log.info("Simulation %s week %d: %s", run.id, week, stats)
        run.status = "completed"
        run.finished_at = sim.clock.now()
    except Exception as exc:
        log.exception("Simulation %s failed", run.id)
        await session.rollback()
        run.status, run.error = "failed", f"{type(exc).__name__}: {exc}"[:2000]
        raise
    finally:
        await session.commit()
    return run
