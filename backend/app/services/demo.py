"""Demo families: create a simulated household from a FamilySpec, and fill it with
Gemini-generated tickets that carry weak labels (label_source=generator)."""

import asyncio
import logging
import random
import secrets
import uuid

from fastapi_users.password import PasswordHelper
from sqlalchemy.ext.asyncio import AsyncSession

from app import config, llm, repository
from app.clock import Clock
from app.db import SessionFactory
from app.domain import LabelSource, Source
from app.models import FamilyProfile, GenerationRun, Household, Membership, Topic, User
from app.services.classification import classify_topics
from ml.text import normalise_text
from sim.family import FamilySpec, build_world
from sim.generator import BATCH_SIZE, PROMPT_VERSION, generate_batch, plan_batch

log = logging.getLogger(__name__)
_password_helper = PasswordHelper()

# Stop a run if Gemini keeps returning only invalid or duplicate tickets.
MAX_EMPTY_BATCHES = 3


async def create_demo_family(
    session: AsyncSession,
    spec: FamilySpec,
    name: str | None = None,
    preset_key: str | None = None,
    training_eligible: bool = False,
    random_seed: int | None = None,
) -> Household:
    """Simulated household + one simulated (login-disabled) user per family member.
    The seed makes names reproducible: the same spec + seed gives the same family."""
    seed = random_seed if random_seed is not None else secrets.randbelow(2**31)
    world = build_world(spec, random.Random(seed))
    household = Household(
        id=uuid.uuid4(),
        name=name or f"Familie {world.surname} (demo)",
        kind=Source.SIMULATED,
        training_eligible=training_eligible,
    )
    session.add(household)
    # Plain foreign keys (without relationship()) don't order inserts, so make sure
    # the household row exists before rows that reference it.
    await session.flush()
    session.add(
        FamilyProfile(
            household_id=household.id,
            spec=spec.model_dump(),
            preset_key=preset_key,
            random_seed=seed,
        )
    )
    for person in world.people:
        user = User(
            id=uuid.uuid4(),
            email=f"sim-{uuid.uuid4().hex[:12]}@simulated.invalid",
            # A real hash of a random, discarded password: login is impossible,
            # and is_active=False blocks it a second time.
            hashed_password=_password_helper.hash(secrets.token_urlsafe(32)),
            is_active=False,
            is_verified=False,
            display_name=person.name,
            is_simulated=True,
        )
        session.add(user)
        session.add(
            Membership(
                user_id=user.id,
                household_id=household.id,
                is_planner=person.role == "adult",
                is_reviewer=person.role == "adult",
                profile=person.model_dump(),
            )
        )
    await session.commit()
    return household


async def generate_tickets(
    session: AsyncSession,
    household: Household,
    count: int,
    clock: Clock,
    random_seed: int | None = None,
) -> GenerationRun:
    """Generate `count` tickets in one go (CLI / tests)."""
    run = await start_generation(session, household, count, random_seed)
    return await continue_generation(session, household, run, clock)


async def start_generation(
    session: AsyncSession, household: Household, count: int, random_seed: int | None = None
) -> GenerationRun:
    """Validate and record a new generation run (status=running)."""
    if household.kind != Source.SIMULATED:
        raise ValueError("Tickets can only be generated for simulated households")
    profile = await repository.get_family_profile(session, household.id)
    if profile is None:
        raise ValueError("Household has no family profile")

    run = GenerationRun(
        household_id=household.id,
        status="running",
        prompt_version=PROMPT_VERSION,
        random_seed=random_seed if random_seed is not None else secrets.randbelow(2**31),
        requested=count,
        models_used={},
    )
    session.add(run)
    await session.commit()
    return run


async def continue_generation(
    session: AsyncSession, household: Household, run: GenerationRun, clock: Clock
) -> GenerationRun:
    """Generate tickets in batches until run.requested is reached. Each batch is
    committed on its own, so a Gemini failure halfway keeps the work already done
    (status=failed + error)."""
    profile = await repository.get_family_profile(session, household.id)
    spec = FamilySpec.model_validate(profile.spec)
    world = build_world(spec, random.Random(profile.random_seed))
    members = await repository.list_members(session, household.id)
    user_by_name = {m.user.display_name: m.user_id for m in members}
    rng = random.Random(run.random_seed)
    count = run.requested

    seen = {normalise_text(t) for t in await repository.list_topic_texts(session, household.id)}
    next_index = 0  # request numbers are unique within the run
    empty_batches_in_a_row = 0
    try:
        while run.produced < count:
            if run.tokens_in + run.tokens_out >= config.LLM_MAX_TOKENS_PER_RUN:
                raise RuntimeError(f"Token cap reached ({config.LLM_MAX_TOKENS_PER_RUN})")
            size = min(BATCH_SIZE, count - run.produced)
            requests = plan_batch(world, size, rng, start_index=next_index)
            next_index += size
            batch = await generate_batch(world, requests, clock.now(), seen)

            topics = [
                Topic(
                    id=uuid.uuid4(),
                    household_id=household.id,
                    text=ticket.text,
                    created_by=user_by_name.get(ticket.submitter.name),
                    occurred_at=clock.now(),
                    source=Source.SIMULATED,
                    category_label=ticket.category,
                    effort_label=ticket.effort,
                    label_source=LabelSource.GENERATOR,
                    labeled_at=clock.now(),
                    generation_run_id=run.id,
                )
                for ticket in batch.accepted
            ]
            session.add_all(topics)
            await session.flush()
            await classify_topics(session, topics, clock)

            run.produced += len(topics)
            run.rejected_invalid += batch.rejected_invalid
            run.rejected_duplicate += batch.rejected_duplicate
            run.category_mismatches += batch.category_mismatches
            run.tokens_in += batch.tokens_in
            run.tokens_out += batch.tokens_out
            run.models_used = {
                **run.models_used,
                batch.model: run.models_used.get(batch.model, 0) + 1,
            }
            await session.commit()
            log.info("Run %s: %d/%d tickets", run.id, run.produced, count)
            empty_batches_in_a_row = 0 if topics else empty_batches_in_a_row + 1
            if empty_batches_in_a_row >= MAX_EMPTY_BATCHES:
                raise RuntimeError(f"{MAX_EMPTY_BATCHES} batches in a row gave no usable tickets")
        run.status = "completed"
    except (llm.LLMUnavailable, RuntimeError) as exc:
        log.warning("Generation run %s failed: %s", run.id, exc)
        await _mark_failed(session, run, str(exc), clock)
        return run
    except asyncio.CancelledError:
        # Process stopped/cancelled mid-run: record it, then let cancellation proceed.
        await asyncio.shield(_mark_failed(session, run, "cancelled", clock))
        raise
    except Exception as exc:
        # Unexpected: record it so the run never stays "running", then re-raise.
        log.exception("Generation run %s crashed", run.id)
        await _mark_failed(session, run, f"{type(exc).__name__}: {exc}", clock)
        raise
    run.finished_at = clock.now()
    await session.commit()
    return run


async def _mark_failed(session: AsyncSession, run: GenerationRun, error: str, clock: Clock) -> None:
    """Record the failure through a FRESH session: after a database error the
    original session may be unusable. Batches committed earlier are kept."""
    run_id, finished_at = run.id, clock.now()
    try:
        await session.rollback()  # discard the half-finished batch
    except Exception:
        log.exception("Rollback failed while recording run %s failure", run_id)
    async with SessionFactory() as fresh:
        stored = await fresh.get(GenerationRun, run_id)
        stored.status, stored.error, stored.finished_at = "failed", error[:2000], finished_at
        await fresh.commit()
    try:
        await session.refresh(run)  # let the caller see the stored state
    except Exception:
        run.status, run.error, run.finished_at = "failed", error[:2000], finished_at
