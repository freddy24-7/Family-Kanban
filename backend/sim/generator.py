"""Synthetic ticket generation with Gemini (weak labels).

For each batch we first decide, with a seeded RNG, *what* to ask for: per ticket a
category (from the family's category weights), a submitter and a writing style.
Gemini then writes the Dutch text and labels it using the labelling guidelines.
Every returned item is validated on its own; invalid items and duplicates are
counted and dropped, never allowed to break the batch.
"""

import random
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from app import llm
from app.domain import Category, Effort
from ml.text import normalise_text
from sim.family import FamilyWorld, Person, category_weights

# Bump when the prompt changes meaningfully: it is stored on every GenerationRun,
# so datasets from different prompt versions can be told apart.
PROMPT_VERSION = "gen-v2"  # v2: no invented pets (EDA finding 6)
BATCH_SIZE = 25

GUIDELINES_PATH = Path(__file__).resolve().parents[1] / "app" / "labeling_guidelines.md"

# Writing styles and how often they occur. Real users mostly type very short,
# informal notes; LLMs default to neat full sentences, so we ask explicitly.
STYLES: dict[str, tuple[str, float]] = {
    "terse": ("2-4 words, lowercase, no punctuation (like 'gras maaien')", 0.35),
    "short": ("a short phrase or sentence with normal spelling", 0.28),
    "sloppy": ("lowercase with a typo or abbreviation, maybe '!!' or 'ff'", 0.15),
    "detailed": ("one longer sentence with a concrete detail (who, when or where)", 0.15),
    "reminder": (
        "phrased as a question or reminder to the family ('wie ...?', 'niet vergeten: ...')",
        0.07,
    ),
}
MONTHS_NL = ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus",
             "september", "oktober", "november", "december"]  # fmt: skip


@dataclass(frozen=True)
class TicketRequest:
    index: int
    category: Category
    submitter: Person
    style: str


class GeneratedTicket(BaseModel):
    """One validated item from Gemini (untrusted input)."""

    index: int
    text: str = Field(min_length=3, max_length=300)
    category: Category
    effort: Effort

    @field_validator("text", mode="before")
    @classmethod
    def _strip_control_characters(cls, value: object) -> object:
        # LLM output occasionally contains control characters; Postgres rejects
        # NUL bytes outright. Remove them before length validation.
        if isinstance(value, str):
            value = "".join(c for c in value if unicodedata.category(c) != "Cc" or c in "\n\t")
            return " ".join(value.split())
        return value


@dataclass
class AcceptedTicket:
    text: str
    category: Category
    effort: Effort
    requested_category: Category
    submitter: Person


@dataclass
class BatchResult:
    accepted: list[AcceptedTicket] = field(default_factory=list)
    rejected_invalid: int = 0
    rejected_duplicate: int = 0
    category_mismatches: int = 0
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0


def plan_batch(
    world: FamilyWorld, size: int, rng: random.Random, start_index: int = 0
) -> list[TicketRequest]:
    weights = category_weights(world.spec)
    categories = [c for c, w in weights.items() if w > 0]
    style_names = list(STYLES)
    style_weights = [STYLES[s][1] for s in style_names]
    # Adults submit most tickets; older kids some.
    submitters = world.submitters
    submitter_weights = [3.0 if p.role == "adult" else 1.0 for p in submitters]
    requests = []
    for i in range(size):
        submitter = rng.choices(submitters, submitter_weights)[0]
        style = rng.choices(style_names, style_weights)[0]
        if submitter.role == "kid" and style == "detailed":
            style = "terse"  # kids rarely write long, careful sentences
        requests.append(
            TicketRequest(
                index=start_index + i,
                category=rng.choices(categories, [weights[c] for c in categories])[0],
                submitter=submitter,
                style=style,
            )
        )
    return requests


def _guidelines_for_prompt() -> str:
    text = GUIDELINES_PATH.read_text(encoding="utf-8")
    return text.split("## Change log")[0].strip()


SYSTEM_PROMPT = f"""You generate realistic synthetic data for a Dutch family task app.
Family members type short notes (tickets) on their phone about things that need doing.
Write every ticket in natural, everyday Dutch as that person would type it.

Label every ticket you write with exactly one category and one effort level,
strictly following these guidelines:

{_guidelines_for_prompt()}
"""

ITEM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "index": {"type": "INTEGER"},
        "text": {"type": "STRING"},
        "category": {"type": "STRING", "enum": [c.value for c in Category]},
        "effort": {"type": "STRING", "enum": [e.value for e in Effort]},
    },
    "required": ["index", "text", "category", "effort"],
}


def build_prompt(
    world: FamilyWorld, requests: list[TicketRequest], now: datetime, avoid: list[str]
) -> str:
    lines = [
        world.describe(),
        "",
        f"It is currently {MONTHS_NL[now.month - 1]}; make tickets fit the season where natural.",
        "",
        "Write one ticket per request below. Each ticket should be about the requested",
        "category, but set `category` to the label the guidelines give for the text you",
        "actually wrote. Estimate `effort` per the guidelines. Return the same `index`.",
        "Use the family's real names and pets where natural. Vary wording and topics:",
        "no two tickets should be near-identical.",
        "Only mention people, pets and things this family actually has (listed above);",
        "never invent other pets, vehicles, rooms or family members.",
        "",
        "Requests:",
    ]
    for r in requests:
        who = r.submitter.name + (f" (age {r.submitter.age})" if r.submitter.role == "kid" else "")
        lines.append(
            f"{r.index}. category={r.category.value}; typed by {who}; style: {STYLES[r.style][0]}"
        )
    if avoid:
        lines += ["", "Already written for this family (do not repeat these):"]
        lines += [f"- {t}" for t in avoid[-40:]]
    return "\n".join(lines)


async def generate_batch(
    world: FamilyWorld, requests: list[TicketRequest], now: datetime, seen_normalised: set[str]
) -> BatchResult:
    """Ask Gemini for one batch and validate each item. `seen_normalised` holds the
    normalised texts already accepted for this family; it is updated in place."""
    avoid = sorted(seen_normalised)
    result = await llm.generate_json_list(
        SYSTEM_PROMPT, build_prompt(world, requests, now, avoid), ITEM_SCHEMA
    )
    batch = BatchResult(
        model=result.model, tokens_in=result.tokens_in, tokens_out=result.tokens_out
    )
    by_index = {r.index: r for r in requests}
    returned = set()
    for raw in result.items:
        try:
            ticket = GeneratedTicket.model_validate(raw)
        except ValidationError:
            continue  # counted below as a request without a valid answer
        request = by_index.get(ticket.index)
        if request is None or ticket.index in returned:
            continue  # unknown or repeated index: ignore the extra item
        returned.add(ticket.index)
        key = normalise_text(ticket.text)
        if not key or key in seen_normalised:
            batch.rejected_duplicate += 1
            continue
        seen_normalised.add(key)
        if ticket.category != request.category:
            batch.category_mismatches += 1
        batch.accepted.append(
            AcceptedTicket(
                text=ticket.text.strip(),
                category=ticket.category,
                effort=ticket.effort,
                requested_category=request.category,
                submitter=request.submitter,
            )
        )
    # Every request without exactly one valid answer (malformed, wrong label value,
    # or silently skipped by Gemini) counts as invalid.
    batch.rejected_invalid = len(requests) - len(returned)
    return batch
