"""Simulated family definitions: what a family consists of, the preset families
used for the seed dataset, and the family's "world" as the generator sees it.

The category weights here are the first piece of the simulator's hidden ground
truth: they decide how often each kind of task occurs in a given family, so a
family with a dog and a garden produces more chores and maintenance tickets."""

import random
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.domain import Category

Gender = Literal["male", "female"]


class AdultSpec(BaseModel):
    gender: Gender


class KidSpec(BaseModel):
    gender: Gender
    age: int = Field(ge=0, le=17)


class FamilySpec(BaseModel):
    """Input of the demo screen: who lives here and what the household has."""

    adults: list[AdultSpec] = Field(min_length=1, max_length=4)
    kids: list[KidSpec] = Field(default_factory=list, max_length=8)
    housing: Literal["house", "apartment"] = "house"
    garden: bool = False
    balcony: bool = False
    garage: bool = False
    dogs: int = Field(default=0, ge=0, le=5)
    cats: int = Field(default=0, ge=0, le=5)
    cars: int = Field(default=0, ge=0, le=4)
    bikes: bool = True

    @model_validator(mode="after")
    def _apartment_has_no_garden(self) -> "FamilySpec":
        if self.housing == "apartment" and self.garden:
            raise ValueError("An apartment has a balcony, not a garden")
        return self


def typical_family(kid_ages: tuple[int, int, int], kid_genders: tuple[Gender, ...]) -> FamilySpec:
    """The developer's reference family: father, mother, three kids, dog, house
    with garden, car."""
    return FamilySpec(
        adults=[AdultSpec(gender="male"), AdultSpec(gender="female")],
        kids=[KidSpec(gender=g, age=a) for g, a in zip(kid_genders, kid_ages, strict=True)],
        housing="house",
        garden=True,
        dogs=1,
        cars=1,
    )


def _adults(*genders: Gender) -> list[AdultSpec]:
    return [AdultSpec(gender=g) for g in genders]


def _kids(*kids: tuple[Gender, int]) -> list[KidSpec]:
    return [KidSpec(gender=g, age=a) for g, a in kids]


# Seed-dataset families: six variations of the typical family + six variants.
PRESETS: dict[str, FamilySpec] = {
    "typical-1": typical_family((3, 8, 14), ("female", "male", "female")),
    "typical-2": typical_family((6, 9, 11), ("male", "male", "female")),
    "typical-3": typical_family((13, 15, 17), ("female", "male", "male")),
    "typical-4": typical_family((1, 4, 7), ("male", "female", "female")),
    "typical-5": typical_family((5, 10, 16), ("female", "female", "male")),
    "typical-6": typical_family((8, 12, 12), ("male", "female", "male")),
    "city-apartment": FamilySpec(
        adults=_adults("male", "female"),
        kids=_kids(("female", 5), ("male", 9)),
        housing="apartment",
        balcony=True,
        cats=1,
        cars=0,
    ),
    "single-parent": FamilySpec(
        adults=_adults("female"), kids=_kids(("male", 7), ("female", 12)), housing="house"
    ),
    "couple-no-kids": FamilySpec(
        adults=_adults("male", "female"), housing="house", garden=True, dogs=1, cars=1
    ),
    "big-family": FamilySpec(
        adults=_adults("male", "female"),
        kids=_kids(("male", 2), ("female", 6), ("male", 10), ("female", 15)),
        housing="house",
        garden=True,
        garage=True,
        dogs=1,
        cats=1,
        cars=2,
    ),
    "teenagers": FamilySpec(
        adults=_adults("male", "female"),
        kids=_kids(("female", 14), ("male", 16), ("female", 17)),
        housing="house",
        garden=True,
        cars=1,
    ),
    "young-family": FamilySpec(
        adults=_adults("male", "female"),
        kids=_kids(("female", 1)),
        housing="apartment",
        balcony=True,
        cars=1,
    ),
}

# --- Names --------------------------------------------------------------------

_NAMES = {
    ("male", "adult"): ["Jeroen", "Mark", "Bas", "Thomas", "Rik", "Joost", "Sander", "Pieter"],
    ("female", "adult"): ["Anouk", "Linda", "Esther", "Marloes", "Sanne", "Femke", "Iris", "Eva"],
    ("male", "kid"): ["Sem", "Daan", "Luuk", "Finn", "Milan", "Bram", "Jesse", "Thijs", "Noah"],
    ("female", "kid"): ["Emma", "Julia", "Tess", "Lotte", "Noor", "Saar", "Fleur", "Lieke", "Mila"],
}
_DOG_NAMES = ["Max", "Bello", "Guus", "Pip", "Luna", "Bobby", "Sam", "Dribbel"]
_CAT_NAMES = ["Minoes", "Tijger", "Poekie", "Snoes", "Kiki", "Mimi", "Felix"]
_SURNAMES = ["de Vries", "Jansen", "Bakker", "Visser", "Smit", "Meijer", "de Boer", "Mulder"]


class Person(BaseModel):
    name: str
    gender: Gender
    role: Literal["adult", "kid"]
    age: int | None = None


class FamilyWorld(BaseModel):
    """A FamilySpec with concrete names: what the generator writes tickets for."""

    surname: str
    people: list[Person]
    dog_names: list[str]
    cat_names: list[str]
    spec: FamilySpec

    @property
    def submitters(self) -> list[Person]:
        """Who types tickets: adults, and kids old enough to use a phone."""
        return [p for p in self.people if p.role == "adult" or (p.age or 0) >= 9]

    def describe(self) -> str:
        spec = self.spec
        lines = [f"Family {self.surname}:"]
        for p in self.people:
            detail = "adult" if p.role == "adult" else f"child, {p.age} years"
            lines.append(f"- {p.name} ({p.gender}, {detail})")
        home = "a house" if spec.housing == "house" else "an apartment"
        extras = [
            label
            for flag, label in [
                (spec.garden, "a garden"),
                (spec.balcony, "a balcony"),
                (spec.garage, "a garage"),
                (spec.bikes, "bikes"),
            ]
            if flag
        ]
        lines.append(f"Home: {home}" + (f" with {', '.join(extras)}" if extras else ""))
        if self.dog_names:
            lines.append(f"Dog(s): {', '.join(self.dog_names)}")
        if self.cat_names:
            lines.append(f"Cat(s): {', '.join(self.cat_names)}")
        lines.append(f"Cars: {spec.cars}" if spec.cars else "No car")
        return "\n".join(lines)


def build_world(spec: FamilySpec, rng: random.Random) -> FamilyWorld:
    used: set[str] = set()

    def pick(pool: list[str]) -> str:
        name = rng.choice([n for n in pool if n not in used] or pool)
        used.add(name)
        return name

    people = [
        Person(name=pick(_NAMES[(a.gender, "adult")]), gender=a.gender, role="adult")
        for a in spec.adults
    ] + [
        Person(name=pick(_NAMES[(k.gender, "kid")]), gender=k.gender, role="kid", age=k.age)
        for k in spec.kids
    ]
    return FamilyWorld(
        surname=rng.choice(_SURNAMES),
        people=people,
        dog_names=[pick(_DOG_NAMES) for _ in range(spec.dogs)],
        cat_names=[pick(_CAT_NAMES) for _ in range(spec.cats)],
        spec=spec,
    )


def category_weights(spec: FamilySpec) -> dict[Category, float]:
    """Relative frequency of each category for this family (hidden ground truth)."""
    kids = len(spec.kids)
    young_kids = sum(1 for k in spec.kids if k.age < 12)
    pets = spec.dogs + spec.cats
    maintenance = 0.8
    maintenance += 0.6 if spec.housing == "house" else 0.1
    maintenance += 0.8 if spec.garden else 0.0
    maintenance += 0.2 if spec.garage else 0.0
    maintenance += 0.3 * spec.cars
    return {
        Category.CHORES: 2.5 + 0.6 * pets + 0.2 * kids,
        Category.GROCERIES: 1.8 + 0.15 * (len(spec.adults) + kids),
        Category.KIDS: 0.0 if kids == 0 else 0.8 + 0.5 * kids + 0.3 * young_kids,
        Category.HOME_MAINTENANCE: maintenance,
        Category.FINANCE: 0.8,
        Category.SOCIAL: 1.0,
        Category.OTHER: 0.6,
    }
