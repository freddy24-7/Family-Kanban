"""Model definitions: baselines, TF-IDF + logistic regression, and an ordinal
classifier for effort. Everything is an sklearn estimator so it can be
cross-validated, pickled into the registry and served the same way.
See docs/learning/02-text-classification-and-evaluation.md."""

import re
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from app.domain import Category

EFFORT_ORDER = ["S", "M", "L"]

# --- Baselines ------------------------------------------------------------------------


def majority_baseline() -> DummyClassifier:
    """Always predicts the most frequent class seen in training."""
    return DummyClassifier(strategy="most_frequent")


# Twenty-odd hand-written rules: what a developer would write without ML.
KEYWORD_RULES: dict[str, list[str]] = {
    Category.GROCERIES: [
        "boodschap",
        "halen",
        "melk",
        "brood",
        "kaas",
        "appel",
        "supermarkt",
        "ah ",
    ],
    Category.FINANCE: ["betal", "belasting", "verzeker", "rekening", "overmak", "bank", "contract"],
    Category.HOME_MAINTENANCE: [
        "repar",
        "maaien",
        "dakgoot",
        "kraan",
        "lamp",
        "apk",
        "fiets",
        "schilder",
    ],
    Category.KIDS: ["school", "zwemles", "voetbal", "huiswerk", "gymtas", "ouderavond", "luier"],
    Category.SOCIAL: ["verjaardag", "cadeau", "oma", "opa", "kaart", "feest", "etentje"],
    Category.CHORES: ["was ", "stofzuig", "afwas", "vaatwasser", "uitlaten", "schoonmak", "koken"],
    Category.OTHER: ["tandarts", "kapper", "paspoort", "huisarts"],
}


class KeywordBaseline(ClassifierMixin, BaseEstimator):
    """First matching rule wins; no match -> the majority class seen in training."""

    def fit(self, X, y):
        values, counts = np.unique(np.asarray(y), return_counts=True)
        self.classes_ = values
        self.default_ = values[np.argmax(counts)]
        return self

    def predict(self, X):
        out = []
        for text in X:
            lowered = f" {str(text).lower()} "
            label = next(
                (
                    str(cat)
                    for cat, words in KEYWORD_RULES.items()
                    if any(w in lowered for w in words)
                ),
                self.default_,
            )
            out.append(label)
        return np.array(out)


# --- TF-IDF + logistic regression ----------------------------------------------------------

Features = Literal["word", "char", "word+char"]


@dataclass(frozen=True)
class TextModelConfig:
    """Hyperparameters of a text classifier; stored with every trained model."""

    features: Features = "char"
    C: float = 1.0
    class_weight: Literal["balanced"] | None = None
    mask_names: bool = False
    ordinal: bool = False  # effort only: Frank & Hall ordinal decomposition

    def as_dict(self) -> dict:
        return asdict(self)


def _word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w\w+\b",
        strip_accents="unicode",
        sublinear_tf=True,
    )


def _char_vectorizer() -> TfidfVectorizer:
    # char_wb: n-grams inside word boundaries, so "vaatwasser" shares "vaat" with "vaat".
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2, strip_accents="unicode", sublinear_tf=True
    )


def _vectorizer(features: Features):
    if features == "word":
        return _word_vectorizer()
    if features == "char":
        return _char_vectorizer()
    return FeatureUnion([("word", _word_vectorizer()), ("char", _char_vectorizer())])


_NAME_PATTERN: re.Pattern | None = None


def mask_names(texts):
    """Replace known first names and pet names by a placeholder token. Experiment
    for the 'Lieke -> kids' shortcut; uses the simulator's name lists."""
    global _NAME_PATTERN
    if _NAME_PATTERN is None:
        from sim.family import _CAT_NAMES, _DOG_NAMES, _NAMES

        names = {n for pool in _NAMES.values() for n in pool} | set(_DOG_NAMES) | set(_CAT_NAMES)
        _NAME_PATTERN = re.compile(
            r"\b(" + "|".join(sorted(names, key=len, reverse=True)) + r")\b", re.I
        )
    return [_NAME_PATTERN.sub(" naamx ", str(t)) for t in texts]


def build_text_classifier(config: TextModelConfig):
    classifier = LogisticRegression(C=config.C, class_weight=config.class_weight, max_iter=3000)
    if config.ordinal:
        classifier = OrdinalClassifier(classifier, order=EFFORT_ORDER)
    steps = []
    if config.mask_names:
        steps.append(("mask_names", FunctionTransformer(mask_names)))
    steps += [("features", _vectorizer(config.features)), ("classifier", classifier)]
    return Pipeline(steps)


# --- Ordinal classification ------------------------------------------------------------------


class OrdinalClassifier(ClassifierMixin, BaseEstimator):
    """Frank & Hall (2001): for ordered classes S < M < L, train one binary model per
    threshold ("is it bigger than S?", "is it bigger than M?") and combine:
        P(S) = 1 - P(>S),  P(M) = P(>S) - P(>M),  P(L) = P(>M).
    Unlike plain multiclass, the models know that L is 'more' than M."""

    def __init__(self, estimator=None, order=None):
        self.estimator = estimator
        self.order = order

    def fit(self, X, y, sample_weight=None):
        order = list(self.order or EFFORT_ORDER)
        rank = np.array([order.index(v) for v in y])
        self.classes_ = np.array(order)
        self.models_ = [
            clone(self.estimator).fit(X, (rank > k).astype(int), sample_weight=sample_weight)
            for k in range(len(order) - 1)
        ]
        return self

    def predict_proba(self, X):
        greater = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
        # Monotonic fix-up: P(>M) must not exceed P(>S).
        greater = np.minimum.accumulate(greater, axis=1)
        n = len(self.classes_)
        proba = np.empty((greater.shape[0], n))
        proba[:, 0] = 1 - greater[:, 0]
        for k in range(1, n - 1):
            proba[:, k] = greater[:, k - 1] - greater[:, k]
        proba[:, n - 1] = greater[:, n - 2]
        proba = np.clip(proba, 0, 1)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
