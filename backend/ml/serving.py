"""The interface every served model implements."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ModelOutput:
    predicted: str
    confidence: float  # probability of the predicted class, 0..1
    probabilities: dict[str, float]  # full distribution over classes


class TextClassifier(Protocol):
    # Name of the text preprocessing the caller must apply first (None = raw text).
    preprocessing: str | None

    def predict(self, texts: list[str]) -> list[ModelOutput]: ...


class StubClassifier:
    """Always predicts the same label with confidence 0.0.

    Exists so the whole prediction pipeline (versioning, logging, "needs review"
    flag) works end to end before a real model is trained. Its 0.0 confidence is
    honest: it knows nothing, so every topic gets flagged."""

    preprocessing = None

    def __init__(self, label: str):
        self.label = label

    def predict(self, texts: list[str]) -> list[ModelOutput]:
        return [ModelOutput(self.label, 0.0, {}) for _ in texts]


class SklearnTextClassifier:
    """Adapter from an sklearn classifier with predict_proba to TextClassifier."""

    def __init__(self, estimator, preprocessing: str | None):
        self.estimator = estimator
        self.preprocessing = preprocessing

    def predict(self, texts: list[str]) -> list[ModelOutput]:
        proba = self.estimator.predict_proba(np.asarray(texts, dtype=object))
        classes = [str(c) for c in self.estimator.classes_]
        outputs = []
        for row in proba:
            best = int(np.argmax(row))
            outputs.append(
                ModelOutput(
                    predicted=classes[best],
                    confidence=round(float(row[best]), 4),
                    probabilities={
                        c: round(float(p), 4) for c, p in zip(classes, row, strict=True)
                    },
                )
            )
        return outputs
