"""The interface every served model implements, plus the Phase 1 placeholder."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelOutput:
    predicted: str
    confidence: float  # probability of the predicted class, 0..1
    probabilities: dict[str, float]  # full distribution over classes


class TextClassifier(Protocol):
    def predict(self, texts: list[str]) -> list[ModelOutput]: ...


class StubClassifier:
    """Always predicts the same label with confidence 0.0.

    Exists so the whole prediction pipeline (versioning, logging, "needs review"
    flag) works end to end before a real model is trained in Phase 3. Its 0.0
    confidence is honest: it knows nothing, so every topic gets flagged.
    """

    def __init__(self, label: str):
        self.label = label

    def predict(self, texts: list[str]) -> list[ModelOutput]:
        return [ModelOutput(self.label, 0.0, {}) for _ in texts]
