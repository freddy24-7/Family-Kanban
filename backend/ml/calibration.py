"""Temperature scaling: one parameter T that sharpens (T < 1) or softens (T > 1) a
model's probabilities so that "80% confident" means right ~80% of the time.
T is fitted on out-of-fold predictions, never on the holdout.
See docs/learning/02-text-classification-and-evaluation.md, section 7."""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, ClassifierMixin


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(proba, 1e-12, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(proba: np.ndarray, y, classes) -> float:
    """T minimising the negative log-likelihood of the true labels."""
    # Column position of each true label. Not searchsorted: `classes` may be in a
    # meaningful, non-alphabetical order (effort is S, M, L).
    position = {str(c): i for i, c in enumerate(classes)}
    index = np.array([position[str(label)] for label in y])

    def nll(t: float) -> float:
        p = apply_temperature(proba, t)
        return float(-np.log(p[np.arange(len(index)), index]).mean())

    return float(minimize_scalar(nll, bounds=(0.05, 10), method="bounded").x)


class TemperatureScaled(ClassifierMixin, BaseEstimator):
    """Wraps an already-fitted classifier; only rescales its probabilities."""

    def __init__(self, model=None, temperature: float = 1.0):
        self.model = model
        self.temperature = temperature

    @property
    def classes_(self):
        return self.model.classes_

    def predict_proba(self, X):
        return apply_temperature(self.model.predict_proba(X), self.temperature)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
