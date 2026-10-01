"""Pure retrieval functions, shared by the planning suggestions (app/services/similar.py)
and the offline evaluation (ml/retrieval_eval.py): the same code in both places, so the
evaluated behaviour is the served behaviour. See docs/learning/07-embeddings.md.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Neighbours below this cosine similarity are not shown: "the most similar of 40
# unrelated tasks" is still unrelated. Chosen with ml/retrieval_eval.py on three
# simulated worlds: at 0.3, ~90% of the shown neighbours are the same kind of task and
# ~63% of tickets get a suggestion (0.4: ~95% but only ~43%).
MIN_SIMILARITY = 0.3


def tfidf_vectors(texts: Sequence[str]) -> np.ndarray:
    """Unit-length TF-IDF vectors (character n-grams within words), fitted on exactly
    these texts, without labels."""
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), strip_accents="unicode", sublinear_tf=True
    )
    return vectorizer.fit_transform([str(t) for t in texts]).toarray().astype(np.float32)


def tfidf_similarities(query: str, candidates: Sequence[str]) -> np.ndarray:
    """Cosine similarity of `query` to each candidate, with TF-IDF fitted on the query
    plus the candidates only: what exists at that moment. Serving and the offline
    evaluation both call this, so IDF weights (and the MIN_SIMILARITY scale) match.
    In the Phase 8 comparison this beat a multilingual sentence embedding for finding
    the same kind of task (docs/learning/07-embeddings.md)."""
    if len(candidates) == 0:
        return np.zeros(0, dtype=np.float32)
    vectors = tfidf_vectors([query, *candidates])
    return vectors[1:] @ vectors[0]


def top_k(
    similarities: np.ndarray, k: int, min_similarity: float = -np.inf
) -> list[tuple[int, float]]:
    """The k most similar candidates at or above `min_similarity`, most similar first,
    as (candidate index, similarity)."""
    order = np.argsort(-similarities, kind="stable")[:k]
    return [(int(i), float(similarities[i])) for i in order if similarities[i] >= min_similarity]


def weighted_vote(
    labels: Sequence[str | None],
    similarities: Sequence[float],
    order: Sequence[str],
    min_similarity: float = MIN_SIMILARITY,
) -> dict[str, float] | None:
    """kNN prediction: each sufficiently similar neighbour votes for its label with
    weight = its similarity. Returns a probability per label, or None without votes."""
    weights = dict.fromkeys(order, 0.0)
    for label, sim in zip(labels, similarities, strict=True):
        if label in weights and sim >= min_similarity:
            weights[label] += sim
    total = sum(weights.values())
    if total == 0:
        return None
    return {label: w / total for label, w in weights.items()}


def precision_at_k(relevant: Sequence[bool], k: int) -> float | None:
    """Fraction of the first k results that are relevant. With fewer than k results
    the denominator is the number returned (a household with little history isn't
    penalised for results that can't exist); None when nothing was returned."""
    shown = list(relevant)[:k]
    return sum(shown) / len(shown) if shown else None
