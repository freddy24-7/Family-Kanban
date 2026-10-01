"""Sentence embeddings: the ONLY module that imports fastembed.
See docs/learning/07-embeddings.md.

A pretrained, frozen multilingual model turns a text into a 384-number vector; texts
with the same meaning end up close together. EXPERIMENTS ONLY: in Phase 8 it lost to
TF-IDF both for similarity search and as classifier features, so fastembed is a dev
dependency and nothing in production uses this module (ml.train refuses to register
embedding-feature models).

- `embed(texts)` -> float32 array (n, DIM), L2-normalised, so cosine = dot product.
  CPU-bound: call it via `asyncio.to_thread` from async code.
- The model loads lazily, once per process (~16 s first download, ~2 s from disk,
  ~400 MB memory). `python -m ml.embeddings download` pre-fetches it.
- Tests swap in a fake with `set_backend()`, so they never load the real model.
"""

import sys
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384
_CACHE_SIZE = 20_000  # texts; ~1.5 kB each

Backend = Callable[[list[str]], np.ndarray]

_backend: Backend | None = None
_lock = threading.Lock()
_cache: OrderedDict[str, np.ndarray] = OrderedDict()


def _fastembed_backend() -> Backend:
    from fastembed import TextEmbedding  # heavy import (onnxruntime), only when needed

    model = TextEmbedding(MODEL_NAME)
    return lambda texts: np.array(list(model.embed(texts, batch_size=64)), dtype=np.float32)


def _get_backend() -> Backend:
    global _backend
    if _backend is None:
        with _lock:
            if _backend is None:
                _backend = _fastembed_backend()
    return _backend


def set_backend(backend: Backend | None) -> None:
    """Replace the model (tests); None = the real model again."""
    global _backend
    _backend = backend
    _cache.clear()


def normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)


def embed(texts: Sequence[str]) -> np.ndarray:
    """Unit-length embeddings, one row per text. Repeated texts come from a cache
    (cross-validation embeds the same texts once per fold)."""
    texts = [str(t) for t in texts]
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    with _lock:
        found = {t: _cache[t] for t in texts if t in _cache}
    missing = list(dict.fromkeys(t for t in texts if t not in found))
    if missing:
        vectors = normalise(_get_backend()(missing))
        found.update(zip(missing, vectors, strict=True))
        with _lock:
            for text, vector in zip(missing, vectors, strict=True):
                _cache[text] = vector
            while len(_cache) > _CACHE_SIZE:
                _cache.popitem(last=False)
    return np.vstack([found[t] for t in texts])


class EmbeddingFeatures(TransformerMixin, BaseEstimator):
    """sklearn step: texts -> embeddings. Stateless (nothing is learned in `fit`), so a
    pickled pipeline stores only the model name; the vectors are computed when used."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    def fit(self, X, y=None):
        if self.model_name != MODEL_NAME:
            raise ValueError(f"Only {MODEL_NAME} is available, not {self.model_name}")
        return self

    def transform(self, X):
        return embed(list(X))


def download() -> None:
    """Fetch the model into the fastembed cache (so the first experiment run doesn't wait)."""
    vectors = embed(["melk halen", "gras maaien"])
    print(f"{MODEL_NAME}: dim {vectors.shape[1]}, cos {float(vectors[0] @ vectors[1]):.2f}")


if __name__ == "__main__":
    if sys.argv[1:] == ["download"]:
        download()
    else:
        sys.exit("usage: python -m ml.embeddings download")
