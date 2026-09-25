"""Model registry: the ONLY place that turns a ModelVersion's artifact_uri into
a model object. Phase 3 adds joblib artifacts (save/load with metadata)."""

from functools import lru_cache

from ml.serving import StubClassifier, TextClassifier

STUB_SCHEME = "stub://"


@lru_cache(maxsize=16)
def load_model(artifact_uri: str) -> TextClassifier:
    """Load (and cache) the model behind an artifact URI. Model versions are
    immutable, so caching by URI is safe."""
    if artifact_uri.startswith(STUB_SCHEME):
        return StubClassifier(label=artifact_uri.removeprefix(STUB_SCHEME))
    raise ValueError(f"Unsupported artifact URI: {artifact_uri}")
