"""Model registry: the ONLY module that stores or loads model artifacts.

Artifacts live in Postgres (`model_artifact`, joblib bytes) and are referenced by a
ModelVersion's `artifact_uri` ("db://<artifact id>"); "stub://<label>" is the
built-in placeholder. Mapping to MLflow (ADR-002): ModelVersion = registered model
version, status active/candidate = aliases champion/challenger, this module =
mlflow.sklearn.log_model / load_model.

Two sides:
- API (async): `load_model()` with an in-process cache (versions are immutable).
- Training CLI (sync): `save_and_register()` and `promote()`.
"""

import hashlib
import io
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import joblib
import sklearn
from sqlalchemy import create_engine, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app import config
from app.domain import ModelStatus, Task
from app.models import ModelArtifact, ModelVersion
from ml.serving import SklearnTextClassifier, StubClassifier, TextClassifier

log = logging.getLogger(__name__)

STUB_SCHEME = "stub://"
DB_SCHEME = "db://"

_cache: dict[str, TextClassifier] = {}


class ModelLoadError(RuntimeError):
    pass


# --- Loading (API) -----------------------------------------------------------------------


async def load_model(session: AsyncSession, artifact_uri: str) -> TextClassifier:
    if artifact_uri in _cache:
        return _cache[artifact_uri]
    if artifact_uri.startswith(STUB_SCHEME):
        model: TextClassifier = StubClassifier(label=artifact_uri.removeprefix(STUB_SCHEME))
    elif artifact_uri.startswith(DB_SCHEME):
        artifact = await session.get(ModelArtifact, uuid.UUID(artifact_uri.removeprefix(DB_SCHEME)))
        if artifact is None:
            raise ModelLoadError(f"Artifact not found: {artifact_uri}")
        model = _deserialize(artifact.data, artifact.sha256, artifact.meta)
    else:
        raise ModelLoadError(f"Unsupported artifact URI: {artifact_uri}")
    _cache[artifact_uri] = model
    return model


def _deserialize(data: bytes, sha256: str, meta: dict[str, Any]) -> TextClassifier:
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ModelLoadError("Artifact bytes do not match their recorded hash")
    trained_with = meta.get("sklearn_version", "")
    if _major_minor(trained_with) != _major_minor(sklearn.__version__):
        # Pickles are only reliable with the same library version.
        raise ModelLoadError(
            f"Model trained with scikit-learn {trained_with}, running {sklearn.__version__}"
        )
    # Only artifacts written by our own training code are ever loaded (never
    # user-supplied bytes): unpickling untrusted data can execute code.
    estimator = joblib.load(io.BytesIO(data))
    return SklearnTextClassifier(estimator, preprocessing=meta.get("preprocessing"))


def _major_minor(version: str) -> tuple[str, ...]:
    return tuple(version.split(".")[:2])


# --- Saving and promotion (training CLI, sync) ---------------------------------------------


def sync_session() -> Session:
    url = config.DATABASE_URL  # psycopg 3 works for both sync and async engines
    return Session(create_engine(url), expire_on_commit=False)


def serialize(estimator) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    joblib.dump(estimator, buffer, compress=3)
    data = buffer.getvalue()
    return data, hashlib.sha256(data).hexdigest()


def active_version(session: Session, task: Task) -> ModelVersion | None:
    stmt = select(ModelVersion).where(
        ModelVersion.task == task, ModelVersion.status == ModelStatus.ACTIVE
    )
    return session.scalar(stmt)


def load_model_sync(session: Session, version: ModelVersion) -> TextClassifier:
    uri = version.artifact_uri
    if uri.startswith(STUB_SCHEME):
        return StubClassifier(label=uri.removeprefix(STUB_SCHEME))
    artifact = session.get(ModelArtifact, uuid.UUID(uri.removeprefix(DB_SCHEME)))
    return _deserialize(artifact.data, artifact.sha256, artifact.meta)


def next_version_name(session: Session, task: Task) -> str:
    count = session.scalar(
        select(func.count())
        .select_from(ModelVersion)
        .where(ModelVersion.task == task, ~ModelVersion.artifact_uri.startswith(STUB_SCHEME))
    )
    return f"{task.value}-v{(count or 0) + 1}"


def save_and_register(
    session: Session,
    task: Task,
    estimator,
    meta: dict[str, Any],
    metrics: dict[str, Any],
    training_set_size: int,
    training_data_hash: str,
    parent: ModelVersion | None,
    notes: str,
) -> ModelVersion:
    """Store the artifact and create a *candidate* ModelVersion (not yet serving)."""
    meta = {**meta, "sklearn_version": sklearn.__version__}
    data, digest = serialize(estimator)
    artifact = ModelArtifact(data=data, sha256=digest, size_bytes=len(data), meta=meta)
    session.add(artifact)
    session.flush()
    version = ModelVersion(
        task=task,
        name=next_version_name(session, task),
        status=ModelStatus.CANDIDATE,
        artifact_uri=f"{DB_SCHEME}{artifact.id}",
        trained_at=datetime.now(UTC),
        training_set_size=training_set_size,
        training_data_hash=training_data_hash,
        metrics=metrics,
        parent_version_id=parent.id if parent else None,
        notes=notes,
    )
    session.add(version)
    session.commit()
    return version


def promote(session: Session, candidate: ModelVersion) -> None:
    """Make `candidate` the active model for its task; the old one is retired.
    One transaction, so there is never zero or two active models."""
    current = active_version(session, candidate.task)
    if current is not None:
        current.status = ModelStatus.RETIRED
        session.flush()  # the partial unique index allows only one active per task
    candidate.status = ModelStatus.ACTIVE
    session.commit()


def reject(session: Session, candidate: ModelVersion, reason: str) -> None:
    candidate.status = ModelStatus.REJECTED
    candidate.notes = f"{candidate.notes or ''}\nRejected: {reason}".strip()
    session.commit()
