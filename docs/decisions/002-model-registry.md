# ADR-002: Lightweight model registry instead of MLflow (for now)

**Status**: accepted · 2026-09-25

## Context
We need model versioning: every prediction must be traceable to the exact model and
training data that produced it, and models must be promotable and rollback-able.

## Decision
A minimal registry built from two parts:
- **`ModelVersion` table** — id, task, status (`candidate`/`active`/`retired`/`rejected`),
  training data hash, training set size, metrics per holdout and per source, parent version.
- **Artifact storage** — `joblib` file + metadata JSON per version on disk
  (a Railway volume in production), referenced by URI. Only `backend/ml/registry.py`
  reads or writes artifacts.

## Why not MLflow now
At household scale (one small model, a single developer, a few versions a month) MLflow's
tracking server, artifact store and UI would add more ops than value.

## How it maps to MLflow later
| Ours | MLflow |
|---|---|
| `ModelVersion` row | Registered model version |
| `status` active/candidate/retired | Aliases (e.g. `champion`, `challenger`) |
| metrics JSON | Run metrics |
| training data hash + params | Run params / dataset tracking |
| `registry.py` | `mlflow.sklearn.log_model` / `load_model` |

Because all artifact access goes through one module, switching means replacing that
module, not the callers.
