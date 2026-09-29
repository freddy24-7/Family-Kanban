# Family Kanban

Portfolio project: a household kanban PWA with a scikit-learn classifier, model
versioning, drift monitoring, retraining, and a simulator that runs the whole app
for LLM-generated demo families. Spec: [docs/PRD.md](docs/PRD.md). Phased plan and
current phase: [docs/PLAN.md](docs/PLAN.md). Decisions: [docs/decisions/](docs/decisions/).

The developer's main goal is to **learn classical ML/MLOps**. For ML work, teach:
see [backend/ml/CLAUDE.md](backend/ml/CLAUDE.md).

## Layout

- `backend/` — FastAPI (`app/`), ML (`ml/`), simulator (`sim/`), `tests/`. Python 3.12, uv.
- `frontend/` — Vite + React + TS PWA.
- `notebooks/` — exploration and lessons; shipped code lives in `backend/ml/`.
- `backend/ml/holdouts/` — frozen holdout manifests (committed, never edited).
- `data/artifacts/` — model artifacts (not committed).

## Commands

- DB: `docker compose up -d` (Postgres 18 + pgvector on localhost:5433)
- Backend: `cd backend && uv run uvicorn app.main:app --reload`
- Backend tests / lint: `cd backend && uv run pytest` · `uv run ruff check . && uv run ruff format .`
  (tests build a `famkanban_test` DB from the real migrations; the DB container must be running)
- Migrations: `cd backend && uv run alembic revision --autogenerate -m "..."` → review → `uv run alembic upgrade head`; `uv run alembic check` must be clean
- Make platform admin: `cd backend && uv run python -m app.cli make-admin <email>`
- Demo families / synthetic tickets: `cd backend && uv run python -m sim.seed_dataset --presets typical-1 --tickets 25`
  (writes to whatever `DATABASE_URL` points at; label rules: `backend/app/labeling_guidelines.md`)
- Train/evaluate/register: `cd backend && uv run python -m ml.train --snapshot seed-v1 --task all [--dry-run | --promote]`
  (reads snapshot + household members; evaluates on every frozen holdout next to the champion)
- Holdouts: `uv run python -m ml.holdout suggest|freeze ...` · snapshots in `data/datasets/` (not committed)
- Deploy: see [docs/deploy.md](docs/deploy.md)
- Frontend: `cd frontend && npm run dev` · `npm run build` · `npm run lint` · `npm run format`

## Language

UI text is **Dutch** (all strings in `frontend/src/i18n/nl.ts`; generated tickets are
Dutch). Code, identifiers, comments, docs, commits: **English**. Never Norwegian.

## Invariants (load-bearing — don't break these)

- **Tenancy**: every tenant-owned table has `household_id`; every query goes through
  household-scoped functions in `app/repository.py`, and every household route depends on
  `app/tenancy.py` (`household_access` / `planner_access`). Cross-household access returns 404.
  Add new household-scoped endpoints to the parametrised list in `tests/test_tenancy.py`.
- **Provenance**: every data row carries `source` (`real` | `simulated`), derived from the
  household's `kind` (never chosen by the caller); metrics are always reported per source.
  Never mix them silently.
- **Predictions are append-only** and always carry `model_version_id`.
- **Frozen holdouts** (`backend/ml/holdouts/`) are never trained on, edited, or regenerated.
- **Promotion gate**: a new model becomes active only if it beats the active one on the
  frozen holdouts without regressing on real data.
- **Training eligibility**: only households with `training_eligible = true` feed training.
- **Clock**: domain time comes from the injectable clock, never `datetime.now()` directly
  (the simulator fast-forwards time).
- **One module per external dependency**: Gemini only in `app/llm.py`, email only in
  `app/mailer.py`, fastapi-users only in `app/auth.py`, DB queries only in
  `app/repository.py` (ML batch reads via `ml/datasets.py`), model artifacts only via
  `ml/registry.py` (stored in Postgres `model_artifact`).
  External services are env-gated and fail soft; the database fails loud.
- **Migrations** via Alembic; never drop tables holding real family data. Enum columns use
  `str_enum()` in `models.py` (VARCHAR + CHECK constraint).
- **Train/serve consistency**: text preprocessing (role tokens, `ml/text.py`) is shared by
  training and serving; models declare the preprocessing they expect in their metadata.
- **Layering**: routers stay thin; logic lives in `app/services/` so the simulator can call
  it directly with a `SimulatedClock`.

## Scope guard

Every technology must map to a row in PRD §8 (Skill Coverage Map). Lightweight local
equivalents (volume-based registry, script-triggered retraining) are deliberate —
don't upgrade them to MLflow/Kubernetes/etc. unless the plan changes.

## Conventions

Follow the `python-react-railway-conventions` skill for code patterns and
`external-service-setup` for Railway / Google OAuth / Resend wiring.
Run the `ml-reviewer` agent on changes under `backend/ml/` or `backend/sim/`.
