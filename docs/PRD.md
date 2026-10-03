# PRD: Family Kanban — AI-Assisted Household Task Management (v2)

> v2 revises the original PRD after a feasibility review. Changes from v1 are
> summarised in §13. The phased build plan lives in [PLAN.md](PLAN.md).

## 1. Purpose

A portfolio project that closes a specific gap: hands-on **classical ML, model
deployment and MLOps** (training, versioning, monitoring, drift detection,
retraining) on top of existing strengths (LLM integration, RAG, full-stack).

A secondary, equally important goal: **the developer learns the ML deeply**.
Every ML component is built with an explanation of the underlying concept, so
the result can be defended from first principles in an interview.

## 2. Problem statement

Household tasks arrive informally and unevenly across family members, with no
shared way to capture, classify, prioritise, plan and review them. At small
scale this mirrors enterprise AI-engineering work: unstructured input in →
classified/prioritised/actioned output → continuous improvement.

## 3. Goals

- A working, deployed PWA used by the developer's own family (real data).
- A **simulator** that runs the whole app for LLM-generated demo families, so
  the full ML lifecycle can be demonstrated quickly and under controlled conditions.
- Full lifecycle: intake → classification (ML) → planning → execution → review → retraining.
- Multi-tenant architecture from day one, so other families *could* use it later.

## 4. Non-goals

- Not a general project-management tool.
- Not a production MLOps platform. Local, lightweight equivalents (disk/volume
  model registry, script-based retraining) are deliberate and stated as such.
- No tenant-management/billing UI in v1 (multi-tenancy is structural, not a feature).
- Photos on topics: deferred (no skill-map value).

## 5. Users & roles

Every person has their own account (email/password or Google). Kids have email addresses.

| Role | Who | Can |
|---|---|---|
| Member | any household member | submit topics, move own sprint items |
| Planner | parent(s) | confirm/correct classifications, plan sprints, assign |
| Reviewer | parent(s), may rotate | run sprint review |
| Platform admin | the developer only | Model Health Dashboard, simulator, retraining, cross-tenant views |

Roles are per-household flags; one person may hold several.

## 6. Core flows

### 6.1 Onboarding
Sign up → creates a household → invite members by email → each member signs in.

### 6.2 Intake
Member opens the PWA and submits a topic: free text (Dutch), optional due date.
The classifier predicts **category**, **effort (S/M/L)** and a **confidence**.
Below a data-driven confidence threshold the topic is flagged "needs review".

### 6.3 Backlog review (labelling point #1)
The Planner sees the classified backlog and **explicitly confirms or corrects**
category and effort. Whether the prediction was changed is logged (guards
against automation bias inflating measured accuracy).

### 6.4 Sprint planning
Planner selects backlog items into a weekly sprint and assigns them. For each item
the system shows similar past tasks, who handled them and how long they took
(embedding similarity + aggregation).

### 6.5 Execution
Kanban board per sprint: To Do / In Progress / Done, filter by person, drag-and-drop.

### 6.6 Sprint review (labelling point #2)
Per item: completed (Y/N), actual effort (S/M/L), optional note. These are the
ground-truth labels for effort.

### 6.7 Demo & simulation (platform admin)
1. **Define a family**: number of adult men/women, boys/girls (with ages), and
   assets: house/apartment, garden, dog, cat, car, garage, bikes, etc.
2. **Generate tickets**: Gemini generates Z Dutch tickets for that family.
3. **Simulate**: fast-forward N weekly sprints through the real service layer —
   intake, classification, backlog confirmation, planning, execution, review —
   driven by a hidden *world model* (ground-truth rules + noise) and optional
   **drift scenarios** (e.g. seasonal shift, "family gets a dog" in week 8).
4. Simulated households are clearly marked as such everywhere.

### 6.8 Model Health Dashboard (platform admin)
- Accuracy / macro-F1 over time, **split by data source** (simulated vs real).
- Confidence distribution over time; calibration.
- Category distribution drift (PSI / chi-square) with detected-vs-planted drift
  for simulations.
- Retraining log: version, data size, holdout metrics, promoted or rejected.

## 7. Data model (high level)

Every tenant-owned table carries `household_id`. Every data row records its
provenance (`source`: `real` | `simulated`).

- **Household**: id, name, kind (`real`|`simulated`), `training_eligible` (bool)
- **User** (fastapi-users base: id, email, hashed_password, is_active, is_verified, is_superuser):
  + display_name; `is_superuser` = platform admin
- **OAuthAccount**: user_id, provider (`google`), account_id, tokens (fastapi-users)
- **AccessToken**: token, user_id, created_at (database token strategy, revocable)
- **Invite**: id, household_id, email, token_hash, invited_by, expires_at, accepted_at
- **Membership**: user_id, household_id, is_planner, is_reviewer, profile (simulator-only:
  gender, age, world-model traits). Simulated family members are `User` rows with
  `is_simulated = true` and login disabled, so assignment works identically for both.
- **FamilyProfile**: household_id, composition, assets (for simulated households)
- **Topic**: id, household_id, text, due_by, created_by, occurred_at, source,
  category_label, effort_label, label_source (`planner`|`review`|`generator`),
  labeled_at, labeled_by, category_prediction_changed, effort_prediction_changed
- **Prediction** (append-only): id, topic_id, household_id, model_version_id, task
  (`category`|`effort`), predicted, confidence, probabilities, source, occurred_at, created_at
- **Sprint**: id, household_id, start_date, end_date, status
- **SprintItem**: id, sprint_id, topic_id, assignee_id, status, completed,
  effort_actual, review_note, reviewed_at
- **SprintReview**: id, sprint_id, notes, completed_count, total_count
- **ModelVersion**: id, task, status (`candidate`|`active`|`retired`|`rejected`),
  trained_at, artifact_uri, training_set_size, training_data_hash,
  metrics (per holdout, per source), parent_version_id, notes
- **HoldoutSet**: id, name, frozen_at, topic_ids manifest hash
- **TopicEmbedding**: topic_id, model_name, vector (pgvector)
- **SimulationRun**: id, household_id, config, random_seed, sim_start_date,
  weeks, status, llm_tokens_used

`occurred_at` (domain time) is separate from database insert time so the simulator
can fast-forward time. Application code never calls "now" directly; it uses an
injectable clock.

## 8. Skill coverage map

Every technology must earn its place here. Nothing is added for CV padding.

| Gap | Covered by |
|---|---|
| Classical ML / scikit-learn | Category + effort classifiers (TF-IDF + logistic regression; embeddings + LR as a comparison experiment) |
| Model evaluation | Frozen holdouts, macro-F1, confusion matrices, calibration curves, sim-vs-real evaluation |
| Model deployment | Classifier served in-process by FastAPI with hot-swap on promotion |
| Model versioning | ModelVersion table + versioned artifacts + data hashes; every prediction tagged with its version |
| Monitoring / drift | Model Health Dashboard; PSI / chi-square drift; validated against planted drift in simulations |
| Retraining / MLOps | Retrain script: champion/challenger, promotion gate, registry pattern designed to map to MLflow |
| Synthetic data / simulation | Gemini ticket generation + world-model simulator with controllable drift |
| RAG / embeddings (reused strength) | Similar-task retrieval for planning suggestions; embeddings (fastembed) vs TF-IDF compared, TF-IDF served because it won (lesson 07) |
| Full-stack / auth / multi-tenancy | React PWA, FastAPI, Postgres, self-hosted auth (fastapi-users + Google OAuth), household-scoped data |
| Agentic (reused strength, stretch) | Planner Assistant proposing a draft sprint |

## 9. Technical architecture

- **Frontend**: Vite + React + TypeScript PWA (`vite-plugin-pwa`), TanStack Query,
  dnd-kit (kanban), Recharts (dashboard). Dutch UI via an i18n string file;
  all code, identifiers and docs in English. Hosted on Vercel or Railway.
- **Backend**: FastAPI (Python, uv), SQLAlchemy 2 + Alembic, on Railway.
- **Database**: Postgres + pgvector on Railway.
- **Auth**: self-hosted with `fastapi-users` (SQLAlchemy adapter): email/password
  with verification and password reset, Google OAuth via `httpx-oauth`. Bearer
  transport + database token strategy (revocable, long-lived for the PWA).
  Passwords hashed with argon2. Note: fastapi-users is in maintenance mode
  (security fixes only) — acceptable here; the auth module is isolated so it
  can be swapped later.
- **Email**: transactional email (verification, reset, invites) via Resend,
  isolated in `app/mailer.py`, inert without its API key (links logged in dev).
- **ML**: scikit-learn, pandas, joblib; exploration in Jupyter notebooks.
  Embeddings via `fastembed` (ONNX, no PyTorch) with a multilingual model, as an
  experiment only: TF-IDF retrieval won the Phase 8 comparison and is what's served.
- **LLM**: Gemini, isolated in one module, inert without `GEMINI_API_KEY`,
  structured JSON output, per-run token cap, defensive parsing.
- **Model artifacts**: Postgres table `model_artifact`, referenced by URI (ADR-002).
- **Retraining**: manual script / admin action in v1; Railway cron later.

## 10. Training data policy

- Only households with `training_eligible = true` feed training: the developer's
  demo (simulated) households and the developer's own family.
- Every metric is reported **per source**. The key question the project answers:
  *how well does a model trained mostly on synthetic data perform on real data,
  and does adding real data close the gap?*
- Frozen holdouts (simulated and real) are never trained on and never regenerated.
  New holdouts may be added; old ones stay fixed so versions stay comparable.

## 11. Success criteria

- The developer's family uses the app for ≥ 2–3 real sprints.
- A simulation with planted drift is detected on the dashboard, and a retraining
  event measurably recovers performance.
- At least one retraining event evaluated on real data.
- The system can be walked through in < 10 minutes, each skill-map row pointing
  to working code.
- The developer can explain every ML choice (why this model, metric, threshold,
  split, drift test).

## 12. Open questions

- Transformer fine-tuning: only if TF-IDF/embeddings plateau on real data.
- Push notifications: deprioritised.
- Weighting real vs simulated samples in training: decided experimentally in Phase 7.

## 13. Changes from v1

- Multi-tenancy + self-hosted auth (fastapi-users: email/password + Google); kids have accounts.
- New demo/simulation feature (family definition → Gemini tickets → simulated sprints).
- Predictions moved to an append-only `Prediction` table.
- Per-item review fields; labels captured at backlog confirmation as well as review.
- Data provenance (`source`) on everything; metrics split by source.
- Frozen holdouts; champion/challenger promotion gate.
- Injectable clock (simulation time).
- Frontend: Vite instead of Next.js (no SSR need). Photos deferred.
- Explicit learning goal: every ML step is taught, not just built.
