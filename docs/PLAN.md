# Build Plan — Family Kanban

**Status**: Phase 6 complete (2026-09-30) · next: Phase 7 (retraining loop)

Sized for ~8–10 hours/week. Each phase ends with something demonstrable.
Phases marked 🎓 include ML lessons: before building, the concept is explained;
after building, we read the results together. Lesson notes go in `docs/learning/`,
experiments in `notebooks/`.

Guiding priorities:
1. Get the real family using the app early — real data can't be backfilled.
2. Log every prediction with its model version and data source from day one.
3. Keep synthetic and real data separable at every step.

---

## Phase 0 — Foundations & Claude setup (~1 week)

**Build**
- Monorepo layout:
  ```
  backend/   app/ (FastAPI), ml/ (train, eval, registry), sim/ (simulator), tests/
  frontend/  Vite + React + TS PWA
  notebooks/ exploration & lessons
  docs/      PRD, PLAN, decisions/ (ADRs), learning/
  data/      local artifacts & frozen holdout manifests (see hooks)
  ```
- `uv` project for backend, `ruff`, `pytest`; `npm` + eslint/prettier for frontend.
- `docker-compose` with Postgres + pgvector for local dev.
- Claude setup (see §"Claude Code setup" below).
- ADR-001: stack choice (incl. self-hosted auth via fastapi-users, maintenance-mode trade-off). ADR-002: why a disk/volume registry instead of MLflow (for now).

**Done when**: `pytest` and `npm run dev` run on an empty skeleton; CLAUDE.md in place.

---

## Phase 1 — Backend core, tenancy & auth (~2 weeks)

**Build**
- SQLAlchemy models + first Alembic migration for the PRD §7 schema.
- Auth with `fastapi-users`: register, login, verify email, forgot/reset password,
  Google OAuth (`httpx-oauth`), bearer + database token strategy. Google Cloud
  OAuth client set up (see `external-service-setup` for redirect-URI gotchas).
- `app/mailer.py` (Resend) for verification/reset/invite mails; in dev, links are logged.
- Dependency resolving current user → household memberships.
- Tenancy: a `current_household` dependency; every query goes through
  household-scoped repository functions (one `db.py` / repository layer).
  Tests prove household A cannot read household B.
- Signup → create household; email invites; roles (planner/reviewer).
- Injectable `Clock` (real clock in the app, simulated clock in the simulator).
- Topics CRUD with `source`; a **stub classifier** (returns `other`/`M`, confidence 0)
  so the Prediction pipeline and version tagging exist before the model does.
- Deploy skeleton to Railway (API + Postgres) — early deploy flushes out infra issues.

- Sprint tables + APIs are deferred to the start of Phase 4 (first used there);
  simulator tables (FamilyProfile, SimulationRun, HoldoutSet) arrive in Phase 2.

**Done when**: a user can sign up, invite a member, create topics via the API, and
each topic has a Prediction row tagged with model version `stub-0`.

**Outcome**: 32 tests (incl. parametrised cross-household isolation, verified to fail when
the access check is broken); Docker image verified locally in production mode. The actual
Railway deploy is a manual step for the developer: [deploy.md](deploy.md).

---

## Phase 2 — Demo family generator & synthetic dataset 🎓 (~1–2 weeks)

**Build**
- `llm.py` — the only module importing the Gemini SDK. Structured JSON output
  with a response schema, per-item defensive parsing, token cap, env-gated.
- Family definition API: composition (adults/kids by gender, kids' ages) + assets.
- Ticket generation: Gemini produces Dutch tickets in batches with an *intended*
  category and effort. Prompts vary style (short/long, typos, informal) to
  avoid a too-clean dataset.
- Seed dataset: generate several diverse families → export to a versioned dataset.
- Freeze **holdout-sim-v1** (manifest of topic IDs + hash).

**🎓 Lessons**
- What a *dataset* is for ML: features, labels, examples; why the label definition matters.
- **Weak labels**: LLM-assigned labels are noisy guesses, not ground truth.
- **Label noise & class imbalance**: why they exist and why they change the metrics you should use.
- **Data provenance**: why every row carries `source`.
- **Exploratory data analysis (EDA)** notebook: class balance, text length,
  vocabulary, near-duplicates (a major problem with LLM-generated data).
- Why the holdout is frozen *before* any modelling.

**Done when**: a seed dataset of a few thousand Dutch tickets exists, EDA notebook
reviewed, holdout frozen.

**Outcome**: 2,400 tickets (12 families × 200) in production, prompt `gen-v1`, ~345k tokens.
Snapshot `seed-v1` (hash `b81598bf…`). `holdout-sim-v1` frozen: typical-2, typical-5,
city-apartment (600 tickets); training frame 1,715 tickets after excluding holdout families
and 85 exact duplicates of holdout texts. EDA findings in `notebooks/01-eda-seed-dataset.ipynb`
(effort 80% S / 1.4% L, 34% of tickets mention names, ~600 cross-family near-duplicates).
Carry into Phase 3: majority baseline for effort, class weights, exclude *near*-duplicates
of holdout texts, optional name-masking experiment.

---

## Phase 3 — Classifier v1: train, evaluate, serve 🎓 (~2 weeks)

**Build**
- Baselines first: `DummyClassifier` (majority class) and keyword rules.
- Category model: sklearn `Pipeline(TfidfVectorizer → LogisticRegression)`.
- Effort model: same pattern; compare plain multiclass vs an ordinal-aware approach.
- Evaluation module: macro-F1, per-class report, confusion matrix, calibration curve;
  results written to JSON alongside the artifact.
- Model registry (`ml/registry.py`, the only code that reads/writes artifacts):
  save artifact + metadata (sklearn version, data hash, metrics), create
  `ModelVersion`, activate.
- Serving: the active model is loaded at startup and swapped in on promotion;
  `/classify` internal endpoint; topic submission creates real Predictions.
- Confidence threshold for "needs review", chosen from the calibration/precision data.

**🎓 Lessons**
- Why always start with a **baseline**.
- **TF-IDF**: turning text into numbers; n-grams; Dutch-specific choices
  (character n-grams handle Dutch compound words like *vaatwasser*).
- **Logistic regression**: what it learns, regularisation (`C`), reading its coefficients.
- **Pipelines & leakage**: why the vectorizer must be fitted on training data only.
- **Train/validation/test**, stratification, cross-validation.
- **Metrics**: accuracy vs macro-F1, precision/recall, confusion matrices.
- **Calibration**: is "0.8 confidence" actually right 80% of the time?
- **Ordinal targets**: S < M < L, and why getting L wrong as S is worse than as M.
- **Serialization**: joblib, version pinning, why untrusted pickles are dangerous.

**Done when**: submitting a topic returns a real prediction from `ModelVersion 1`,
metrics on holdout-sim-v1 are recorded, and you can explain the confusion matrix.

**Outcome**: `category-v1` (holdout macro-F1 0.929, 95% CI 0.899–0.952; keyword rules 0.662)
and `effort-v1` (ordinal; macro-F1 0.709 vs majority 0.295; L recall 0.5 on 10 tickets) active
in production, promoted by a paired-bootstrap gate. Role tokens (`naamkind`/`naamouder`) replace
household names (needs `Membership.is_child`); temperature scaling (ECE ≈ 0.03 on holdout);
review threshold 0.8. Artifacts in Postgres. An ML review before promotion found 3 blockers
(temperature class order, DB-dependent masking, missing CLI flag), all fixed. Lessons 02 and 03.

---

## Phase 4 — PWA frontend v1 (~3 weeks)

**Build**
- Backend first: Sprint / SprintItem / SprintReview tables + APIs (planning, board moves,
  per-item review feeding labels with `label_source=review`).
- Vite + React + TS, `vite-plugin-pwa` (installable, offline shell), Dutch i18n file.
- Auth screens (email/password + Google), household invites.
- Intake (mobile-first), backlog with **explicit confirm/correct** of labels
  (logs `prediction_was_changed`), sprint planning, kanban (dnd-kit), per-item review.
- Deploy frontend; **your family starts using it** → real data clock starts.
- Freeze **holdout-real-v1** once enough real labelled topics exist (Phase 7+).

**Done when**: family members can install the PWA, submit topics, and run a sprint end-to-end.

**Outcome (2026-09-30)**: sprint API + Dutch PWA live on Railway
(https://ideal-achievement-production-6b63.up.railway.app, Dockerfile + Caddy). Verified end to
end in headless Chromium (phone viewport) locally, and CORS/login/service worker live. Open:
invite emails (Resend needs an own domain; until then invite links are copied from the API
logs), family onboarding, first real sprint.

---

## Phase 5 — Simulator 🎓 (~2 weeks)

**Build**
- **World model**: hidden ground-truth rules per simulated family — task frequencies
  by category and season, who tends to take what, true effort distributions,
  per-person completion rates, noise.
- Simulation runner: fast-forwards N weekly sprints through the *real service layer*
  using the simulated clock — intake → classification → planner confirmation
  (with a configurable "rubber-stamp" rate) → planning → execution → review.
- Weekly topics are drawn from a pre-generated Gemini ticket pool (cheap), and
  the pool is topped up when a scenario needs new kinds of tasks.
- **Drift scenarios** (config): seasonal shift, "family gets a dog" at week 8,
  a new kid's activity, vocabulary change.
- Admin UI: define family → generate tickets → configure & run simulation → view results.
- Every run is reproducible from its `random_seed` + config.

**🎓 Lessons**
- Simulation as a **controlled experiment**: you know the truth, so you can test your tools.
- The circularity problem: a model trained on simulated labels learns the simulator.
- **Types of drift**: data/covariate drift (inputs change), label/prior drift
  (category mix changes), concept drift (the same text now means different effort).
- **Delayed labels**: ground truth arrives only at review, so accuracy always lags.
- **Automation bias**: how rubber-stamping inflates measured accuracy (you'll see it in the sim).

**Done when**: a 26-week simulation with a planted drift runs in minutes and
produces a realistic database for a simulated household.

**Outcome**: world model + Gemini pool + deterministic runner through the real services;
`baseline` and `drift-demo` scenarios; CLI and admin page (create / replay / progress). First
26-week run: concept drift visible only in labels (effort acc 0.69 → 0.49, confidence flat), the
new-typist covariate drift hidden in averages, automation bias measurable by replay. See lesson 04.

---

## Phase 6 — Monitoring: Model Health Dashboard 🎓 (~2 weeks)

**Build**
- Metrics job computing per time-window, per source, per model version:
  macro-F1 on newly labelled data, confidence histogram, category distribution,
  PSI / chi-square vs the training distribution, correction rate at backlog review.
- Admin dashboard (Recharts): time series, detected drift vs planted drift
  (for simulated households), retraining log.

**🎓 Lessons**
- Monitoring **without labels** (confidence, input distributions) vs **with labels** (accuracy).
- **PSI** and **chi-square**: what they compute, thresholds, and why small samples
  make them noisy (a real issue for one family).
- Windowing choices and false alarms.
- Reading a dashboard critically: is this drift, or just noise?

**Done when**: the dashboard shows the planted drift from a Phase 5 simulation, and
you can explain why it triggered when it did.

**Outcome**: ml/monitoring.py (PSI + chi-square with smoothing and small-bin merging, Wilson
intervals, rolling windows, onset-based detection summary), reference profiles stored per model
version (backfilled for v1), admin dashboard "Modelgezondheid" (accuracy with bands and truth,
PSI small multiples with alarm dots, category mix, detection table, model log). Validated against a
control (baseline) run, locally and in production: the dog event is confounded with the September
season; overall-accuracy detection of the concept drift did not replicate; per-category effort
segments show it clearly (groceries 0.52 → 0.12) but one household lacks the statistical power for a
Bonferroni-corrected alarm (needs ~2-3 months or pooled households). See lesson 05.

---

## Phase 7 — Retraining loop 🎓 (~2 weeks)

**Build**
- `ml/retrain.py`: select training-eligible, labelled data (excluding all frozen
  holdouts) → snapshot + hash → train challenger → evaluate champion and challenger
  on every holdout, per source → **promotion gate** (challenger must beat champion on
  agreed metrics without regressing on real data) → register → activate or reject.
- `/retrain` Claude skill + an admin button; later a Railway cron.
- Experiments: real-sample weighting; how much real data closes the sim-to-real gap.
- Demo scenario: planted drift → dashboard alarm → retrain → recovery.

**🎓 Lessons**
- **Champion/challenger**, promotion gates, rollback.
- **Data versioning & reproducibility**: why the training data hash matters.
- **Sim-to-real gap** and sample weighting.
- **Learning curves**: does more data still help?
- How this maps to MLflow / a real model registry (ADR update).

**Done when**: at least one real-data retraining event and one simulated
drift-recovery event are visible in the retraining log.

---

## Phase 8 — Embeddings & planning suggestions 🎓 (~2 weeks)

**Build**
- `fastembed` with a multilingual model; embeddings stored in pgvector.
- Similar-task retrieval for sprint planning: top-k similar past topics within the
  household → who handled them, actual effort, completion.
- Experiment: embeddings + LR vs TF-IDF + LR on both holdouts. Promote only if it wins.

**🎓 Lessons**
- What **embeddings** are, and why they capture meaning TF-IDF can't.
- Cosine similarity, k-nearest-neighbours, approximate indexes (HNSW) in pgvector.
- Evaluating retrieval (precision@k) as opposed to evaluating classification.
- Fair model comparison: same holdouts, same metrics, significance with small n.

---

## Phase 9 — Story & polish (~1 week)

- README: architecture diagram, skill-map walkthrough, honest framing of what's
  lightweight and how it would scale (MLflow, feature store, scheduled pipelines).
- Interview walkthrough script (< 10 min).
- Stretch: Planner Assistant agent (Gemini) proposing a draft sprint.

---

## Claude Code setup (done in Phase 0)

| Piece | Type | Purpose |
|---|---|---|
| `CLAUDE.md` (root) | memory | Project purpose, layout, commands, invariants, scope guard. Refined with `/init` once the scaffold exists. |
| `backend/ml/CLAUDE.md` | nested memory | ML rules (leakage checklist, metric definitions) + **teaching mode**: explain concept → build → interpret results. Loads only when working in ML code. |
| `protect-frozen` | PreToolUse hook | Blocks edits to frozen holdout manifests, model artifacts, and `.env*`. |
| `format` | PostToolUse hook | `ruff format/check` on `.py`, `prettier` on `.ts/.tsx`. |
| `ml-reviewer` | subagent (read-only) | Reviews `ml/` + `sim/` changes for leakage, holdout contamination, source mixing, missing version tags. |
| `/retrain` | skill (user-invoked only) | Runs retraining, interprets metrics, asks before promotion. |
| `/simulate` | skill (user-invoked only) | Runs a simulation from a config, summarises what drift was planted and what was detected. |
| existing: `python-react-railway-conventions`, `external-service-setup`, `code-conventions-auditor` | global | Code conventions, Railway/Vercel/Google OAuth wiring, structural audits. |

Skills are created when their phase arrives (not all up front), so they describe
real commands.

---

## Key risks

| Risk | Mitigation |
|---|---|
| Synthetic data too clean/repetitive → inflated metrics | Style variation in prompts, near-duplicate detection in EDA, real holdout as the final judge |
| Small real sample → noisy metrics | Report confidence intervals; don't over-read single-sprint changes |
| Family stops using it | Ship Phase 4 early; keep intake to one screen |
| Gemini cost/format drift | Token caps, pre-generated ticket pools, schema-validated output |
| Scope creep | Skill-coverage map as the gate (PRD §8) |
