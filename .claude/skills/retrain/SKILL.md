---
name: retrain
description: Run the Family Kanban retraining loop safely - snapshot, train challengers, gate against the champion on all frozen holdouts, shadow-replay an unseen simulated world, and only promote with explicit user approval. Use when the user asks to retrain, evaluate a new model, promote or roll back a model version.
disable-model-invocation: true
---

# Retraining procedure (lesson: docs/learning/06-retraining.md)

Work from `backend/`. Production commands need `DATABASE_URL` set to the pgvector service's
public URL (`railway variables -s pgvector --json`); training itself is fine from a laptop, but
long simulations must run inside Railway (admin page), never over the public proxy.

1. **Status.** `uv run python -m ml.retrain status`: which versions are active, their holdout scores.
2. **New holdout first (if the world changed).** A holdout for new data must be frozen *before* any
   model trains on that data: `uv run python -m ml.holdout freeze-sim --run <id> --from-week <k> --name <name>`
   (simulated worlds) or a temporal split for real data. Commit the manifest.
3. **Snapshot.** `uv run python -m ml.retrain snapshot --name train-YYYY-MM-DD`. Report rows by
   source and where the effort labels come from (review / planner / generator).
4. **Dry run.** `uv run python -m ml.retrain run --snapshot <name> --dry-run [--real-weight N]
   [--primary-holdout <name>]`. Show the user, per task and per holdout: challenger vs champion
   macro-F1, the paired-bootstrap gain with its 95% interval, and the gate decision. Explain any
   regression on an old holdout (stale world vs real problem).
5. **Register candidates** (no `--dry-run`, no `--promote`) if the user wants to evaluate further.
6. **Shadow evaluation.** Replay an *unseen* simulated world with the candidates pinned (admin page
   → Simulaties → Opnieuw afspelen with model versions, or `sim.simulate replay ... --category-model
   ... --effort-model ...` for local data). Compare with the champion's replay on the dashboard.
7. **Promote only with explicit user approval**: `uv run python -m ml.retrain run --snapshot <name>
   --promote` (the gate still applies). Then check the API picks it up (a new ticket's prediction
   carries the new version).
8. **Rollback** if anything looks wrong: `uv run python -m ml.retrain rollback --task <category|effort>`.

Never: train on a frozen holdout, edit a manifest, loosen the gate to get a model through, or
promote without the user saying so.
