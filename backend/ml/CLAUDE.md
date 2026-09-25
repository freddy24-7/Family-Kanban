# ML package rules

## Teaching mode

The developer is learning classical ML through this project. For any ML step
(data prep, features, models, evaluation, calibration, drift, retraining, embeddings):

1. **Before building**: explain the concept in plain language — what it is, why this
   project needs it, and the common mistake it avoids. Short and concrete, using this
   project's data as the example.
2. **Build** it, keeping the code readable over clever (it will be studied).
3. **After running**: walk through the output — what each number means, what a good vs
   worrying result looks like here, and what to try next.
4. Record the lesson in `docs/learning/NN-topic.md` (concept, how it shows up in this
   codebase, how to read the results, one or two self-check questions).

Prefer to show experiments in `notebooks/` first, then move what ships into `backend/ml/`.

## Correctness checklist (leakage and evaluation)

- Split **before** fitting anything; vectorizers/scalers live inside an sklearn `Pipeline`.
- Frozen holdouts are excluded from every training query — check by topic ID.
- No post-hoc fields as features: `effort_actual`, `completed`, review notes, labels.
- Near-duplicate texts (common in LLM-generated data) must not straddle train/test.
- Stratify splits by label; report **macro-F1** plus per-class metrics, not accuracy alone.
- Always report metrics **per source** (`real` / `simulated`) and per holdout.
- Compare every model against a baseline (`DummyClassifier`) on the same holdout.
- Set and record `random_state` everywhere; record the training data hash.
- Small samples: report confidence intervals / support; don't over-read small differences.

## Registry

- Only `registry.py` reads/writes artifacts (joblib + metadata JSON: sklearn version,
  data hash, metrics, parent version). Never load artifacts from untrusted sources.
- Model versions are immutable; retraining creates a new version.
