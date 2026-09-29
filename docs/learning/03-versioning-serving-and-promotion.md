# 03 — Model versioning, serving and the promotion gate

Phase 3. Code: `backend/ml/registry.py`, `backend/ml/train.py`, `backend/ml/serving.py`,
`backend/app/services/classification.py`.

## 1. Serve the pipeline before the model exists

Phase 1 shipped a *stub* model (always `other`/`M`, confidence 0). Versioning, prediction
logging and the "needs review" flag worked end to end before any ML. The first real model
then replaced the stub by changing one database row, not any API code.

## 2. A model version is immutable and fully described

Each `ModelVersion` records: the artifact (bytes + sha256), the scikit-learn version, the
training snapshot and the hash of the exact training rows, the configuration, the text
preprocessing it expects, the git commit, CV metrics and holdout metrics. Retraining
never edits a version; it creates a new one. That answers "what exactly did model v3
learn from, and how good was it?" forever.

Every **prediction** stores the version that made it, so when labels arrive later,
accuracy can be attributed to the right model (Phase 6).

## 3. Training/serving skew

If the model sees different input in production than in training, it silently degrades.
Here the risk was the role tokens (*"Lieke"* → `naamkind`). Protections:

- one shared function (`ml/text.py`) used by both training and serving;
- the model declares its preprocessing (`role-mask-v1`) and serving refuses unknown ones;
- the household member lists are stored **in the snapshot**, so training can't silently
  run unmasked because it happened to point at a database without those families.

That last one was a real bug found in review: against the local database, masking was a
no-op while production serving did mask.

## 4. Artifacts and pickles

Models are stored with `joblib` (pickle). Two rules: only load artifacts your own code wrote
(unpickling runs code), and only load with the same scikit-learn major.minor version (the
registry checks). The API caches loaded models in memory by URI; that's safe because
versions never change.

## 5. The promotion gate

A new model (challenger) replaces the active one (champion) only if it is **clearly**
better on every frozen holdout:

- **Paired bootstrap**: resample the same holdout rows for both models many times and look
  at the *difference* in macro-F1. Pairing removes the noise both share (which tickets
  happened to be sampled), so it detects real gains that two separate confidence intervals
  would miss. The 95% interval of the gain must lie above zero.
- Blocks if frozen holdout topics are missing, and (once real data exists) if the
  challenger is worse on real-data rows.
- With no active model, the challenger must beat the majority baseline.

## 6. What happened in this project

| | category-v1 | effort-v1 |
|---|---|---|
| Holdout macro-F1 (95% CI) | **0.929** (0.899–0.952) | **0.709** (0.609–0.791) |
| Majority baseline | 0.054 | 0.295 (accuracy 0.79!) |
| Keyword rules | 0.662 | – |
| Holdout calibration error (ECE) | 0.026 | 0.031 |
| Temperature | 0.39 (was underconfident) | 0.62 |

Weakest spots: `kids` (0.89 F1, overlaps with every other category), `finance` recall
(0.79), and effort `L` (F1 0.46 on only 10 holdout tickets). Remember: these are scores on
*synthetic* families. Real tickets will score lower, and that sim-to-real gap is what
Phase 4 onward measures.

## Self-check

1. Why store the git commit and the training data hash with every model?
2. What is training/serving skew, and how could it happen in this project?
3. Why is a paired bootstrap more sensitive than comparing two confidence intervals?
4. Why is it acceptable to cache loaded models forever in the API process?
