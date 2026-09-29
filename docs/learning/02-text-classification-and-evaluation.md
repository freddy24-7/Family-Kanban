# 02 — Text classification: baselines, TF-IDF, logistic regression, honest evaluation

Phase 3. Code: `backend/ml/models.py`, `backend/ml/evaluate.py`, `backend/ml/train.py`,
notebook `notebooks/02-model-experiments.ipynb`.

## 1. Always start with a baseline

A baseline is the dumbest thing that could work. It tells you what a score *means*.

- **Majority baseline**: always predict the most common class. On our effort labels
  (80% `S`) it scores **80% accuracy** while knowing nothing.
- **Keyword baseline**: a handful of hand-written rules (*"belasting" → finance*). If ML
  can't clearly beat twenty lines of rules, it isn't worth deploying.

Every model result in this project is shown next to its baselines.

## 2. From text to numbers: TF-IDF

A classifier needs numbers. **Bag of words**: each ticket becomes a long vector with one
position per word in the vocabulary, counting how often it occurs. Word order is ignored.

**TF-IDF** re-weights those counts: *term frequency* (how often in this ticket) ×
*inverse document frequency* (rare across all tickets = more informative). "de" appears
everywhere → weight ≈ 0; "dakgoot" appears rarely → high weight.

Two ways to cut text into features:

| | Word n-grams | Character n-grams |
|---|---|---|
| Unit | `gras`, `maaien`, `gras maaien` | `gra`, `ras`, `maa`, `aie`, … |
| Handles typos | no (*stoffugen* ≠ *stofzuigen*) | mostly (shares most n-grams) |
| Handles Dutch compounds | no (*vaatwasser* ≠ *vaat*) | yes (*vaatwasser* contains *vaat*) |
| Risk | sparse, misses variants | more features, slower |

Dutch writes compounds as one word, and our tickets are full of typos, so character
n-grams should help. We'll test it rather than assume it.

## 3. Logistic regression

For each class, logistic regression learns one **weight per feature**. A ticket's score
for a class is the sum of weights of its features; a *softmax* turns the scores into
probabilities that add up to 1. So:

- It is **interpretable**: the largest weights for `home_maintenance` should be words like
  *dakgoot*, *kraan*, *apk*. If *lieke* shows up as a top `kids` feature, the model has
  learned a family-specific shortcut.
- **Regularisation** (`C`) limits how large weights may grow. Small `C` = strong
  regularisation = simpler model that generalises better but may underfit. We pick `C`
  by cross-validation.
- **Class weights** (`class_weight="balanced"`) make mistakes on rare classes cost more,
  so the model doesn't just ignore `L` or `other`.

## 4. Pipelines prevent leakage

The TF-IDF vocabulary and IDF weights are *learned from data*. If you fit the vectorizer on
all data (including test rows) before splitting, test information leaks into training.
An sklearn `Pipeline(vectorizer → classifier)` is fitted as one unit, so in every
cross-validation fold the vectorizer only sees that fold's training part.

## 5. Choosing a model without touching the holdout

Every time you look at the holdout score and change something, you tune to the holdout
and its score stops being an honest estimate. So:

1. Compare models with **cross-validation on the training families only**.
   We use **GroupKFold by household**: each fold holds out whole families, mirroring the
   holdout design (9 training families → 3 folds of 3 families).
2. Choose the configuration from the CV results.
3. Evaluate the chosen model **once** on `holdout-sim-v1`, and report that number.

## 6. Metrics

- **Accuracy**: share correct. Misleading under imbalance (see baseline).
- **Precision** for a class: of the tickets we *called* `finance`, how many were finance.
  **Recall**: of the actual `finance` tickets, how many did we find. **F1**: their
  harmonic mean.
- **Macro-F1**: average F1 over classes, each class counting equally. Our headline metric.
- **Confusion matrix**: which classes get mixed up (expect `chores` ↔ `home_maintenance`,
  `kids` ↔ everything, since kids' tasks look like other categories).
- **Support** matters: an F1 computed on 16 `other` tickets can swing ±15 points from a
  couple of tickets. We report **bootstrap confidence intervals**: resample the test set
  many times and look at the spread of the score.
- **Effort is ordinal** (S < M < L). Besides macro-F1 we report the **mean absolute error
  in steps** (S→L = 2 steps wrong, S→M = 1), and the recall of `L` specifically.

## 7. Calibration: what does "80% confident" mean?

A model is **calibrated** if, among all predictions made with ~80% confidence, ~80% are
correct. Logistic regression is often reasonably calibrated, but class weighting and
strong regularisation distort it. We check with a **reliability diagram** (confidence
bins vs actual accuracy) and the **expected calibration error (ECE)**.

This matters for the product: tickets below a confidence threshold are flagged "needs
review". The threshold is chosen from **cross-validation predictions** (e.g. "flag what we
get right less than 90% of the time"), never from the holdout.

## Self-check

1. Why can a model with 80% accuracy be worse than useless?
2. Why do character n-grams suit Dutch family tickets?
3. What goes wrong if you fit TF-IDF on all data before cross-validation?
4. Why GroupKFold by household instead of a plain random KFold?
5. A model says 0.9 confidence but is right only 70% of the time at that level. What is
   that called, and why does it matter for the "needs review" flag?
