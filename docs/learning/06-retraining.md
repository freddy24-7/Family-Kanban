# 06 — The retraining loop: champion, challenger and honest comparison

Phase 7. Code: `backend/ml/retrain.py`, `backend/ml/train.py`, `backend/ml/holdout.py`,
`backend/ml/registry.py` (rollback), shadow replays in `backend/sim/runner.py`.

## 1. What changes between v1 and v2 is the data, not the code

Retraining = same pipeline, **new labelled data**. So the first questions are about data:

- **Which labels?** Category: the planner's confirmation (or the generator's weak label for seed data).
  Effort: the **actual effort from the sprint review** when it exists (the truth we care about), else
  the planner's estimate, else the generator's guess. Every training row records which one it used.
- **Which households?** Only `training_eligible` ones, never frozen holdout topics or their (near)
  duplicates, never deleted tickets.
- **Which snapshot?** Every retrain freezes its exact input as a new snapshot (rows + member lists +
  hash). "What did v2 learn from?" always has an exact answer (*data versioning*).

## 2. Champion vs challenger, on the same frozen tests

The active model is the **champion**; the new one the **challenger**. Both are scored on the **same
frozen holdouts** with the **same labels**, and the gate uses a **paired bootstrap** of the difference
(lesson 03). Two rules:

- the challenger must be **clearly better** on the holdout that represents today's world;
- it must **not be clearly worse** anywhere else (no silent regressions).

## 3. When the world changes, old holdouts go stale

After concept drift ("groceries now take longer"), the old frozen holdout still says groceries are
small tasks. A model that learned the new world will look *worse* on the old holdout. That's not a
bug in the model; the holdout describes a world that no longer exists. You then need a **new holdout
from the new world** (frozen *before* training on it), and a policy for how much weight the old one
still gets. Deciding that is a judgement call you document, not something to hide.

## 4. Shadow evaluation: try before you promote

Promoting a model changes every family's predictions immediately. Before that, run the challenger in
**shadow**: the simulator replays an unseen world with the challenger instead of the champion,
while nothing changes for real users. Same pool + same seed, only the model differs: a clean
before/after on the dashboard.

## 5. Rollback

Every promotion keeps the previous version (status `retired`). Rolling back = re-activating it, one
transaction, no retraining. Because predictions record their model version, the dashboard shows
exactly which weeks were served by which model.

## 6. Sim-to-real: how much real data do you need?

The models learned from Gemini's text. Real text differs (shorter, vaguer, other words). Two levers:

- **sample weights**: count each real ticket as, say, 5 synthetic ones, so the model doesn't drown
  out the few real examples;
- **learning curves**: train on growing amounts of real data and watch the real holdout score. When
  the curve flattens, more of the same data won't help.

Both need a **real holdout**. With one family a group split is impossible, so real data uses a
**temporal split**: the most recent weeks are the test set (the model is always evaluated on the
future, never on the past it trained on).

## 7. When retraining can't fix it: household-specific drift

Retraining effort-v2 on world A improved effort overall but **not** the drifted category: groceries
after the move stayed at 0.18 accuracy. Weighting the reliable review labels barely helped (0.31 at
weight 40), because reviews from *before* the move ("groceries = S") were amplified too. The same
text ("melk halen") has two answers, and a global model has no way to know which family it is.

The drift is **household-specific** (this family moved), so the fix belongs at the household level:
a global model plus a small **per-household adjustment**, learned from that family's own reviews.

**Label-shift correction (Bayes' rule).** For a household and category, compare what the model
expects with what the reviews say:

- π_model(e): the model's average predicted probability of effort e on the household's recent
  reviewed tickets of that category ("what I expected");
- π_household(e): the actual efforts in those reviews, smoothed toward π_model with a few pseudo
  counts so that 2 reviews can't flip everything;

and re-weight each new prediction: p_adj(e) ∝ p_model(e) × π_household(e) / π_model(e).
If the model says "S, 70%" for groceries but this family's groceries have mostly been M, the S
probability is scaled down and M up. Only the **last few reviews** per category are used, so the
adjustment follows change and forgets the old world: the recency the global model lacks.

It only adjusts the label *mix* per segment; it can't fix a model that doesn't understand a text.
And it's evaluated like any change: in a shadow replay of an unseen world, before switching it on.

## What happened in this project

1. **Data**: world A (a 26-week drift-demo run, 327 labelled tickets, 310 with a sprint-review effort)
   became training data. World B, an unseen drift-demo world, was frozen as `holdout-drift-v1`
   (210 tickets after the move, hidden-truth labels) **before** training.
2. **Gate** (snapshot `train-2026-09-30`): effort-v2 +0.143 macro-F1 on the new world (95% CI
   +0.065..+0.219), −0.020 on the old world (within noise) → pass. category-v2 +0.008 (not
   significant) → correctly not promoted: more data without a real gain only adds risk.
3. **Per-category view** exposed what the overall score hid: effort-v2 still scored **0.18 on
   groceries** after the move, exactly like v1. Weighting review labels (×40) only reached 0.31,
   because pre-move reviews were amplified too.
4. **Shadow replays of world B** (same tickets, only the model differs), against the hidden truth:

| After the move | v1 | effort-v2 | v2 + household adjustment |
|---|---|---|---|
| Effort accuracy | 0.53 | 0.58 | 0.60 |
| Effort MAE (steps) | 0.57 | 0.47 | 0.44 |
| Groceries | 0.18 | 0.18 | 0.33 |

   Before the move both v2 variants score 0.68 (the adjustment does no harm when nothing drifted).
   A first shadow replay was invalid: pinning only the effort model removed the category model
   entirely (fixed, with a regression test).
5. **effort-v2 promoted** (the evaluated candidate itself, gate re-checked), effort-v1 retired.
   On the developer's real tickets it now says M for the vet visit (labelled M) and L for clearing
   the attic. The household adjustment stays off in production until it's tuned on world A.

## Self-check

1. Why is the sprint-review effort a better training target than the planner's estimate?
2. A challenger is better on the new-world holdout but worse on the old one. What do you do?
3. Why replay an *unseen* world in shadow, not the world the challenger was trained on?
4. Why a temporal split for one family's data instead of a random one?
5. Why can't more data fix household-specific concept drift, and what does the adjustment add?
