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

## Self-check

1. Why is the sprint-review effort a better training target than the planner's estimate?
2. A challenger is better on the new-world holdout but worse on the old one. What do you do?
3. Why replay an *unseen* world in shadow, not the world the challenger was trained on?
4. Why a temporal split for one family's data instead of a random one?
