# 04 — Simulation as a controlled experiment, and the three kinds of drift

Phase 5. Code: `backend/sim/world.py` (hidden ground truth), `backend/sim/pool.py`
(Gemini ticket pool), `backend/sim/runner.py` (weekly simulation), `backend/sim/report.py`.

## 1. Why simulate at all?

With real data you never know the truth for sure: labels are human judgements, and when
accuracy drops you can't tell *why*. In a simulation **you decide the truth** (a hidden
*world model*) and **you decide when the world changes**. That turns monitoring into an
experiment: "I planted a change in week 12; does my drift detector fire in week 12?"

This is how you test monitoring and retraining *before* you trust them on real data.

## 2. The world model (hidden ground truth)

For every simulated family and every week, the world model decides:

- **how many tickets** arrive (varies per week),
- **which category** each ticket truly is: family weights (a dog → more chores) × **season**
  (garden work in summer, school in September, gifts and finance in December),
- **the true effort** of each ticket (a distribution per category),
- **who submits** and **who does** a task, and whether it gets **completed**.

Gemini only *writes the text* for a ticket whose true category and effort were decided
first. So the truth is known by construction; Gemini's own opinion is stored too, as the
weak label it is.

## 3. The circularity problem

If the simulator used the same rules the model learned from, the model would look perfect.
Two things keep it honest: the simulated families are **new** (not in training data), and
their texts are **freshly generated** (not the training tickets). And we accept the
limitation: a model trained on Gemini text, tested on Gemini text, measures *consistency*
more than real-world quality. Real data (your family) is still the final judge.

## 4. The three kinds of drift

A model learns P(label | text). Drift is anything that changes the data after training:

| Kind | What changes | Example planted in the simulator | Visible without labels? |
|---|---|---|---|
| **Covariate / data drift** | the input P(text) | a child starts typing: shorter, sloppier tickets | **yes**: confidence drops, text statistics change |
| **Prior / label drift** | the mix of labels P(label) | summer: more garden work; December: more social/finance; a dog arrives | partly: the *predicted* category mix shifts |
| **Concept drift** | the relationship P(label \| text) | the family moves further from the shop: *"boodschappen doen"* now takes M, not S | **no**: same texts, same predictions, only labels reveal it |

Concept drift is the hardest: the model is as confident as ever and simply wrong. Only
monitoring against (delayed) labels catches it.

## 5. Delayed labels

The true effort of a task is only known at the **sprint review**, a week or more after the
prediction. So accuracy monitoring always lags. Label-free signals (confidence, input
statistics, predicted mix) react immediately but can't see concept drift. You need both.

## 6. Automation bias, measured

The simulated planner doesn't always check carefully: with some probability they
**rubber-stamp** the model's prediction. Because the simulator knows the truth, we can compare:

- **true accuracy**: prediction vs the world model's truth;
- **measured accuracy**: prediction vs the planner's labels.

Rubber-stamping makes measured accuracy look better than true accuracy, a gap that is
invisible with real data alone. That is why the backlog UI leaves uncertain fields empty.

## 7. Reproducibility: pool vs run

Gemini isn't deterministic, so a simulation is split in two:

1. **Pool** (once, uses Gemini): the tickets for every week, with their hidden truth, stored.
2. **Run** (repeatable, no LLM): replays the pool through the real app services with a
   seeded random generator and a fast-forwarding clock. Same pool + same seed = same run.

So you can re-run the same world with a sloppier planner, or after retraining a model, and
compare like with like.

## What happened in this project

First 26-week run of `drift-demo` (July 2026 → January 2027, 366 tickets, one Gemini pool):

| Weeks | Category acc (truth) | Effort acc | Effort MAE | Category confidence | Flagged |
|---|---|---|---|---|---|
| 1-8 (seasons only) | 0.94 | 0.69 | 0.35 | 0.94 | 0.30 |
| 9-12 (dog arrives) | 0.96 | 0.72 | 0.32 | 0.93 | 0.47 |
| 13-16 (**concept drift**: groceries take longer) | 0.91 | **0.49** | **0.55** | **0.94** | 0.52 |
| 17-26 (8-year-old starts typing) | 0.92 | 0.54 | 0.54 | 0.93 | 0.38 |

- **Concept drift is invisible without labels**: effort accuracy collapses (0.69 → 0.49) while the
  model's confidence doesn't move (0.94). Only the sprint-review labels reveal it.
- **Weekly numbers are noisy** (9-28 tickets a week: week 14 shows 0.33 effort accuracy). Detectors
  need windows and significance tests, not single weeks (Phase 6).
- **The new typist is barely visible in averages** (mean words 7.0 → 6.8): a small change in a
  small subgroup disappears in aggregates. Detecting it needs per-segment or distribution-level
  tests (Phase 6).
- **Automation bias**: replaying the same world with a planner who rubber-stamps 90% of confident
  predictions lifted *measured* category accuracy above the *true* accuracy (0.94 vs 0.92). The
  effect stays small because the UI only pre-fills confident predictions.
- Effort accuracy is lower than on the frozen holdout (0.69 vs 0.90) even before any event: the
  world model has more M/L tasks than Gemini's seed data. That's a built-in label-prior shift, a
  small-scale version of the sim-to-real gap.

## Self-check

1. Which drift type can't be detected from inputs or predictions alone, and why?
2. Why do we store Gemini's labels but use the world model's labels as truth?
3. What does it mean when measured accuracy is higher than true accuracy?
4. Why split a simulation into a pool and a run?
