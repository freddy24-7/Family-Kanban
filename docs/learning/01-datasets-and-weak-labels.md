# 01 — Datasets, weak labels and the frozen holdout

Phase 2. Code: `backend/sim/` (generator), `backend/ml/datasets.py`, `backend/ml/holdout.py`.

## 1. What a dataset is, in this project

A supervised ML dataset is a table of **examples**. Each example has **features**
(what the model may look at) and a **label** (what it must predict).

| Column | Role | Example |
|---|---|---|
| `text` | **feature** — the only input the classifier sees | *Gras maaien en de heg snoeien* |
| `category_label` | **label** (target of the category model) | `home_maintenance` |
| `effort_label` | **label** (target of the effort model) | `M` |
| `household_id`, `source`, `label_source`, `generation_run_id` | **metadata** — never features; used to split, filter and explain | `simulated`, `generator` |

The distinction matters: if metadata leaks into features (e.g. the model can see which
family wrote a ticket), it can "cheat" in evaluation and then fail on a new family.

## 2. The label definition is a modelling decision

Before any model, you must decide what each label *means*
([labeling-guidelines.md](../../backend/app/labeling_guidelines.md)). Our first Gemini test labelled
*"Hondenbrokken kopen"* as `groceries`; someone else might say `chores` or `other`.
Without a rule, the "truth" is inconsistent, and no model can beat the inconsistency.

> **Bayes error / label noise ceiling**: if two careful humans agree on only 85% of
> labels, a model that scores 85% against one of them may already be as good as it gets.
> Later, measure this: label a sample twice and compute agreement.

## 3. Weak labels

Labels produced by a cheap, imperfect process (here: Gemini assigning the category it
had in mind while writing the ticket) are **weak labels**. They are useful for getting
started, but:

- They contain **noise**: some are simply wrong.
- The noise is **systematic**, not random: Gemini has its own habits (e.g. always
  calling garden work `home_maintenance`, writing unusually clean sentences). A model
  trained on them learns Gemini's habits, not your family's.
- So metrics measured against weak labels tell you how well you imitate the generator.
  Only **human-confirmed labels on real data** tell you how well you do the real job.
  That's why every row carries `source` and `label_source`, and metrics are always
  split by source.

## 4. Synthetic data is too clean

LLM-written text is grammatical, complete and repetitive. Real input is short, lowercase,
full of typos and abbreviations (*"gras maaien"*, *"dierenwinkel!!"*, *"apk"*). To
reduce the gap we vary the **style** in prompts (short, sloppy, with typos, long) and
check for **near-duplicates** (the same ticket phrased identically across families).
The remaining gap between synthetic and real is the *sim-to-real gap* we measure later.

## 5. Data versioning: why a snapshot, not a script

Running the generator twice gives different tickets: LLM output isn't reproducible,
even with the same prompt. So the reproducible thing is not the generator run but the
**dataset snapshot**: an exported file plus a content **hash**. Every trained model
records the hash of the data it was trained on, so you can always answer "what exactly
did model v3 learn from?".

## 6. The frozen holdout

A **holdout** is data the model never sees during training or tuning, used only to
estimate performance on new data. Rules:

1. **Freeze it before any modelling.** If you look at holdout results while choosing
   models or thresholds, you slowly fit to it and the estimate becomes optimistic.
2. **Never change it.** Comparing model v1 and v5 only works on identical test data.
   New holdouts can be *added* (e.g. `holdout-real-v1`); old ones stay.
3. **Split the way the model will be used.** Our model will face *new families*, so we
   hold out **whole families** (a *group split*), not random tickets. Otherwise tickets
   from the same family (same names, same pets, same phrasing) end up on both sides and
   inflate the score.
4. **Exclude duplicates.** If the exact same text appears in both training and holdout,
   the model can just memorise it. Training excludes any text whose normalised hash is
   in a holdout manifest.

## Self-check

1. Why is `household_id` not a feature, even though it might improve accuracy?
2. A model scores 95% on the simulated holdout and 70% on real data. Name two reasons.
3. Why do we split by family instead of by ticket?
4. You change the rule "garden work is `home_maintenance`" to `chores`. What happens to
   comparisons between models trained before and after?
