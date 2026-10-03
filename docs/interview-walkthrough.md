# Interview walkthrough (≤ 10 minutes)

A script for presenting Family Kanban, not a text to read aloud. Each part has a time box,
what to show, the point to land and one line to say. After the script: likely questions with
short answers you can defend from first principles.

**Before the call**: log in on the live app (phone or narrow browser window) and in a second
tab as admin (Beheer → Monitoring, Simulaties). Have the repo open at
[docs/learning/](learning/). Mute notifications.

---

## 1. The hook (0:00–0:45)

**Show**: nothing yet, or the README's lifecycle diagram.

**Point**: why this project exists.

> "I'm strong in LLM integration and full-stack, and I wanted hands-on classical ML and MLOps:
> training, versioning, monitoring, drift and retraining. So I built a household kanban that
> my family actually uses, with a classifier in the loop, and a simulator that runs the whole
> app for generated families so I can plant drift at a known week and see if I catch it."

## 2. The product, 60 seconds (0:45–1:45)

**Show** (phone view): type "melk halen" → the predicted category and effort appear with
confidence; the Backlog (planner confirms); Plan with "Eerder gedaan"; the Board; the Review.

**Point**: the two labelling moments are product features, not chores.

> "Every task gets two labels for free: the planner confirms the category when planning, and
> the sprint review records how big it *really* was. That second one is the ground truth the
> model can't see in the text."

## 3. The model and how it's evaluated (1:45–3:15)

**Show**: [lesson 03](learning/03-versioning-serving-and-promotion.md) table, or `ml.retrain status`.

**Points**:
- TF-IDF (word + character n-grams) + logistic regression; effort as an **ordinal** model
  (S < M < L); confidence **temperature-scaled** so 80% means 80%.
- Holdouts split **by household** and frozen in git, near-duplicates excluded: LLM-generated
  data repeats itself, so a random split would leak.
- Macro-F1 with bootstrap intervals, always next to a baseline: category 0.93 vs 0.05 for
  "always the most common class", effort 0.71 vs 0.30.

> "Accuracy alone would have fooled me: the majority baseline reaches 0.79 accuracy on effort,
> because most tasks are small. Macro-F1 says 0.30."

## 4. Versioning and the promotion gate (3:15–4:30)

**Show**: Monitoring → model log (versions, which is active), or [lesson 06](learning/06-retraining.md).

**Points**:
- Every prediction stores the model version that made it, so accuracy is attributable.
- A new model is a **challenger**; it must be **clearly better** (paired bootstrap, 95% interval
  above zero) on today's world and **not clearly worse** on older holdouts. Rollback is one
  command.
- effort-v2 passed (+0.14, CI +0.07..+0.22) and was promoted; category-v2 gained +0.008, not
  significant, and was **not** promoted.

> "More data without a measurable gain is only risk, so category-v2 stayed a candidate."

## 5. Drift: the honest part (4:30–7:00)

**Show**: Beheer → Monitoring for the drift-demo run (event lines at the dog and the effort
shift), then the control run.

**Points**, as a story:
1. **Concept drift is invisible without labels**: groceries start taking longer; effort accuracy
   falls 0.69 → 0.49 while confidence stays at 0.94.
2. **The dog**: the detector fired in exactly the right week. Then I ran a **control world
   without a dog**, and it fired in the same week: September, school starts. Confounded.
3. **Replication**: the concept-drift detection worked locally and did *not* replicate in a
   second world. With ~20 tickets a week, one household is underpowered. I didn't loosen the
   thresholds to make it appear: that would be p-hacking.
4. **Retraining didn't fix the drifted category**: effort-v2 improved overall, but groceries
   stayed at 0.18. A global model can't know *which* family moved. A per-household adjustment
   from the family's own recent reviews (label-shift correction, Bayes' rule) lifted it to
   0.33 in a shadow replay, and it isn't switched on until it's tuned on a separate world.

> "The most valuable thing the simulator gave me was the failures: a false discovery, a
> detection that didn't replicate, and a retrain that improved the average and not the problem."

## 6. Embeddings: measuring beats assuming (7:00–8:15)

**Show**: [lesson 07](learning/07-embeddings.md) "What happened" tables.

**Points**:
- The plan was embeddings + pgvector. A multilingual sentence model lost on every test:
  retrieval precision@3 0.61 vs 0.67; as classifier features 0.87 vs 0.94 and 0.53 vs 0.63.
- Why: it was trained for **paraphrase invariance**. "Melk halen" ≈ "grote weekboodschappen",
  and for effort that difference *is* the signal.
- So production serves TF-IDF retrieval, no 400 MB model on the server. And the family's own
  history beat the model on drifted groceries (0.42 vs 0.00, n = 19): it's shown to the
  planner as history, not as a prediction, until more than one world confirms it.

> "I shipped the simpler thing because I measured it. Fair comparison meant the same data, the
> same tuning budget chosen without the holdouts, and a paired test."

## 7. Architecture and scale (8:15–9:15)

**Show**: README architecture diagram and the "deliberately lightweight" table.

**Points**: FastAPI + Postgres + React PWA on Railway; multi-tenant from day one (every query
household-scoped, cross-household returns 404, tested per endpoint); one module per external
dependency; ~140 tests. The registry is a Postgres table that maps 1:1 to MLflow; the retrain
CLI is the pipeline a scheduler would run.

> "At household scale MLflow and Airflow would be more ops than value. I designed the pieces so
> each one maps onto the company-scale tool."

## 8. Close (9:15–10:00)

**What's next / limits, said plainly**:
- Real data is the gap: the model is trained mostly on synthetic text, and my family has only a
  handful of labelled real tickets. The next step is a real holdout with a **temporal** split,
  and measuring the sim-to-real gap.
- Every lesson, including the failures, is in `docs/learning/`.

> "If I did this at work, I'd keep the same discipline: a control group, a frozen test set, a
> gate, and the willingness to report what didn't work."

---

## Likely questions

**Why logistic regression and not a transformer?** A few thousand short Dutch texts, CPU
hosting, and I need calibrated probabilities and fast retraining. LR on TF-IDF is strong for
short texts, trains in seconds, and is explainable. I did test a pretrained encoder (as
features): it lost.

**Why macro-F1?** Classes are imbalanced (most tasks are small). Macro-F1 weighs every class
equally, so a model that ignores "L" is punished; accuracy would hide it.

**Why split by household?** Tickets from one family share names, phrasing and habits. A random
split puts the same family on both sides and measures memorisation. Splitting by household
measures "a family the model hasn't seen", which is the real situation.

**What's temperature scaling for?** The UI only pre-fills confident predictions, so the
confidence must mean what it says. One parameter, fitted on out-of-fold predictions, softens or
sharpens the probabilities without changing which class wins.

**What's a paired bootstrap, and why paired?** Resample the same holdout tickets many times and
compute *new minus old* on each sample. Both models see the same tickets, so the luck of which
tickets are in the test set cancels out; that's far more sensitive than comparing two separate
confidence intervals.

**What's PSI and when don't you trust it?** Population Stability Index: how far a distribution
(category mix, confidence) moved from a reference. With ~60 tickets it's noisy, so an alarm also
needs a chi-square test with p < 0.01, and windows under 30 tickets aren't shown.

**Data drift vs concept drift?** Data drift: the inputs change (new kinds of tasks, a child
typing). Concept drift: the same input now has a different right answer (groceries take longer
after a move). Only labels reveal concept drift.

**How do you avoid training on the test set?** Holdouts are frozen manifests in git, excluded
by ID *and* by near-duplicate text, checked after role-token masking as well; a hook blocks
edits to them. Model selection (like choosing C) uses cross-validation on training families
only.

**What's role-token masking?** Names are replaced by "naamkind" / "naamouder" in training *and*
serving, with the same function, so the model learns "a child's name → probably kids" instead
of "Lieke → kids", which only works in one family.

**What would you do with more real data?** Freeze a real holdout from the most recent weeks
(temporal split), weight real tickets more than synthetic ones, plot learning curves to see when
more data stops helping, and switch on the per-household adjustment once it's tuned.

**What did Claude Code do?** It was the pair programmer and teacher; I made the decisions and
can explain each one. The repo has its own reviewer agent that checks ML changes for leakage.
Among its finds: a calibration bug from a wrong class order, training/serving skew in the name
masking, an experimental model that could have been promoted and broken production, and a
conclusion in my own lesson that was overstated. I fixed each one, with a test where possible.
