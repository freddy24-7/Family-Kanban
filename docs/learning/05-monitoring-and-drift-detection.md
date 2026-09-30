# 05 — Monitoring a model in production: drift detection

Phase 6. Code: `backend/ml/monitoring.py` (detectors), `backend/app/services/monitoring.py`
(data), the *Modelgezondheid* page in the admin area.

## 1. Two kinds of signals

| | Without labels | With labels |
|---|---|---|
| Examples | predicted category mix, confidence distribution, text length | accuracy vs planner labels, effort vs sprint review |
| Speed | **immediate** (every new ticket) | **delayed** (labels come days to weeks later) |
| Catches | covariate and prior drift | everything, including **concept drift** |
| Weakness | blind to concept drift; drift ≠ harm | slow; few labels = wide intervals |

You need both: label-free signals as the smoke alarm, labelled accuracy as the fire itself.

## 2. Reference vs current window

Drift is always *relative*: "different from what?". The **reference** is how the data looked
when the model was validated: its predictions on the frozen holdout, stored with each model
version. The **current window** is the last few weeks of production data. Every week the
window moves one week forward (a *rolling window*).

Why weeks and not days: a family writes ~15 tickets a week. One week is too few tickets for
any test; a 4-week window (~60 tickets) is the minimum for a rough answer.

## 3. PSI: Population Stability Index

PSI compares two distributions over the same bins (categories, confidence bins, length bins):

    PSI = Σ (current% − reference%) × ln(current% / reference%)

- Each bin contributes when its share moved; big relative moves in small bins weigh heavily.
- Rules of thumb: **< 0.1** stable, **0.1–0.25** moderate shift, **> 0.25** significant shift.
- Empty bins would make `ln` explode, so a small epsilon is added (a known weakness: with few
  tickets, PSI is noisy and biased upward).

## 4. Chi-square test: is the shift more than chance?

PSI says *how much* the mix moved; the chi-square goodness-of-fit test asks *could this be
chance?* It compares observed counts in the window with the counts expected from the reference
proportions and gives a **p-value**. Small p (< 0.01) = unlikely to be noise.

Caveat: chi-square is unreliable when expected counts are below ~5 per bin. With small windows
we **merge rare bins** and require a minimum window size. Otherwise the test "detects" noise.

## 5. Accuracy with an interval

Accuracy on 20 labelled tickets is a rough estimate. We report a **Wilson interval** (a
confidence interval for a proportion that behaves well for small n and for values near 0 or 1)
and only alarm when the whole interval lies below the reference accuracy: "significantly worse",
not "a bad week".

## 6. False alarms vs detection delay

Every detector trades two errors:

- **False alarm**: fires without real change (noise, seasons). Too many and people ignore it.
- **Detection delay**: weeks between the change and the alarm. Too slow and users suffer.

Bigger windows and stricter thresholds → fewer false alarms but longer delays. In the simulator
we *know* when changes happened, so we can measure both honestly: alarms before the first event
are false alarms (or seasons!), and the delay is the first alarm after an event.

## 7. Drift is not the same as harm

A seasonal shift (more garden work in summer) is real drift, but if the model handles garden
tickets well, nothing needs fixing. The question is always: **did performance drop?** That's why
the dashboard shows drift signals *next to* accuracy, not instead of it.

## What happened in this project

The detectors were tested on the 26-week `drift-demo` run **and on a control run** (`baseline`:
seasons only, nothing planted), both with the first 4 weeks as reference.

| Planted change | drift-demo | Also in the control run? | Verdict |
|---|---|---|---|
| Dog (week 9) | category mix alarm, delay 0 | **yes, same week** (September: school starts) | **confounded**: can't be attributed to the dog |
| Concept drift (week 13) | effort accuracy alarm, **delay 2 weeks** | no (effort accuracy stays 0.50–0.68) | **real detection**, delayed by the labels as predicted |
| New typist (week 17) | nothing convincing | – | **missed**: a small change in a subgroup |

Lessons learned the hard way:

- **A control experiment is essential.** Without the baseline run, the dog looked perfectly detected.
  It coincided with the September school start, which the world model also makes a big shift.
- **Reference choice decides what you see.** Against the training holdout, effort accuracy alarms from
  week 2 onward and never stops (the simulated world has far more M/L tasks than the training data):
  true, but an always-on alarm can't detect anything new. A post-deployment reference fixes that.
- **Count onsets, not weeks.** A first version counted "an alarm is on" as a detection, which made
  every event look detected with zero delay.
- **PSI alone is too noisy for ~60 tickets.** Every label-free alarm now also needs a chi-square
  p < 0.01, and PSI isn't shown for windows under 30 tickets.
- Tests caught two detector bugs: a float artefact (Wilson's upper bound 0.9999… < 1.0 raised an
  accuracy alarm at 100%) and an undefined chi-square when a never-seen category appeared.
- At household scale, label-free drift mostly measures **seasons**. The reliable harm signal is
  labelled accuracy, which is why the sprint review matters.

## Self-check

1. Which signal would catch "groceries now take longer", and how late?
2. Why add an epsilon in PSI, and what does that do with very small windows?
3. A chi-square test says p = 0.001 on a window of 12 tickets. Should you trust it?
4. PSI on the category mix jumps in December every year. Drift, harm, or neither?
