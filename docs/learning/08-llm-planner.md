# 08 — An LLM as decision-maker: baselines, simulation and honest evaluation

Phase 9 (stretch). Code: `backend/app/services/planner_assistant.py` (context, rules, Gemini,
validation), `backend/sim/world.py` (load work model), `backend/sim/runner.py` (planning in the
simulator), `backend/sim/compare_planners.py` (the comparison).

## 1. From predicting to deciding

Lessons 02–07 were about **predictions**: a category, an effort, a similar task. A wrong
prediction is visible and cheap: the planner corrects it. The Planner Assistant makes a
**decision**: which tasks this week, and who does them. Decisions have consequences that only
show up later (did it get done? did the 9-year-old get the tax return?), and there is no
"correct answer" to compare with per task. That changes how you evaluate.

## 2. "It sounds sensible" is not a metric

LLM output is fluent. A proposal with neat Dutch reasons *looks* good, and that's the trap:
fluency is not quality. To know whether the assistant helps you need:

1. **A baseline**: what would a simple, transparent rule do? Here: most urgent / oldest first,
   each task to the member with the most estimated capacity left who doesn't usually fail
   that kind of task. If Gemini doesn't beat this, its cost and opacity aren't worth it.
2. **The same information for both**: one function (`build_context`) builds the context, and
   both planners get exactly that. Giving the LLM more (or less) would compare information,
   not planners.
3. **An outcome metric, declared before the experiment**: true hours of *finished* work per
   week, guarded by the completion rate (section 5). Not "number of planned tasks" (easy to
   inflate) and not "tasks finished" (rewards only planning small ones).

## 3. Evaluating decisions needs a world model

You can't replay a real week with a different plan: the family only lived it once. The
simulator can. But the old simulator ignored load: completion depended only on the task's
size and whether a child did it, so *every* planner that filled the sprint would score the
same. A world that can't tell good from bad decisions can't evaluate a decision-maker.

The **load work model** (opt-in, old runs unchanged) adds what makes planning matter:

- every person has weekly hours for household tasks (adults 5, a 12-year-old 2.5, an
  8-year-old 1, a 5-year-old 0);
- a person works through their tasks in board order; a task that doesn't fit in the hours
  left gets done far less often (×0.3);
- a task unsuited to the person (a child doing finances) gets done less often (×0.4).

These rules are **hidden** from both planners. They only see what the app sees: who finished
what in recent weeks. A planner has to *learn* that the 5-year-old never finishes anything.

**Caveat, by construction**: the world model encodes my assumptions about families. A planner
that matches those assumptions wins. The result says "better in this world", which is only
evidence for the real one if the world is roughly right.

## 4. Common random numbers

Two runs that differ only by the planner still differ by luck. **Common random numbers**
remove that: under the load model every random event (arrival time, whether a task gets done,
whether the planner checks a label, review noise) gets its own random stream, keyed by run
seed, ticket, week and purpose. Whoever planned, the same ticket in the same week has the same
luck, so the difference that remains is the planning. A test runs two *different* planners on
one world and checks that arrival times and labels are identical.

Two bugs on the way, both caught before any result was read:

- **Only the completion roll was keyed at first.** Everything else drew from one shared stream,
  and how many draws it took depended on what got done. From week 1 the two runs drifted
  apart by labelling and arrival luck (found by the ml-reviewer).
- **Demo-family members join in one transaction**, so their order was arbitrary and replays of
  the *same* planner differed (found by the determinism test).

## 5. The metric that rewards the wrong thing

The first pre-declared metric was "finished hours per week". The review pointed out the flaw:
an overloaded task still sometimes gets done, and a failed task costs nothing, so planning
*more* can never lower finished hours. The metric would punish exactly the "don't overload
anyone" behaviour it should reward. The decision rule was changed **before any result was
seen** (changing it afterwards would be fishing for a win):

- a planner wins only if it finishes **more hours** (paired weekly difference, 95% interval
  above zero) **and** its **completion rate** isn't clearly lower. That's the promotion gate's
  logic: better on the goal, not clearly worse on the guard.
- next to it: the age of the **oldest open task**, because "average wait of finished tasks"
  improves when a planner simply never does the old ones (survivorship).

Weeks aren't independent (this week's plan shapes next week's backlog), so the interval uses
a **moving-block bootstrap**: blocks of 3 consecutive weeks are resampled together. And the
LLM isn't deterministic, so one Gemini run is one sample; it's repeated to see its own spread.
Weeks where Gemini failed and the rules stood in are excluded from the comparison.

## 6. Guardrails for an LLM that acts

- **Validation**: Gemini may invent refs, repeat tasks or ignore the limit; only valid,
  unique items within `max_items` survive (tested with invented refs).
- **Human in the loop**: a proposal changes nothing; the planner ticks what to accept.
- **Fail soft**: no key or an error → the rule proposal, with a note.
- **Privacy by design**: Gemini sees "lid 1 (child)" and texts with members' names (full and
  first name) replaced by role tokens; the reasons are mapped back to names on the server
  (tested: no member name in the prompt). Names of non-members ("oma Riet") aren't recognised.

## What happened in this project

Five 26-week replays under the load model, seed 202 (commit `913094d`, after the review fixes):
the rule planner and Gemini (`gemini-3.5-flash`, ~80k tokens per run) on the drift-demo world
and on the baseline world, plus a second Gemini run on drift-demo to see the LLM's own spread.
The simulated family has ~13.5 hours a week for household tasks.

| Per 26 weeks | Rules (drift) | Gemini (drift) | Gemini again | Rules (baseline) | Gemini (baseline) |
|---|---|---|---|---|---|
| Planned hours | 161.5 | 401.5 | 427.0 | 168.0 | 372.0 |
| **Finished hours** | 118.0 | **228.5** | **221.0** | 124.5 | **181.5** |
| **Completion rate** | **0.73** | 0.60 | 0.56 | **0.75** | 0.53 |
| Overloaded items | 15 | 107 | 134 | 14 | 107 |
| Oldest open task at the end (days) | 150 | 129 | 99 | 148 | 129 |
| Backlog at the end | 250 | 168 | 177 | 271 | 231 |

Paired per week (Gemini − rules, moving-block 95% CI):

| World | Finished hours / week | Completion rate |
|---|---|---|
| drift-demo | +4.4 (+3.0..+6.3) | −0.12 (−0.21..−0.03) |
| drift-demo, 2nd Gemini run | +4.1 (+2.9..+5.8) | −0.16 (−0.24..−0.08) |
| baseline | +2.3 (+1.2..+3.3) | −0.23 (−0.31..−0.15) |
| Gemini vs Gemini | −0.3 (−0.8..+0.2) | −0.05 (−0.10..−0.00) |

**Decision rule: no clear winner, in every world.** Gemini finishes clearly more work, but by
planning 2.5× as much and overloading people, so a clearly lower share of what it plans gets
done. That's exactly the trade-off the completion-rate guard was added to expose: with
"finished hours" alone, Gemini would have "won".

**What each planner gets wrong**

- **The rules are stuck in a feedback loop.** They estimate a member's capacity from the hours
  that member *finished*, which can never exceed what the rules *gave* them. So they plan 5–8
  hours a week for a family that has 13.5, and the estimate never grows. This is a
  small-scale version of a classic problem: a system that only learns from the outcomes of its
  own decisions never sees what the alternatives would have produced (no exploration).
- **Gemini ignores its first instruction.** The prompt puts "don't overload anyone" first; it
  plans ~15.5 hours a week, more than the family has, and a third of its items are overloaded.
  Fluent, plausible reasons per task, and still a systematically too-full sprint.
- **The LLM varies with itself**, but less than it differs from the rules: two Gemini runs on
  the same world differ by −0.3 hours a week (not significant) and 0.05 in completion rate.

**Decisions**

- No default switch on the evidence: neither planner is better by the rule declared up front.
  The proposal stays a proposal the planner trims (human in the loop), which suits Gemini's
  failure mode: an over-full sprint is easy to cut, an under-full one is invisible.
- Improving either planner is a new experiment, not a tweak: for example giving both planners
  the same capacity estimate with some exploration, or telling Gemini the estimated hours per
  member. It must then be tested on a world/seed it wasn't tuned on, or it's overfitting to
  seed 202.

## Self-check

1. Why is "hours finished" alone a bad metric for a planner, and what does the completion-rate
   guard add?
2. Why must both planners get their context from the same function?
3. What do common random numbers remove, and what can they never remove?
4. The LLM wins in the simulator. What would you need before believing it helps a real family?
5. Why is a planner that leaves tasks *out* of the sprint sometimes the better planner?
