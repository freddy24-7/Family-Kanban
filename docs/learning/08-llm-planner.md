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

(Filled in after the experiment.)

## Self-check

1. Why is "hours finished" alone a bad metric for a planner, and what does the completion-rate
   guard add?
2. Why must both planners get their context from the same function?
3. What do common random numbers remove, and what can they never remove?
4. The LLM wins in the simulator. What would you need before believing it helps a real family?
5. Why is a planner that leaves tasks *out* of the sprint sometimes the better planner?
