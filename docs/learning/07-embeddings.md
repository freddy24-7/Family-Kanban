# 07 — Embeddings, similarity search and fair model comparison

Phase 8. Code: `backend/ml/retrieval.py` (shared retrieval functions),
`backend/app/services/similar.py` (planning suggestions), `backend/ml/retrieval_eval.py`
(precision@k, kNN effort), `backend/ml/embeddings.py` (the only module that touches
fastembed; a dev dependency) and `ml.train --features embedding --C 10` (the classifier
experiment).

**Outcome in one line**: the plan was embeddings + pgvector; TF-IDF won every measurement,
so production serves TF-IDF and the embedding code stays as a documented experiment.

## 1. What an embedding is

TF-IDF turns a text into a long, sparse vector with one column per word (or character
n-gram): "gras maaien" has weight on the columns `gras` and `maaien` and zero everywhere
else. Two texts are only similar if they **share words**.

An **embedding** is a short, dense vector (here 384 numbers) produced by a neural network
that was trained on millions of sentence pairs to put texts with the **same meaning** close
together, whatever words they use. We use `paraphrase-multilingual-MiniLM-L12-v2`: small
(220 MB), multilingual (Dutch included), trained specifically on paraphrases, which is
exactly our question: "is this the same kind of task?". It runs on CPU via ONNX (`fastembed`),
so no PyTorch and no GPU.

The model is **pretrained and frozen**: we don't train it. Our own data is only used to
train small models *on top* (logistic regression) or to search in (retrieval). This is
called **transfer learning**: knowledge about language learned elsewhere, reused here.

## 2. Cosine similarity, measured on our own kind of text

Similarity between two vectors = the cosine of the angle between them: 1 = same direction,
0 = unrelated. For L2-normalised vectors (length 1) it's simply the dot product; we
normalise on the way in, so `cos(a, b) = a · b` and pgvector's cosine distance is `1 − a · b`.

Measured with this project's model (`notebooks`-style check, Phase 8):

| Pair | TF-IDF | Embedding |
|---|---|---|
| gras maaien · het gazon bijwerken | **0.00** | **0.69** |
| hond uitlaten in het park · wandeling met de hond | 0.16 | 0.69 |
| gras maaien · belastingaangifte invullen | 0.00 | 0.10 |
| melk halen bij de supermarkt · boodschappen doen bij de Albert Heijn | 0.20 | 0.28 |
| melk halen bij de supermarkt · melk opwarmen voor de baby | 0.28 | **0.63** |

Two lessons in one table:

- **Synonyms**: no shared word, yet the embedding knows mowing the grass = tending the lawn.
  TF-IDF can't, by construction.
- **Embeddings are not magic**: "melk opwarmen voor de baby" (a care task) scores *closer* to
  "melk halen" (groceries) than "boodschappen doen" does. A shared, prominent word still pulls.
  Whether embeddings are *better for our task* is an empirical question → section 5.

## 3. k-nearest neighbours (kNN)

Retrieval = "give me the k stored vectors closest to this one". Used two ways:

- **as a feature for people** (planning suggestions): "similar earlier tasks: Sem did
  'gazon maaien' 3 weeks ago, it took M";
- **as a model**: predict a ticket's effort as the (similarity-weighted) vote of its k most
  similar *reviewed* tickets. kNN has no training step; its "model" is the data. Add a review →
  the next prediction already uses it. That makes it **household-specific and recent by
  construction**, exactly what the global model lacked in lesson 06 (groceries after the move).

## 4. Exact vs approximate search (HNSW)

Exact kNN compares the query with every stored vector: O(n). For a household (hundreds of
tickets) that's microseconds, and it's always correct. At millions of vectors you need an
**approximate** index. pgvector's **HNSW** (Hierarchical Navigable Small World) builds a
layered graph: coarse long-distance links on top, fine local links at the bottom; a search
greedily walks down the layers. Very fast, *usually* finds the true nearest neighbours
(recall < 100%, tunable with `hnsw.ef_search`).

A trap that matters for multi-tenancy: an HNSW index over *all* households finds the global
nearest neighbours first and filters by `household_id` afterwards, so a small household can
get fewer than k results (pgvector 0.8 adds *iterative scans* to fix this). At household scale
exact search is the right choice anyway. (A first version of this phase stored embeddings in a
pgvector table with an HNSW index; it was dropped when TF-IDF won, see below.)

## 5. Evaluating retrieval is not evaluating classification

A classifier is scored per ticket (right or wrong label). A retriever returns a *ranked list*,
so we score lists:

- **precision@k**: of the k returned neighbours, what fraction is relevant? Here "relevant" =
  same *true* category (the simulator's hidden truth; the confirmed label for real tickets), a
  proxy for "the same kind of task".
- **Temporal honesty**: a ticket may only retrieve tickets that existed *before* it (otherwise
  week-5 tickets "find" week-20 answers: leakage through time).
- **Within the household**, as in production.

## 6. Fair comparison with small n

"Embeddings + LR vs TF-IDF + LR" is only fair if:

1. **Same data**: same snapshot, same holdout exclusions, same role-token masking.
2. **Same tuning budget**: TF-IDF's settings were chosen by group CV on training families
   (notebook 02). The embedding model gets the same: C chosen by group CV, **never** by
   looking at the holdouts (choosing the best of five on the test set = overfitting the test).
3. **Same test, paired**: both models predict the same frozen holdout tickets, and the
   difference is bootstrapped *per ticket pair* (lesson 03). Paired tests are much more
   sensitive than comparing two separate confidence intervals.
4. **Same gate**: promote only if clearly better on the primary holdout and not clearly worse
   elsewhere. "Embeddings are modern" is not a reason.

## What happened in this project

The plan said "embeddings + pgvector". The measurements said something else, and the
measurements win.

**Data.** Three simulated runs with ~350 reviewed tickets each, but only **two independent
worlds**: the two drift-demo runs are replays of the same ticket pool (same tickets, same model
predictions), so their agreement is not replication. Every number is time-honest (a ticket only
sees tasks reviewed before it was created) and within one household. TF-IDF is fitted per query
on the query plus those earlier tasks, exactly as in serving.

**1. Retrieval quality** (`ml.retrieval_eval --simulated`). precision@3, relevant = same true
category, no threshold (every retriever returns 3, so the counts are equal):

| World | TF-IDF (char n-grams) | Embeddings | Hybrid (mean of both) |
|---|---|---|---|
| drift-demo | 0.671 | 0.609 | 0.673 |
| drift-demo (replay) | 0.674 | 0.612 | 0.678 |
| baseline | 0.642 | 0.593 | 0.659 |

With a similarity threshold, compare at the **same coverage** (coverage = share of reviewed
tickets with any earlier review that get a suggestion; thresholds aren't comparable between
retrievers, each has its own scale). At ~43% coverage the shown neighbours are the same kind of
task 0.93–0.97 (TF-IDF), 0.93–0.94 (hybrid), ~0.82 (embeddings); at ~64–70%: 0.84–0.89
(TF-IDF), ~0.83 (hybrid), ~0.75 (embeddings). Hybrid adds nothing over TF-IDF here.

**2. Embeddings + LR as a classifier.** C chosen by group CV on training families only, *no
holdout scored* (`ml.train --cv-only --features embedding --C 1|10|100`, reproducible):

| CV macro-F1 (95% CI) | TF-IDF (production config) | Embeddings, best C |
|---|---|---|
| category | **0.936** (0.922–0.948) | 0.867 (C=100) |
| effort | **0.631** (0.592–0.668) | 0.527 (C=10) |

The intervals don't overlap. The gate needs *clearly better*; this is clearly worse, so it never
reached the holdouts.

**3. kNN effort** (similarity-weighted vote of the similar tasks' reviewed effort) vs the effort
model's logged prediction, paired on the same tickets:

- **TF-IDF at the served threshold (0.3)**, ~64% coverage: **a tie** (−0.005 to −0.04, every
  95% CI spans zero).
- TF-IDF at full coverage: slightly worse (−0.04 to −0.09).
- Embeddings at full coverage: clearly worse (−0.11 to −0.16, CIs below zero).
- **Groceries after the move** (drift-demo, from week 12): kNN **0.42** vs the model **0.00**,
  n = 19, gain 95% CI +0.21..+0.68. Household history knows that *this* family's groceries
  became bigger; the global model can't (lesson 06). Real, but from one independent world and
  19 tickets.

Reviews agree with the hidden truth 87–90% of the time, so label noise isn't what limits kNN
overall: similar text often doesn't mean similar effort. Where it shines is exactly where the
model is blind: household-specific change.

**Why the general-purpose embedding loses here**

- It was trained for **paraphrase invariance**: "melk halen" ≈ "grote weekboodschappen doen".
  For effort, that difference *is* the signal. The model was trained to throw it away.
- Families repeat the same chores in the same words ("vuilnis buiten zetten" every week);
  exact lexical overlap is a strong, cheap signal, and char n-grams also catch Dutch compounds.
- 384 dense numbers vs ~50k sparse features: a linear model on top has far less to work with.
- The text is synthetic (one LLM's phrasing per world), which favours lexical matching. Real
  families may paraphrase more; we can't measure that yet (6 real labels).

**Decisions**

- No embedding classifier; `ml.train` refuses to register embedding-feature models at all
  (fastembed isn't installed in production: such a model would load and then fail on its
  first prediction). Prediction failures are now fail-soft, like load failures.
- Planning suggestions use **TF-IDF retrieval** (`retrieval.tfidf_similarities`, threshold
  0.3: ~84–89% of shown neighbours are the same kind of task, ~64% of tickets get one),
  computed per request: milliseconds, always reflects edits and deletions.
- **No fastembed in production** (~400 MB memory + a 220 MB model for a measured loss). It's a
  dev dependency for the experiments. Revisit when real families' text exists.
- The kNN vote is not shown as a number; the Plan page shows the similar tasks as *history*
  (who did it, how big it really was, finished or not) next to the model's guess. That is the
  form in which the groceries finding helps: the planner sees "last time: Middel" when the
  model says "Klein?" (seen in a test family for "Gras maaien achtertuin"). Showing a kNN
  *prediction* would need evidence from more than one drifted world.

## Self-check (results)

6. Why must retrievers be compared at equal coverage rather than at the same threshold?
7. The embedding model is "better at meaning". Why is that a disadvantage for effort?

## Self-check

1. Why can TF-IDF never see that "gras maaien" and "gazon bijwerken" are related?
2. The embedding model was never trained on our data. Where does its knowledge come from, and
   what does that mean for Dutch household jargon it never saw?
3. Why must the kNN effort predictor only look at tickets reviewed *before* the query?
4. Why would tuning C on the holdout make the comparison unfair, even if both models get it?
5. A household has 40 tickets and the HNSW index covers 100,000. What can go wrong with
   "nearest 5 in this household"?
