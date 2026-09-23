# ArgGYM dataset card

What the benchmark measures, how items are made, and what a score does and does
not license. Written to the shape of Gebru et al.'s datasheets, which ask the
questions a reader needs before citing a number.

The interface is specified separately in `docs/dataset-contract.md`; the DSL and
the semantics conventions are in `NOTATION.md`.

---

## What it measures

Defeasible reasoning in ASPIC+ under grounded semantics: whether a model can
work out what a structured argumentation theory entails, and construct
directives that change what it entails.

Twelve tasks fall into two families.

**Six ask what a theory says.** `status_query`, `semantics_query`,
`claim_chain`, `defeat_diagnosis`, `perturbation`, `formalization`. The answer
is a reading of the theory: the status of a claim, the line that justifies it,
what changed when the theory did.

**Six ask the model to change what a theory says.** `preference_construction`,
`counter_argument`, `counter_argument_strict`, `attack`, `defence`,
`attack_defense`. The answer is a set of directives that must reach a stated
goal, keep the theory consistent, and do it in close to the fewest moves.

The second family is why the benchmark exists. Its answers have no oracle: any
directive set reaching the goals is correct, so grading means running the engine
on theory plus answer. A model cannot pattern-match a reference it was never
shown.

## How an item is made

Backwards, and then verified.

1. A theory is constructed to a level's shape. Each task states its own shape,
   in its own module under `arggym/tasks/`; what the tasks share is a set of
   level-indexed knobs -- junction density, ternary junctions, the language
   profiles -- in `arggym/core/curriculum.py`. Those numbers come from
   generation sweeps rather than from choice.
2. A reference answer is constructed alongside it, and for the construction
   tasks a minimum is searched for rather than assumed.
3. The item is evaluated through PyArg. If the reference does not score 1.0, or
   an invariant fails, the draw is rejected and the seed advances.

So the gold is never asserted. It is what the engine returns.

This is also the benchmark's defence against contamination, and a real one: the
frozen taskset is for citable comparison, and any contested result can be re-run
on fresh seeds from the same spec. A fixed dataset cannot offer that.

## What is in a release

| | |
|---|---|
| Tasks | 12 |
| Levels | 1 through 15 |
| Strength orderings | last-link and weakest-link, each elitist and democratic |
| Seeds per cell | 10 (lite: 2) |
| Items | 7200 (lite: 1440) |
| Engine | `python-argumentation==2.0.2`, pinned |
| Semantics | grounded, except `semantics_query`, which asks about five |
| License | MIT |

The grid is `tasksets/standard.yaml`. Ten seeds give each (task, level) 40
items, which is what a per-task level curve needs; at two, a (task, level) mean
over 8 items moves in steps of 0.125. `tasksets/lite.yaml` is the same grid at
two seeds, for a model ranking at a fifth of the cost. Its rows are the first two of every standard
cell, so a lite score is a score on a subset of the standard release rather than
on another draw.

A release is identified by its `taskset_hash`, and the manifest records the
arggym version, the engine version, the prompt and scoring versions, which seeds
produced the items and which were skipped.

## How to read a score

**Per task, not in aggregate.** The twelve metrics are of four kinds: engine
verification, F1 over label pairs, an ordered-sequence constraint, and a
weighted blend for `formalization`. An unweighted mean over them moves mostly
with which tasks are in the basket, which is a property of the basket rather
than of the model.

**Against the chance floor.** An uninformed answer is worth measuring on every
task, and on four of them it is worth a lot. Measured at level 3:

| task | floor | the answer that earns it |
|---|---|---|
| `semantics_query` | 0.806 | "undecided" under grounded, "justified" under credulous preferred |
| `status_query` | 0.375 | answer "justified" to every query |
| `formalization` | 0.219 | `[premise: x]` for every literal the question queries |
| `claim_chain` | 0.190 | hand the theory back |
| `perturbation` | 0.153 | "overruled" for every claim of the original theory |
| `defeat_diagnosis` | 0.131 | the line `status: overruled`, and no failure points |
| the six engine-checked tasks | 0.000 | no fixed answer reaches a goal |

The floor is the best of what the search tried, so it is a lower bound on what
an uninformed answer gets, and every one is printed with the strategy that
reached it. That is what makes a zero readable: `formalization` reported
`0.000 empty` while an answer shaped like a theory was worth 0.22, and only the
strategy name said the search had found nothing fitting the answer format rather
than nothing that pays (#103). On the six above, everything the search tries
scores zero, and the tie goes to `empty` because it is listed first.

A model scoring 0.45 on `semantics_query` did worse than a fixed reply. Report
floors beside the scores. They are a property of the scorer and of the search
that measures them, so `arggym floors <taskset>` re-measures them from the rows,
and they move when `scoring_version` moves or when `FLOORS_VERSION` does -- the
change that gave `semantics_query` the map above moved its floor from 0.459 to
0.671 with no scorer change at all. A scoring artifact records both. If a single
headline number is wanted, chance-correct per task first:
`(score - floor) / (1 - floor)`, which is `arggym.corrected`.

**A program that computes no extension beats the floor on `semantics_query`.**
That table measures a fixed map, one status per semantics. A short program that
reads the theory as text and computes no extension does better, because the
shapes the generator uses are recognisable. On the stable column, 81.4% of
answers (503 of 618, over 400 items) fall to such a program: 51.8% to a rule
that finds the literals carrying an odd ring of undercuts and reads which side
of the contested pair they stand on, and the rest to a constant. The same
program gets 53.2% before the odd-cycle cluster existed, when stable duplicated
sceptical preferred and any rule that worked on one worked on both.

The cluster hangs its ring one rule away from the literal the question asks
about, so the program needs a lookup to connect them. That is a lookup, not an
inference: it succeeds on 24 of 24 items carrying a shielded ring. Odd cycles
are the only structure that separates stable from the preferred semantics at
all, so a benchmark that asks about stable has to build one and cannot then hide
it. Read the stable column as measuring whether a model finds that structure,
not as evidence it computed an extension.

No floor bounds that program, and none should: a floor may read the theory to
enumerate the coordinates a question asks about, never to decide what to say
about one (`docs/dataset-contract.md` section 10), and this program decides from
it. What the floor does now carry is the spread between the columns, which a
single constant pooled away: one status per semantics is worth 0.671
over the shipped grid against 0.459 for the best single constant (#95).

**`success` and `mean_score` are not the same question.** For the construction
tasks `success` means every goal met with the theory consistent; economy is a
separate multiplier on top. A model can succeed at every item and still score
0.6 by being wasteful.

**Format failures are reported separately from reasoning failures.** An answer
that could not be read and an answer that was wrong are different events, and
collapsing them reports measurement error as a result.

## What a score does not license

**`formalization` measures template inversion, not argument modelling.** The
formal theory is sampled first and the natural-language argument is rendered
from it, which is the only way to have engine-verified gold for a
natural-language task. But two lexical markers give away the two hardest
modelling decisions: whether a fact is disputable, and whether an inference is
defeasible. A high score here should not be reported as "models can formalize
natural-language argumentation".

**`min_directives` is minimal among the candidates the generator produced**, not
proven globally minimal. An answer shorter than the reference is possible, and
efficiency is clamped at 1.0, so it earns full credit rather than extra. The
manifest records the caveat with the taskset.

**`counter_argument_strict` is an ablation of `counter_argument`, not a twelfth
independent measurement.** The two arms publish the same theory -- byte for byte
on all 120 of their cells, every level and ordering at two seeds -- and vary one
thing, whether the answer may add a strict rule. Permitting one buys a cheaper
minimum on every cell. The plain arm's own reference, submitted to the strict
item, earns full credit on none of them: on 89 of them it scores between 0.75
and 0.92, docked on economy for directives the cheaper minimum no longer needs,
and on the other 31 it runs past twice the strict minimum and scores zero for
bloat. What the strict arm then asks is not argumentation. A strict rule cannot
be attacked in ASPIC+, so a strict counter-argument defeats every defeasible
chain with no preference, and the answer is a walk read off the question line: a
strict rebuttal of the target, then one undercut for each strict rule concluding
it. On all 120 cells that walk scores 1.0 in exactly the gold's number of lines,
and the gold carries no preference directive (#139). Read the arm as the contrast
against `counter_argument` -- what removing the preference machinery does to a
model's score -- and not as a reasoning score of its own.

**Preference direction is arbitrary, deliberately.** ASPIC+ takes the ordering
as a parameter and derives nothing; Modgil and Prakken define the set orderings
"assuming a preordering over the elements". ArgGYM's own choice is a per-item
coin flip. If preference direction always matched plausibility, a model could
answer from priors and the verified oracle would be decorative.

**Elitist and democratic diverge only where a preference decides between sets
with more than one element**, and the theories rarely hold one. Under last-link
the compared set is an argument's last defeasible rules, and outside
`status_query` that set has one member or no preference ranks its members:
across all twelve tasks at levels 1, 3, 6, 9, 12 and 15, built under either
last-link ordering, the two give the same defeat relation, status map and
extensions everywhere except 8 of the 24 `status_query` theories, and every
construction reference still scores 1.0 under the partner ordering (#113). So on
eleven tasks the two last-link columns are two draws of one question, not
independent evidence about the ordering; on `status_query` a queried label
differs between them on 5 of 60 theories (#72).
`tests/test_the_last_link_orderings_are_one_ordering.py` fails if a generator
changes that. Under weakest-link the axis reaches an answer on two tasks: a
`preference_construction` reference re-scored under its partner ordering still
scores 1.0 on only 6 of 30 items and a `counter_argument` one on 17 of 30,
because the answer's own preference directives are what put several defeasible
rules into an argument (#72).

**A frozen row is re-scorable only against the pinned engine.** Scoring the
construction tasks runs `python-argumentation==2.0.2` at scoring time, not just
at generation time. `score` compares the recorded version and refuses on a
mismatch rather than returning a quietly different number.

## Composition and bias

Items are synthetic. Literals are opaque symbols drawn from a per-item pool, and
rule names are randomized, so nothing carries world knowledge and no answer can
be reached from priors about the subject matter. That is the point, since the
benchmark isolates the reasoning, and it is also the limit: it says nothing
about defeasible reasoning over real arguments in natural language, except
through `formalization`, whose limits are above.

There is no personal data, no scraped text, and no human annotation. The natural
language in `formalization` comes from surface forms taken from the ASPIC+ and
argumentation-schemes literature (`arggym/core/nlforms.py`).

## Maintenance

Issues and releases: https://github.com/ieddeveci/ArgGYM

A release that changes any prompt, any gold, or any scoring policy gets a new
`taskset_hash` and a new tag. Results across different hashes are not
comparable, and the manifest is what tells you which you have.
