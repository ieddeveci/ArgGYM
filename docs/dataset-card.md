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
| Levels | 3, 6, 9, 12, 15 |
| Strength orderings | last-link and weakest-link, each elitist and democratic |
| Seeds per cell | 2 |
| Items | 480 |
| Engine | `python-argumentation==2.0.2`, pinned |
| Semantics | grounded, except `semantics_query`, which asks about five |
| License | MIT |

The grid is `tasksets/standard.yaml`. A release is identified by its
`taskset_hash`, and the manifest records the arggym version, the engine version,
the prompt and scoring versions, which seeds produced the items and which were
skipped.

## How to read a score

**Per task, not in aggregate.** The twelve metrics are of four kinds: engine
verification, F1 over label pairs, an ordered-sequence constraint, and a
weighted blend for `formalization`. An unweighted mean over them moves mostly
with which tasks are in the basket, which is a property of the basket rather
than of the model.

**Against the chance floor.** A constant answer is worth measuring on every
task, and on four of them it is worth a lot. Measured at level 3:

| task | floor | the constant that earns it |
|---|---|---|
| `semantics_query` | 0.490 | answer "justified" to every query |
| `status_query` | 0.375 | the same |
| `claim_chain` | 0.190 | hand the theory back |
| `perturbation` | 0.178 | one status for every literal in the theory |
| the other eight | 0.000 | engine-checked, so no constant reaches a goal |

A model scoring 0.45 on `semantics_query` did worse than a fixed reply. Report
floors beside the scores. They are a property of the scorer, so `arggym floors
<taskset>` re-measures them from the rows and they move whenever
`scoring_version` does. If a single headline number is wanted, chance-correct
per task first: `(score - floor) / (1 - floor)`, which is `arggym.corrected`.

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

**Preference direction is arbitrary, deliberately.** ASPIC+ takes the ordering
as a parameter and derives nothing; Modgil and Prakken define the set orderings
"assuming a preordering over the elements". ArgGYM's own choice is a per-item
coin flip. If preference direction always matched plausibility, a model could
answer from priors and the verified oracle would be decorative.

**Elitist and democratic diverge only where a preference decides between sets
with more than one element**, so on eight of the twelve tasks the two halves of
that axis produce the same answers (#72). Both are generated because measuring
where they diverge is only possible if both exist, not because every task uses
the distinction.

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
