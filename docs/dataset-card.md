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
task, and on four of them it is worth a lot. Measured on the level-3 rows of
`tasksets/lite.yaml`:

| task | floor | the answer that earns it |
|---|---|---|
| `semantics_query` | 0.727 | "undecided" under grounded, "justified" under credulous preferred |
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
and they move when `scoring_version` moves or when `FLOORS_VERSION` does, since
a change to the search moves a floor with no scorer change at all. A scoring
artifact records both. If a single headline number is wanted, chance-correct per
task first: `(score - floor) / (1 - floor)`, which is `arggym.corrected`.

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
single constant pooled away: one status per semantics is worth 0.657 over the
shipped grid against 0.461 for the best single constant (#95).

**`success` and `mean_score` are not the same question.** For the construction
tasks `success` means every goal met with the theory consistent; economy is a
separate multiplier on top. A model can succeed at every item and still score
0.6 by being wasteful. For `formalization`, `success` means the theory gives
every literal the reference names the status the reference gives it, not only
the literals the question asks about; the score adds the directive shape and the
axiom and strict decisions, graded for both promotion and demotion.

**Format failures are reported separately from reasoning failures.** An answer
that could not be read and an answer that was wrong are different events, and
collapsing them reports measurement error as a result. The parsers ignore
rendering a chat model adds on its own: whitespace, bullets, numbering, CRLF
line endings, lines that are only a code fence, and backticks. So a correct
answer in a markdown code block scores what the bare answer scores. A word
outside the answer's lines, such as a leading `Answer:`, still zeroes the whole
answer, backticked or not.

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

**Past twice the minimum, a construction answer scores zero.** Partial credit
for economy covers only `(minimum, 2*minimum]`, and on small minima that band
is narrow. On the standard grid 137 of the 600 `preference_construction` rows
have a minimum of 2 or less (117 of 2, 20 of 1), as do a third of `defence`
rows (200 of 600) and 40% of `counter_argument_strict` rows (240 of 600), so a
long answer on those rows is zeroed rather than docked. Reports carry
`bloat_rate`, the share of scored answers the gate zeroed, so a reader can see
how much of a model's score it decided (`docs/evaluation.md`).

**`counter_argument_strict` is an ablation of `counter_argument`, not a twelfth
independent measurement.** The two arms publish the same theory -- byte for byte
on all 120 of their cells, every level and ordering at two seeds -- and vary one
thing, whether the answer may add a strict rule. Neither arm needs a preference.
An undercut defeats its target whatever the ordering says, so the plain arm's
cheapest answer is a walk read off the theory: rebut the target from an
uncontested premise, then undercut each of the `n` chains reaching it, at the
rule that reaches the target or, where that rule is strict, at the chain's first
rule. It costs `1 + n` directives, or `n` from level 9, where the theory's decoy
already rebuts the target. Under last-link, outranking a defeasible chain's
final rule costs the same as undercutting it, and the gold keeps the preference
answer; under weakest-link the gold is the walk. So a correct weakest-link
answer that ranks each chain with preferences, the way the gold did before #184,
is long: it runs past twice the minimum and scores zero for bloat on 18 of the
30 `weakest_link_democratic` cells, and scores 0.75 to 0.93 on the rest and on
every `weakest_link_elitist` cell. A drop from last-link to weakest-link on this
task is that cost, not an effect of the ordering on the answer. A strict rule
cannot be attacked in ASPIC+, so a strict rebuttal defeats every defeasible
chain for free, and only the `k` chains that reach the target strictly still
need an undercut: `1 + k`, with `k` at least 1 and below `n`. On all 120 cells
that strict walk scores 1.0 in exactly the gold's number of lines (#139). The
gap between the arms therefore measures whether a model uses the unattackability
of strict rules; preference reasoning is `preference_construction`'s task. The
plain arm's own reference, submitted to the strict item, earns full credit on
none of the cells: on 120 of them it scores between 0.75 and 0.92, docked on
economy for directives the cheaper minimum no longer needs, and on 0 of them it
runs past twice the strict minimum. Read the strict arm as the contrast against
`counter_argument`, and not as a reasoning score of its own.

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
changes that. Under weakest-link the axis reaches an answer on one task: a
`preference_construction` reference re-scored under its partner ordering still
scores 1.0 on only 6 of 30 items, because the answer's own preference directives
are what put several defeasible rules into an argument (#72). The axis does not
reach `counter_argument`: its minimum is the same under all four orderings, and
its weakest-link reference, a rebuttal plus one undercut per chain, scores 1.0
under the partner ordering on 30 of 30.

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

Nor does a derivation's shape give the answer away. In `claim_chain`, every
derivation of the claim is built the same way. Below level 4 the justifying line
is unattacked and each decoy falls to a single attack. From level 4 each
derivation starts at a negated premise preferred over its positive twin, has the
same junctions with the same negated branches, the same dead-end rules hanging
off it from level 11, and is attacked by one tower: a rule rebutting one of its
steps, and above it rules each undercutting the one below. That tower alone
decides the derivation. The line's has even height and a decoy's odd height of at
least 3, so the first rule of every tower is itself undercut, and only following
each tower to its top tells the line from the decoys. Heights grow with the
level: 2 against 3 at levels 4 to 7, then the line's 2 or 4 against decoys' 3 or
5, and from level 12 the line's 4 or 6 against 5 or 7. The target step is drawn
the same way for every tower, and from level 8 the line's tower is the shortest
or the longest no more often than any other. From level 4, picking a derivation
by the sign of its premises, a preference on its root, its size, which of its
literals other rules build on, where it is listed, how many directives attack it,
or whether each of its attackers is itself attacked scores what a random
derivation scores (`tests/test_claim_chain_gold_has_no_surface_cue.py`). Counting
a tower's height does better by design, since that is the walk. So does a partial
count at levels 4 to 11, where a height of 2 or 3 settles a tower without reading
past it: at levels 8 to 11, 5 of every 6 decoy towers are 3 high, so "a tower of
height 3 means a decoy" scores 0.83 where a random derivation scores 0.33.

There is no personal data, no scraped text, and no human annotation. The natural
language in `formalization` comes from surface forms taken from the ASPIC+ and
argumentation-schemes literature (`arggym/core/nlforms.py`).

## Maintenance

Issues and releases: https://github.com/ieddeveci/ArgGYM

A release that changes any prompt, any gold, or any scoring policy gets a new
`taskset_hash` and a new tag. Results across different hashes are not
comparable, and the manifest is what tells you which you have.
