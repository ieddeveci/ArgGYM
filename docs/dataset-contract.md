# The dataset contract

What ArgGYM promises to anyone who evaluates a model against it, and what it
deliberately refuses to promise.

This document is the design. It says what each decision is, why it is that way,
and what was rejected. The issues it closes are #10, #49, #50, #51, #53 and #54.

---

## 1. The one rule

**ArgGYM owns what a legal answer is. The evaluator owns how the answer is
delivered.**

Every decision below follows from that sentence, so it is worth being precise
about which side a thing falls on. The test is not "is this formatting" but:

> Does the constraint come from defeasible argumentation, or from our
> implementation?

Constraints from the subject are the benchmark. Constraints from our
implementation are our accidents, and charging a user for them measures the
wrong thing.

| Constraint | Comes from | Verdict |
|---|---|---|
| The answer is a set of ASPIC+ directives | the subject | keep |
| An answer using more than twice the minimum scores zero | the subject: economy is a measured capability | keep |
| Preference is a preorder, so declaring both directions settles nothing | the subject (Modgil & Prakken) | keep |
| Grounded semantics, four strength orderings | the subject | keep, and do not make them swappable |
| The answer sits between `[answer]` and `[/answer]`, always | our implementation | **remove** |
| The answer is DSL *text* | our implementation | **remove** |

The last two are why this document exists.

### The fence is stated, and it is still not the benchmark's

"Remove" above is about the word *always*, and the distinction is easy to lose.
A submission convention has to be **in the prompt**: the parser is strict, so
every line inside the fence must be an answer line, and a model that reasons
before writing its answer needs somewhere to put the reasoning. Take the fence
out of the prompt and a reasoning completion is read as a malformed answer,
which scores zero for a reason that has nothing to do with argumentation.

What must not happen is the fence becoming a fixed property of the dataset. So
it is a render-time parameter. `AnswerTemplate` holds the pair and the sentence
that asks for it, the default is `<answer>`/`</answer>` matching reasoning-gym
(`reasoning_gym/utils.py:25` reads it back with `<{tag}>\s?(.*?)\s?</{tag}>`),
and a harness with another convention re-renders instead of editing prompt
strings. The row records `metadata.answer_template`, so a frozen taskset says
what its questions asked for. `extract_answer` reads the current pair first, the
older `[answer]` pair after it so recorded generations keep scoring, and returns
a completion whole when neither is present.

The content half of the answer-format block never moves: "one directive per
line", "copied exactly as it appears above", "one line per claim, written as
`claim: status`" are the task's, and the template only ever adds the sentence
after them.

### What we refuse to generalize

A benchmark that accepts every possible way of doing everything measures
nothing. Three things stay fixed and opinionated:

- **The semantics.** Grounded, with the four Modgil-Prakken set orderings.
- **The gold.** Engine-verified by construction, never asserted. Nobody plugs in
  their own grader.
- **The notation.** One DSL for directives, one meaning per keyword.

Those are the benchmark's claims. Making them configurable would leave nothing
to cite.

---

## 2. What a row is

```jsonc
{
  "id": "status_query/L9/weakest_link_elitist/s0",
  "task": "status_query",
  "question": "<theory> <ask> <notation>",
  "reference_answer": "<one correct answer, raw>",
  "metadata": {
    "source_dataset": "status_query",     // the registered task name
    "source_index": 0,                    // row position in this file
    "seed": 0,                            // the seed that built it
    "level": 9,
    "ordering": "weakest_link_elitist",

    "arggym_version": "2.1.0",
    "pyarg_version": "2.0.2",
    "prompt_version": 3,
    "theory_schema": 1,
    "scoring_version": 1,

    "checker": "graded",                  // exact | graded | verified
    "answer_shape": "label_map",          // see 4

    "theory_ops": [ ... ],                // the theory shown in the question
    "state": { ... },                     // per-task, non-gold
    "gold": { ... }                       // everything that gives the answer away
  }
}
```

Three fields carry most of the design.

### `reference_answer`, not `answer`

The field is one correct answer, not the key to the answer. For six of the
twelve tasks there is no oracle at all: `preference_construction`,
`counter_argument`, `counter_argument_strict`, `attack`, `defence` and
`attack_defense` are graded by running the engine on theory + answer and
checking the goals, so any directive set that reaches them is correct. Naming
the field `answer` invites string comparison, and string comparison is wrong on
those six. It is also wrong on `formalization`, whose score is
a weighted blend of behavioural, shape and type components
(`arggym/tasks/formalization.py:479-481`) —
a correct formalization written differently matches at 0 and scores high.

Reasoning-gym has the same distinction and handles it by overriding
`score_answer` per dataset (`reasoning_gym/dataset.py:63`, *"Overwrite this
method in derived classes if a single oracle answer is not available"*). We do
the same, and never inherit a default: **the base `score_value` raises.** A
task that forgets to implement it must fail loudly, not fall back to substring
matching.

### `metadata.gold`

Everything that gives the answer away sits under one key, so "do not show the
model `metadata.gold`" is one rule instead of twelve. It is not always obvious
what belongs there. `min_directives` does — the prompt never states how many
directives are needed, so publishing it beside the question leaks the answer
size. So does `perturbation.survivors`, which names the claims that did *not*
change.

### `checker`

Three values, describing how an answer is judged rather than whether the
reference is unique:

- **`exact`** — normalized comparison against the reference is sound.
- **`graded`** — a continuous scorer over a unique gold.
- **`verified`** — the engine is run on theory + answer.

No v2 task is `exact` today. The value exists because a future task might be,
and because a harness needs to know that `graded` and `verified` rows cannot be
scored by comparison.

---

## 3. The question is a view of the state

The question text is a pure function of the item's structured state, of which
the operation list is one field.

This is not tidiness. It closes a bug class that has already shipped twice. From
the pre-freeze audit (#12):

> `validate_entry` never inspects the prompt. It reads `entry["answer"]` and
> `entry["metadata"]` only; gold is derived from `metadata["ops"]` while the
> model reads `entry["question"]`. Nothing checks those describe the same
> theory.

Two shipped bugs had internally coherent ops, correctly computed gold, a
reference that self-scored 1.0, and a prompt the model could not answer. If the
question is rendered from the state, the two cannot disagree.

**One exception, named.** `formalization`'s question is natural-language prose,
and the sentences and the operations are generated together in one interleaved
RNG stream (`arggym/tasks/formalization.py:168-333`). Re-rendering would require
the RNG state at each sentence, which is not a property of the operations. Its
surface text is itself generated state and is stored as a string. That is also
the task where `reference_ops` *is* the gold answer
(`arggym/tasks/formalization.py:350`), not the theory in the question — which is
why the schema names keys by what they hold (`theory_ops`, `added_ops`,
`answer_ops`) rather than one generic `theory`.

`perturbation` has two operation lists, `theory_ops` and `added_ops`, both
rendered into the question.

---

## 4. Answers are values; text is one serialization

```python
parse(text, item) -> Value          # text -> the answer, no scoring
score_value(value, item) -> ScoreResult
score(text, item)  = score_value(parse(text, item), item)
```

The scorer already works this way and hides it. For the seven operation-list
tasks it converts DSL text into `Operation`s and grades the resulting engine
state; the text is thrown away. Requiring text is therefore us making a user
imitate our serialization, and the cost is measured (#10): across the pilot
runs, `no_answer_region` reached 0.135 and `zero_score_with_valid_region` was
**0.180** on qwen3.6-27b — answers that transported fine and died in the parser,
some confirmed correct. That is larger than some level effects, which makes it a
confound rather than a finding.

With the seam, a user with constrained decoding or a JSON schema submits a
value. A user with a plain completion submits text. Both reach the same scorer.

**Four answer shapes, not three.** #10 counted three across the v1 tasks; the v2
twelve have four:

| Shape | Tasks | Value |
|---|---|---|
| operation list | `preference_construction`, both `counter_argument`, `attack`, `defence`, `attack_defense`, `formalization` | `List[Operation]` |
| label map | `status_query`, `semantics_query`, `perturbation` | `Dict[key, status]` |
| ordered sequence | `claim_chain` | `List[str]`, order scored (`arggym/tasks/claim_chain.py:401-404`) |
| record list | `defeat_diagnosis` | `status` + `List[{defeated_at, defeater, kind, survives_because?}]` |

`claim_chain`'s parse resolves each quoted line against the theory, so its
signature is `parse(text, item)` rather than `parse(text)`. Making the item
available to every parser costs nothing and avoids a special case.

### One parser, not seven

The `[answer]` region regex is defined seven times, identically, in
`core/scoring.py` and six task modules. The DSL rule regex is defined twice,
in `core/scoring.py:19` and `tasks/formalization.py:383`.

Until #81 those two disagreed: one made the rule name optional and invented one,
the other required it, so the same answer text scored differently depending on
which task received it. That was #19, and it is fixed — both now require a name
and check the arrow against the rule kind, with `formalization` importing
`ARROW` from `core.scoring`. One divergence survives: the consequent is
`(-?[A-Za-z]\w*)` in one and `(-?\w+)` in the other, so `[defeasible r1: p => -9x]`
parses in `formalization` and not in the scorer.

Two copies of a rule that must agree is a bug waiting to recur, and seven copies
of the transport regex is the argument for the seam in this section rather than
against it. `parse_operations` becomes one function.

---

## 5. `ScoreResult`

```python
@dataclass
class ScoreResult:
    score: float          # 0.0 - 1.0
    success: bool         # the task's own definition of fully correct
    reason: str           # "ok", "bloated:16_used_vs_5_minimum", ...
    diagnostics: dict     # everything task-specific
```

`score_answer(text, item) -> float` is a thin wrapper, so RL loops and
reasoning-gym-shaped harnesses work unchanged.

### `success` is currently broken, and this is the fix

`score_item` returns a `success` key on three of nine branches
(`arggym/core/scoring.py:209,215,221`). Every early zero — no answer region,
unparseable lines, no directives, all directives illegal, bloated, engine
rejected — returns without it. And the six graded scorers never return it at
all.

The harness then computes

```python
succ = [r for r in rows if r.get("success") is not None]
out["success_rate"] = round(sum(1 for r in succ if r["success"]) / len(succ), 4)
```

So `success_rate` is computed over the rows that reached a late branch of one
scorer. **Half the benchmark is absent from the denominator, and so is every
format failure on the other half.** Any published `success_rate` is overstated.

`success` is therefore defined per task, always returned, and never `None`:

| Task | `success` |
|---|---|
| the six construction tasks | all goals met and the theory consistent |
| `status_query`, `semantics_query`, `perturbation` | exact match over the label map |
| `defeat_diagnosis` | exact match, correct status, survivors correct |
| `claim_chain` | the directive set matches **and** the order constraint holds |
| `formalization` | the theory behaves like the reference on every queried literal |

Two of these are judgement calls and are defended here because a paper will have
to defend them.

**`claim_chain`.** The prompt asks for "all and only the directives that form
the argumentation line justifying {claim}, in order from the premise to the
claim". Set equality covers "all and only"; the order constraint covers "in
order", checked as a property — every rule arrives after everything it rests on
— rather than against the reference sequence, which is one of many valid orders.
`behaviourally_justifies` is *not* in the conjunction: it is computed on the
resolved lines only, so it can hold for an answer that also contains junk.

**`formalization`.** `success` is behavioural equivalence alone. The prompt
states its own success condition behaviourally ("Under a correct formalization:
..." followed by the status of each queried literal), so a formalization that
reproduces every queried status is what was asked for, whatever names and
groupings it used. Requiring `shape_f1 == 1.0` as well would demand the
reference's exact directive multiset, which is stricter than the question and
would make the number uninformative before it made it wrong.

**`success` deliberately excludes minimality.** A construction answer that meets
every goal is a success at `score=0.5`; the efficiency term is a separate
multiplier. Goal satisfaction is the task; economy is a second measurement on
the same answer. `success_rate` and `mean_score` are not two views of one thing.

### Scoring policy is per-item, not a constant

`BLOAT_FACTOR = 2` is stated in the prompt as "more than twice the fewest
directives". Under the rule in section 3 that sentence is rendered from the
value, so changing the constant changes the prompt and the hash. `PARTIAL_CAP`
and the partial-credit weights change the score without changing the prompt, so
they are covered by `scoring_version`. `strict_parse` is a parameter on three of
twelve scorers today and asserted by every prompt; it becomes part of the item.

---

## 6. The generator is the product

```python
ds = arggym.create("status_query", level=9,
                   ordering="weakest_link_elitist", size=100, seed=0)
len(ds); ds[0]; ds.score(text, ds[0])
```

A dataset object is homogeneous in level and ordering, and **the index is the
seed**. Difficulty lives in the config, matching reasoning-gym
(`reasoning_gym/dataset.py:13`).

### On failure, raise

`make_item` can fail to build. The rejected alternative was to skip to the next
seed so the index stays dense. It is wrong for three reasons.

**It is not reproducible.** `ds[5]` could not be computed without knowing
whether seeds 0-4 built, so random access becomes linear, sharding breaks, and
any change that flips one low-index build shifts every item above it while the
config, the seed and `len(ds)` stay identical. A total dataset change presenting
as a no-op.

**It is not what reasoning-gym does.** Every `__getitem__` builds
`Random(self.seed + idx)` once and retries inside that stream, then raises
(`reasoning_gym/games/maze.py:74,126`) or degrades explicitly. No dataset in the
library skips an index.

**It hides a real defect.** The frozen v2 taskset records
`"status_query|L12|last_link_elitist": 12` rejections — 12 of 20 seeds. Dense
indexing turns that into "20 items, looks fine". That is exactly the failure #72
describes: a generator that silently discards half its candidates looks like one
that is working.

So `create` raises on failure, naming the level, ordering, seed and reason. The
frozen artifact is densified once, at export, where it is recorded.

---

## 7. The taskset spec

The spec is the whole input; the manifest records what happened.

```yaml
arggym: "2.1.0"
pyarg: "2.0.2"
prompt_version: 3
theory_schema: 1
scoring_version: 1
profile: FULL
tasks: [status_query, semantics_query, ...]
levels: [3, 6, 9, 12, 15]
orderings: [last_link_elitist, last_link_democratic,
            weakest_link_elitist, weakest_link_democratic]
seeds:
  start: 0
  take: 2            # items required per cell
  scan_limit: 40     # refuse the cell past this
min_acceptance: 0.5  # refuse a cell needing more than 2 seeds per item
```

```jsonc
"status_query|L12|last_link_elitist": {
  "n": 2,
  "seeds_used": [0, 1],
  "seeds_skipped": [{"seed": 2, "reason": "reference_not_irredundant"}],
  "reason_counts": {"reference_not_irredundant": 8, "claim_not_justified": 4},
  "scan_end": 12,
  "acceptance_rate": 0.41
}
```

**`take`, not `stop`.** With a stop bound, a cell that rejects 12 of 20 ships 8
items and nothing complains. With `take` and `scan_limit`, the export either
produces what was asked for or fails loudly.

**Explicit seed lists were rejected.** A spec naming the seeds per cell cannot
be written before running the generator, which makes it an output pretending to
be an input.

**This is also the determinism check, and a better one than "export twice, same
hash".** Compare `seeds_skipped` first: if the skip lists match and the hash
differs, a renderer or a scorer changed; if the skip lists differ, a generator
changed. The hash alone cannot tell you which.

**It needs one generator change to be worth anything.** `build` returns a bare
`None` today and the reason is lost — `claim_chain.build` alone has four
distinct rejection sites. `build` returns a reason instead, `make_item` collects
them, and "L12 last_link_elitist rejects 60%" becomes "it rejects because the
reference fails the irredundance check". That is #72.

### `profile` is recorded, and refused

`arggym/core/curriculum.py` defines four language profiles and nine tasks branch
on them. But `counter_argument` and `semantics_query` cannot take one at all,
only `status_query` mixes it into its seed, `TASK_PROFILES` has no references
anywhere, and no test mentions it. So it is a real axis that is under-built.

It goes in the spec and the manifest as one value for the whole export, which is
what #51 asks for. It stays out of the row `id`, because an id must not promise
a distinction three tasks cannot make. And `create` refuses a non-`FULL` profile
with a message naming the gaps, until they are closed.

---

## 8. Item identity

`id` is `task/L<level>/<ordering>/s<seed>`, and it names the **seed**, because
the seed is what regenerates the item.

What it does not promise: identity across versions. The manifest records
`python` and `pythonhashseed` precisely because the item can depend on them, and
it certainly depends on the arggym commit and the pinned engine. **Ids are
comparable within one taskset**, identified by its `taskset_hash`. Across
tasksets, compare hashes.

`source_index` is the row's position in the file. It is not the seed and is not
part of the id.

---

## 9. Where the harness sits

```
arggym/            the package: generate, render, parse, score
evals/             the reference evaluator: Hydra configs, a client, a report
```

`evals/` never reaches into a private name, and a test enforces it. Dependencies
follow uv's split: anything an adopter installs is an extra, anything local-only
is a PEP 735 dependency group.

```toml
dependencies = ["python-argumentation==2.0.2", "typer>=0.12"]

[project.optional-dependencies]
inspector = ["flask>=3.0"]
evals     = ["hydra-core>=1.3", "omegaconf>=2.3"]
report    = ["matplotlib>=3.8"]

[dependency-groups]
dev = ["pytest>=8.0", "pytest-xdist>=3.0", "ruff>=0.6", "mypy>=1.11"]
```

`pip install arggym` stops pulling a web framework, which is #54.

Run traces do not live here. Reasoning-gym keeps theirs in a separate repository
for the same reason (`eval/README.md`), and the existing local run directory is
3.9 GB.

---

## 10. Reporting

Two rules, both from #9.

**Publish chance floors beside the scores.** Measured on the v1 pilot, a
constant string scored 0.490 on one task and an empty answer scored 0.300 on
another. A score of 0.45 there is worse than answering nothing. No v2 task has a
measured floor yet; measuring them is part of this work, and they must be
re-measured whenever `scoring_version` changes, because a floor is a property of
the scorer.

**Do not rank models by an unweighted mean.** Averaging twelve tasks whose
floors span 0.02 to 0.49, over metrics of four different kinds, produces a
number that moves mostly with which tasks are in the basket. Report the per-task
table. If a single number is wanted, chance-correct per task first:
`(score - floor) / (1 - floor)`.

---

## 11. What this does not fix

Stated so nobody has to discover it.

**A frozen row is only re-scorable against the pinned engine.** Scoring the six
construction tasks runs `python-argumentation==2.0.2` at *scoring* time, not
just at generation time. `score` compares `pyarg_version` and refuses on a
mismatch rather than returning a quietly different number.

**`min_directives` is minimal among the candidates the generator produced**, not
proven globally minimal. This caveat currently lives in the manifest, which
rows get separated from; it moves into the row.

**`formalization` measures template inversion, not argument modelling.** The
formal theory is sampled first and the prose is rendered from it, which is the
only way to have engine-verified gold for a natural-language task. But two
lexical markers give away the two hardest modelling decisions. A high score here
does not license "models can formalize natural-language argumentation".

**Prompts change, and the old pilot is already stale.** The notation block
landed on `main` on 2026-09-04, after the last frozen taskset was built, so
those numbers were already incomparable before this document. There is no cost
to changing prompts now and every reason to land the remaining prompt-text
issues in the same window.
