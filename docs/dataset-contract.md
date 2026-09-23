# The dataset contract

What ArgGYM promises to anyone who evaluates a model against it, and what it
deliberately refuses to promise. This document is the design: what each decision
is, why it is that way, and what was rejected.

---

## 1. The one rule

**ArgGYM owns what a legal answer is. The harness owns how it gets one.**

The harness composes the prompt from the question, calls whatever solver it
likes, and hands over the answer. The dataset receives an answer and scores it.
A solver is anything that turns a question into an answer: a bare model, an
agent with tools, a symbolic procedure.

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
| The answer must be fenced, and the question says how | our implementation | **remove** |
| The answer is DSL *text* | our implementation | **remove** |

The economy row is where the distinction is easy to get wrong: finding the
cheapest set of directives is part of the task, so the bloat rule is the
benchmark's. Where a model puts that set is not.

The boundary is drawn where it is because everything on the harness side is a
choice we have no standing to make. A fence is one way to find an answer in a
completion; a JSON schema, a tool call, constrained decoding and a solver that
returns the answer directly are others, and a benchmark that reads back one
particular pair of delimiters has quietly required its users to imitate it.

So the question states the task and what a legal answer must contain, and stops.
No sentence in it says where to put the answer, and no scorer unwraps anything.

### Why `AnswerTemplate` and `extract_answer` exist anyway

A harness that calls a chat model does need a submission convention, and needs
it in the prompt. The parser is strict, so every line it reads must be an answer
line, and a model that reasons before answering has to put the reasoning
somewhere the parser will not see. Take that sentence out of the prompt entirely
and a reasoning completion is read as a malformed answer, which scores zero for
a reason that has nothing to do with argumentation.

That sentence belongs to the harness, and generation has no way to write it:
no argument to `create()`, no metadata key, no task module that renders one.
`AnswerTemplate` holds the delimiter pair and the sentence that asks for it;
`arggym.XML_TAGS` is the common one, matching reasoning-gym, whose system
prompts and `extract_answer` both use `<answer>` and `</answer>`
(`reasoning_gym/utils.py:8,25`). Both ship in the package rather than in
`evals/`, because `evals/` is not part of the wheel (`pyproject.toml`) and an
adopter installing `arggym` alone would otherwise have neither.

```python
ds = arggym.create("attack", level=6)
question = ds[0]["question"] + "\n" + arggym.XML_TAGS.instruction
```

`extract_answer` reads the `<answer>` region back and returns an unfenced
completion whole. The region is the body of the last complete
`<answer>...</answer>` pair, and a pair runs from the last opening tag before
its close (`AnswerTemplate.region`, `arggym/core/answers.py:43`). So a revised
answer beats the draft before it, and a model that names the tag while
reasoning does not have that prose read as answer lines. An opening tag with no
close after it, as in a truncated generation, is not a pair. The harness in
`evals/` reads every template through the same method. Nothing in ArgGYM's
scorers calls `extract_answer` (`arggym/core/answers.py:70`);
`examples/evaluate.py` names its convention once and derives both the
instruction and the extraction from it, which is the shape a harness wants.

The content half of the answer-format block belongs to the task and does not
move: "one directive per line", "copied exactly as it appears above", "one line
per claim, written as `claim: status`". Nothing is appended to it in generation.

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
  "question": "<theory> <ask> <answer format>",
  "reference_answer": "<one correct answer, raw>",
  "metadata": {
    "source_dataset": "status_query",     // the registered task name
    "source_index": 0,                    // row position in this file
    "seed": 0,                            // the seed that built it
    "level": 9,
    "ordering": "weakest_link_elitist",
    "profile": "FULL",

    "theory_schema": 1,
    "pyarg_version": "2.0.2",

    "checker": "graded",                  // exact | graded | verified
    "answer_shape": "label_map",          // see 4

    "base_ops": [ ... ],                  // the theory shown in the question
    "state": { ... },                     // per-task, non-gold
    "gold": { ... }                       // everything that gives the answer away
  }
}
```

The theory field is named for what it holds, so a task that shows two theories
carries two: `perturbation` has `base_ops` and `pert_ops`, both rendered into
the question (`arggym/core/registry.py:143-145`). `formalization` has neither,
because the theory is the answer rather than the question. A row written by
`arggym freeze` also carries `metadata.reference_score`, the score its own
reference answer got, and the arggym, prompt and scoring versions live once in
the taskset manifest rather than on every row.

Three fields carry most of the design.

### `reference_answer`, not `answer`

The field is one correct answer, not the key to the answer. For six of the
twelve tasks there is no oracle at all: `preference_construction`,
`counter_argument`, `counter_argument_strict`, `attack`, `defence` and
`attack_defense` are graded by running the engine on theory + answer and
checking the goals, so any directive set that reaches them is correct. Naming
the field `answer` invites string comparison, and string comparison is wrong on
those six. It is also wrong on `formalization`, whose score is a weighted blend
of behavioural, shape and type components
(`arggym/tasks/formalization.py:615-618`, section 5): a correct formalization written
differently matches at 0 and scores high.

Reasoning-gym has the same distinction and handles it by overriding
`score_answer` per dataset (`reasoning_gym/dataset.py:63`, *"Overwrite this
method in derived classes if a single oracle answer is not available"*). Each
ArgGYM task implements its own scorer and inherits no default, so there is
nothing to fall back to substring matching.

### `metadata.gold`

Everything that gives the answer away sits under one key, so "do not show the
model `metadata.gold`" is one rule instead of twelve. It is not always obvious
what belongs there. `min_directives` does, because the prompt never states how
many directives are needed and publishing the number beside the question leaks
the answer size. So does `perturbation.survivors`, which names the claims that
did *not* change. The generator's own statistics blob is gold by default
(`arggym/core/registry.py:104`): most of it describes the reference, and an
allowlist of the safe keys would leak the first one somebody forgot.

### `checker`

Three values, describing how an answer is judged rather than whether the
reference is unique:

- **`exact`** -- normalized comparison against the reference is sound.
- **`graded`** -- a continuous scorer over a unique gold.
- **`verified`** -- the engine is run on theory + answer.

No task is `exact`. The value exists because a future task might be, and because
a harness needs to know that `graded` and `verified` rows cannot be scored by
comparison.

---

## 3. The question is a view of the state

The question text is a pure function of the item's structured state, of which
the operation list is one field.

This is not tidiness. It closes a bug class that has shipped twice: an item with
internally coherent operations, correctly computed gold, a reference that
self-scores 1.0, and a prompt the model cannot answer, because nothing checked
that the prompt and the gold describe the same theory. The pre-freeze audit
(#12) found that `validate_entry` read `entry["answer"]` and
`entry["metadata"]` and never inspected the prompt at all. Render the question
from the state and the two cannot disagree.

**One exception, named.** `formalization`'s question is natural-language prose,
and the sentences and the operations are generated together in one interleaved
RNG stream (`arggym/tasks/formalization.py:168-383`). Re-rendering would need
the RNG state at each sentence, which is not a property of the operations. Its
surface text is itself generated state and is stored as a string. That is also
the task whose `reference_ops` *is* the gold answer
(`arggym/tasks/formalization.py:97,371`) rather than the theory in the question,
which is why the schema names keys by what they hold -- `base_ops`, `pert_ops`,
`reference_ops` -- rather than one generic `theory`.

---

## 4. Answers are values; text is one serialization

Every task exposes three functions, and `score` is the composition of the other
two rather than a fourth implementation:

```python
parse(text, item) -> Value          # text -> the answer, no scoring
score_value(value, item) -> ScoreResult
score(text, item)  = score_value(parse(text, item), item)
```

A solver with a JSON schema, a tool call or constrained decoding submits the
value and never writes a directive. A solver returning text calls `parse` first.
Both reach the same scorer and get the same object back, which is the property
worth having, rather than two scorers that happen to agree today.

`parse` raises `UnparseableAnswer` on text that spells out no answer at all;
`score_value` never raises it, because a solver that hands over a value has done
its own parsing and its failures are its own. `score` turns the exception back
into a zero `ScoreResult`, so a harness scoring a whole taskset gets a row
rather than a stack trace on one bad generation.

Empty is a value, not a failure: submitting nothing is an answer, and the score
says so.

The seam is where it is because the text is thrown away anyway. For the seven
operation-list tasks the scorer converts DSL text into `Operation`s and grades
the resulting engine state; nothing downstream sees a character of what the
model wrote. Requiring text would therefore be us making a user imitate our
serialization, and the cost of that is measured (#10): across the pilot runs
`no_answer_region` reached 0.135 and `zero_score_with_valid_region` was 0.180 on
qwen3.6-27b -- answers that transported fine and died in the parser, some
confirmed correct. That is larger than some level effects, which makes it a
confound rather than a finding.

**Four answer shapes.**

| Shape | Tasks | Value |
|---|---|---|
| operation list | `preference_construction`, both `counter_argument`, `attack`, `defence`, `attack_defense`, `formalization` | `List[Operation]` |
| label map | `status_query`, `semantics_query`, `perturbation` | `Dict[key, status]`, or a list of statuses per key where an answer contradicts itself |
| ordered sequence | `claim_chain` | `List[str]`, order scored (`arggym/tasks/claim_chain.py:435-441`) |
| record list | `defeat_diagnosis` | `status` + `List[{defeated_at, defeater, kind, survives_because?}]` |

Every parser takes `parse(text, item)`, including the ones that read nothing
from the item. A uniform signature is what lets a harness loop over tasks
without a table of exceptions, and it costs an unused parameter in the two tasks
whose text can be read without knowing the theory.

Resolving a quoted line against the theory is `claim_chain`'s *scoring*, not its
parsing: `parse` returns the lines the answer gave, and `score_value` decides
which of them the theory contains. Drawing it the other way would put a piece of
grading in the half a solver is allowed to replace.

### One parser

The DSL is parsed in one place: three patterns at `arggym/core/scoring.py:14-22`
and `parse_answer` at `:33`. `formalization` imports it
(`arggym/tasks/formalization.py:38`) rather than
reading the same grammar again. Two copies of a rule that must agree diverge:
when they existed, one made the rule name optional and invented one while the
other required it, so the same answer text scored differently depending on which
task received it (#19). One definition cannot do that.

---

## 5. `ScoreResult`

```python
@dataclass(frozen=True)
class ScoreResult:
    score: float          # 0.0 - 1.0
    success: bool         # the task's own definition of fully correct
    reason: str           # "ok", "bloated:16_used_vs_5_minimum", ...
    diagnostics: dict     # everything task-specific
```

`score_answer(text, item) -> float` is a thin wrapper, so RL loops and
reasoning-gym-shaped harnesses work unchanged.

### `success` is defined per task and always returned

`success` has no default, so a scorer that omits it fails at construction rather
than returning `None` and dropping out of a harness's denominator. A
`success_rate` computed over the rows where the field happened to be present is
overstated by exactly the rows that failed early, which is the failure this
rules out.

| Task | `success` |
|---|---|
| the six construction tasks | all goals met and the theory consistent |
| `status_query`, `semantics_query`, `perturbation` | exact match over the label map |
| `defeat_diagnosis` | exact match, correct status, survivors correct |
| `claim_chain` | the directive set matches **and** the order constraint holds |
| `formalization` | the theory behaves like the reference on every literal the reference names |

Two of these are judgement calls and are defended here because a paper will have
to defend them.

**`claim_chain`.** The prompt asks for "all and only the directives that form
the argumentation line justifying {claim}, in order from the premise to the
claim". Set equality covers "all and only"; the order constraint covers "in
order", checked as a property -- every rule arrives after everything it rests on
-- rather than against the reference sequence, which is one of many valid
orders. `behaviourally_justifies` is *not* in the conjunction: it is computed on
the resolved lines only, so it can hold for an answer that also contains junk.

**`formalization`.** `success` is behavioural equivalence: the answer's theory
gives the same status as the reference to every atom the reference names
(premise and axiom contents, rule antecedents and consequents, rule names
excluded), in both polarities, with `UNSATISFIABLE` for a literal no argument
concludes. Names and groupings are free. Requiring `shape_f1 == 1.0` as well
would demand the reference's exact directive multiset, which is stricter than
the question and would make the number uninformative before it made it wrong.

The queried literals alone would not do. The question prints their statuses
("Under a correct formalization: ..."), and one or two directives per printed
status reproduce them without reading the text. An undercut written as a
rebuttal also agrees on them, while it makes the rebutting literal justified
where the reference has no argument for it. `diagnostics.mismatched_literals`
names the first six literals on which a failed answer differs.

Behaviour cannot see a directive that changes no status, and success does not
require one. The largest case is an undercut of a strict rule, which ASPIC+
makes inert (`NOTATION.md`, "an undercut cannot be aimed at a strict rule"):
the generator aims half its undercut units at a strict rule
(`arggym/tasks/formalization.py:347`), so on 82 of the 120 `formalization`
rows of `tasksets/lite.yaml` the success test cannot tell whether that
undercut was written. A support rule with a second route to its conclusion, or
an undercut of such a rule, is the same case. Dropping the reference's last
rule still succeeds on 44 of those 120 items: 19 drop an inert undercut, 20 a
support rule with a second route, and 5 an undercut of a defeasible rule with
a second route. Only the shape term sees those omissions. By the same
argument, a theory with no effective attack is behaviourally its atoms, so on a
small reference an answer that lists the atoms as premises can still succeed
(11 of the 120, all at levels 1 and 2).

The other side of this holds. An answer that makes an inert undercut work by
demoting its strict target to a defeasible rule fails success on 58 of the 82
items with such an undercut, at a mean score of 0.807. On each of the 58 the
question prints the target's conclusion as justified and the demoted rule loses
it, so the test calls the demotion wrong, which it is. On the other 24 no status
the reference names moves, and the demotion costs only type and shape credit.

The score adds what behaviour cannot see. It is `0.25 * behavioural + 0.35 *
shape_f1 + 0.40 * type_score`, or `0.4 * behavioural + 0.6 * shape_f1` where
the gold has no axiom and no strict rule. `behavioural` is the F1 of the queried
statuses. `type_score` is the F1 of the axiom and strict decisions: recall is
the share of the gold's axioms and strict rules the answer writes with that
type, and precision is the share of the answer's axiom and strict directives
that the gold types the same way. An axiom where the text gives a premise costs
precision, and a premise where it gives an axiom costs recall.

**`success` deliberately excludes minimality.** A construction answer that meets
every goal is a success at `score=0.5`; the efficiency term is a separate
multiplier. Goal satisfaction is the task; economy is a second measurement on
the same answer. `success_rate` and `mean_score` answer two questions.

### Scoring policy is versioned, not configurable

`scoring_version` covers everything that moves a score without moving a prompt:
`PARTIAL_CAP`, the partial-credit weights, the F1 details. `prompt_version`
covers the row a harness reads: the question text, and the metadata schema too,
since dropping a key changes the taskset hash without changing a question.
A rule stated in the prompt and enforced by the scorer
moves both and bumps both -- the prompt hash records that the question changed,
`scoring_version` records that an unchanged question is now scored differently.
The bloat factor is stated twice on purpose, as
`BLOAT_FACTOR = 2` (`arggym/core/scoring.py:12`) and as the sentence "more than
twice the fewest directives that work scores zero"
(`arggym/core/prompting.py:39`), because the model has to be told the rule it is
scored by. Changing it means changing both and bumping both versions.

Strict parsing is not a parameter. Every prompt states the rule flatly, so an
item that relaxed it would carry a question claiming a rule the scorer was not
applying, which is the mismatch section 3 exists to stop. A study that wants to
separate reasoning from format compliance reports a second score beside the
first, rather than making the question untrue.

---

## 6. The generator is the product

```python
ds = arggym.create("status_query", level=9,
                   ordering="weakest_link_elitist", size=100, seed=0)
len(ds); ds[0]; ds.score(text, ds[0])
```

A dataset object is homogeneous in level and ordering, and `ds[k]` is the item
built from seed `start + k`. Difficulty lives in the config, matching
reasoning-gym (`reasoning_gym/dataset.py:13`).

### On failure, raise

`make_item` can fail to build, and then `ds[k]` raises `BuildFailed` naming the
task, level, ordering and seed (`arggym/core/dataset.py:95-99`). The rejected
alternative was to skip to the next seed so the index stays dense. It is wrong
for three reasons.

**It is not reproducible.** `ds[5]` could not be computed without knowing
whether seeds 0-4 built, so random access becomes linear, sharding breaks, and
any change that flips one low-index build shifts every item above it while the
config, the seed and `len(ds)` stay identical. A total dataset change presenting
as a no-op.

**It is not what reasoning-gym does.** Every `__getitem__` builds
`Random(self.seed + idx)` once and retries inside that stream, then raises
(`reasoning_gym/games/maze.py:74,126`) or degrades explicitly. No dataset in the
library skips an index.

**It hides a real defect.** A cell can reject most of the seeds it is given.
Dense indexing turns "rejected most of them" into "full cell, looks fine", which
is the failure #72 describes: a generator that silently discards its candidates
looks like one that is working.

Densifying is the export's job, where it is recorded.

---

## 7. The taskset spec

The spec is the whole input; the manifest records what happened.

```yaml
arggym: "2.0.0"
pyarg: "2.0.2"
theory_schema: 1
tasks: [preference_construction, counter_argument, ...]
levels: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
orderings: [last_link_elitist, last_link_democratic,
            weakest_link_elitist, weakest_link_democratic]
seeds:
  start: 0
  take: 10           # items required per cell; lite.yaml asks for 2
  scan_limit: 40     # refuse the cell past this
min_acceptance: 0.5  # refuse a cell needing more than 2 seeds per item
min_build_acceptance: 0.13  # refuse a cell keeping under 13 candidates in 100
profile: FULL
```

An unknown key is refused rather than ignored (`arggym/core/spec.py:121-126`),
and `prompt_version` and `scoring_version` may be pinned the same way as
`arggym` and `pyarg` when a spec wants to assert them.

```jsonc
"semantics_query|L9|last_link_elitist": {
  "n": 2,
  "seeds_used": [0, 1],
  "seeds_skipped": [],           // {"seed", "tries", "reason", "reasons"} per failure
  "reason_counts": {},           // over seeds_skipped
  "scan_end": 1,
  "acceptance_rate": 1.0,
  "build_calls": 5,              // candidates build was asked for, across every seed
  "build_rejections": {"theory_over_max_directives": 3},
  "build_acceptance_rate": 0.4
}
```

**`take`, not `stop`.** With a stop bound, a cell that rejects most of its scan
ships what it got and nothing complains. With `take` and `scan_limit`, the
export either produces what was asked for or fails naming the cell.

**Explicit seed lists were rejected.** A spec naming the seeds per cell cannot
be written before running the generator, which makes it an output pretending to
be an input.

**This is also the determinism check, and a better one than "export twice, same
hash".** Compare `seeds_skipped` first: if the skip lists match and the hash
differs, a renderer or a scorer changed; if the skip lists differ, a generator
changed. The hash alone cannot tell you which. Since #124 there is a third case:
a skip list also differs when the freeze's own acceptance policy changes, because
`minimality_unproven` refuses a candidate after `build` has returned it. Read the
skip reasons before reaching for the generator.

**The build counts, not just the seed counts.** `seeds_skipped` records a seed
that exhausted its retry budget, and the grid barely uses it: across the 96
cells of levels 3 and 9, not one seed failed and every `acceptance_rate` came
out 1.00. The cell above still throws away five candidates out of seven. So
`build` answers `Rejected(reason)` rather than `None` (#72), and the cell records
how many candidates it asked for and what each refusal was, counted over every
seed rather than only the failed ones. A reason is a short stable key naming the
check that failed, so it stays a histogram bucket; `arggym/core/build.py` holds
the vocabulary.

Read `build_acceptance_rate` when asking whether a cell is healthy, and only
against another export of the same `seeds.take`: the scan stops as soon as the
cell is full, so `semantics_query` at level 13 under weakest-link democratic
reports 0.17 at `take: 2` and 0.26 at `take: 10`. Two exports whose reasons move while the
hash holds mean a generator changed what it discards without changing what it
ships. `min_acceptance` gates the seed rate, which falls below 1.00 on the
standard grid only where a `preference_construction` seed is skipped for
`minimality_unproven` (below), bottoming at 0.83; `min_build_acceptance` gates
the candidate rate the retry loop reports. The standard and lite specs share
0.13, set under the thinnest cell of either: `perturbation` at level 6 under
last-link elitist keeps 10 candidates of 44 at `take: 10`, and `semantics_query`
at level 13 under weakest-link democratic keeps 2 of 12 at `take: 2` (#114).
Those floors move with `seeds.take` for the reason above, so a spec that changes
`take` has to measure its own floor rather than inherit these. A spec that names
neither guard keeps the seed guard alone.

One more reason a seed is skipped comes after `build` rather than from it:
`minimality_unproven`. The six construction generators write `minimality_proven`
into the row's statistics, and a row whose minimum search ran out of budget
carries an upper bound under the name of a minimum -- on the row #124 found it
said 55 where 10 suffice, so its bloat gate admitted five times what it should.
The freeze skips that seed and names it, the way it names a seed that built
nothing, and the skip carries the candidate rejections that seed had already
recorded. The standard grid skips five, all `preference_construction` under
weakest-link democratic at levels 9, 12, 14 and 15. A `minimality_unproven` skip
is counted in `build_rejections`, and so in `n_rejected`, beside `build`'s own
refusals: one histogram answers what the cell threw away and why, where two
would make every reader add them up. The cost is that neither field is `build`'s
alone any more -- of the 1339 candidates `n_rejected` counts on the standard
grid, 5 are rows `build` returned and the freeze refused.

### `profile` is recorded, and refused

`arggym/core/curriculum.py:105-118` defines four language profiles and nine
tasks branch on them. But `counter_argument` and `semantics_query` take no
profile at all, only `status_query` mixes it into its seed, and `TASK_PROFILES`
has no reader. It is a real axis that is under-built.

So it goes in the spec and the manifest as one value for the whole export, which
is what #51 asks for. It stays out of the row `id`, because an id must not
promise a distinction three tasks cannot make. And `create` refuses a non-`FULL`
profile with a message naming the gaps (`arggym/core/dataset.py:65-70`), until
they are closed.

---

## 8. Item identity

`id` is `task/L<level>/<ordering>/s<seed>` (`arggym/core/dataset.py:139`), and it
names the **seed**, because the seed is what regenerates the item.

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
arggym/            the package: generate, render, score
examples/          a reference evaluator, standard library only
evals/             the harness: run a solver, score it, report it
```

`examples/evaluate.py` reads a frozen taskset, calls any OpenAI-compatible
endpoint, and hands the answer to `arggym.score_row`. It uses three public names
and reaches into no private one, which is the demonstration: if the public
surface were not enough, this file could not exist. It also shows the three
decisions a harness owns. It composes the prompt, naming its submission
convention once so the instruction and the extraction cannot disagree. It calls
the model. And it keeps an API error, a truncation and a wrong answer as three
events, because reporting infrastructure trouble as a reasoning result is the
mistake this field makes routinely.

Dependencies follow uv's split: anything an adopter installs is an extra,
anything local-only is a PEP 735 dependency group.

```toml
dependencies = ["python-argumentation==2.0.2", "typer>=0.12", "pyyaml>=6.0"]

[project.optional-dependencies]
inspector = ["flask>=3.0"]
report    = ["matplotlib>=3.8"]

[dependency-groups]
evals = ["openai>=3.8", "hydra-core>=1.3", "tqdm>=4.66"]
dev   = [{include-group = "evals"}, "pytest>=8.0", "pytest-xdist>=3.0",
         "ruff>=0.6", "flask>=3.0"]
```

`pip install arggym` pulls no web framework, which is #54, and CI checks that
rather than trusting a comment. It pulls no HTTP client either: `evals/` is not
in the wheel and its dependencies are a local group, so an adopter who scores
model outputs in CI receives nothing for talking to a model provider.

`evals/` is one harness, not the harness. It reaches an OpenAI-compatible
endpoint -- which every provider we use serves -- and holds its solver behind a
four-line protocol, so a team preferring litellm, pydantic-ai or an agent
framework writes a solver rather than a fork (`docs/evaluation.md`).

Run traces do not live here. They are large, and reasoning-gym keeps theirs in a
separate repository for the same reason (`eval/README.md`).

---

## 10. Reporting

Two rules, both from #9, and the definition the first one rests on.

**Publish chance floors beside the scores.** `arggym floors <taskset>` measures
what the best uninformed answer gets on each task, and it is not small: on the
level-3 rows of `tasksets/lite.yaml`, `semantics_query` sits at 0.727 (0.786 over
the 40 level-3 rows of the standard grid), `status_query` at 0.375, `formalization` at
0.219, `claim_chain` at 0.190, `perturbation` at 0.153 and `defeat_diagnosis` at
0.131, while the six engine-checked tasks sit at 0.000 because no fixed answer
reaches a goal. A score of 0.45 on `semantics_query` is worse than answering the
same thing every time. Every floor is printed with the strategy that reached
it, so a zero says which of two things happened: the search found nothing that
fits the answer format, or nothing that fits it pays. `formalization` was the
first and read as the second until the search learned to write a directive.
Ties go to `empty`, which is listed first, so the six zeroes above are the
second case. A floor is a property of the scorer and of the search, so it is
re-measured whenever `scoring_version` or `FLOORS_VERSION` moves, and a scoring
artifact records both, since a change to the search moves a floor without
touching a scorer.

What counts as uninformed is fixed here rather than left to whatever the search
happens to try. A floor strategy fixes, once per task, the answer it gives to
each coordinate the answer format exposes; it reads the item only to learn which
coordinates are asked, and never inspects the theory to decide what to answer --
equivalently, its answer is invariant under any change to the theory that leaves
the ask list and the closed vocabularies alone. Enumerating the asked
coordinates is not deciding an answer to them, and that holds where the theory
is the only place they are written: `perturbation` asks which claims of the
original theory changed status and lists none of them, so its ask list is every
claim that theory mentions, and the status it gives each of them is one constant
fixed once per task. The enumeration has to be total, and *every* is the word
carrying that. Choosing which of the exposed coordinates to answer is deciding
an answer to the ones left out, and it pays: `pair_f1` is `2tp/(|pred| +
|gold|)`, so on this task, where the floor answers 3152 coordinates against a
gold of 1076 over the shipped grid, dropping the claims nothing in the theory
attacks would raise the number by shrinking the denominator alone. That rule
reads the theory to decide, which is the next paragraph's definition of a
solver. The invariance covers less here than elsewhere for the same reason: an
edit to a `perturbation` theory usually moves its ask list, so what this floor
is invariant under is the edits that leave the theory's claims alone.

One constant per item is not the widest map that allows: `semantics_query` asks
its claims under as many as five semantics at once, and one constant per
semantics is fixed in the same sense, worth 0.657 over the shipped grid against
0.461 for the best single constant (#95). Such a map is searched against labels,
so it is fitted
once over a whole task and reused for every group the report corrects, and it is
admitted only while it stays small: a handful of entries, each answering many
coordinates. A map with one entry per coordinate is the gold, not a floor.

A rule that reads the theory is a solver -- "an odd cycle of undercuts means no
stable extension", "a rule consequent is usually justified", anything
conditioned on theory size -- and reporting one as the floor would leave "the
model beat the floor" saying nothing, because past that line nothing stops short
of a full solver. `copy_theory` is the one strategy outside the rule and it is
not a guess: it echoes the input, and it is searched to check what a scorer pays
for echoing -- on `claim_chain` that is 0.155, which is that task's floor.

**A floor is a lower bound, not the best score available without reasoning.** It
is the best of what the search tried, and drawing the line above rules out
strategies that do better: on `semantics_query`, a program that reads the odd
ring of undercuts off the theory beats this floor by several points
(`docs/dataset-card.md`). The gap is deliberate and it is not free -- an
understated floor inflates `corrected` -- but a floor that a theory-reading
program can raise is no longer a floor.

**Do not rank models by an unweighted mean.** Averaging twelve tasks whose
floors span half the range, over metrics of four different kinds, produces a
number that moves mostly with which tasks are in the basket. Report the per-task
table. If a single number is wanted, chance-correct per task first:
`(score - floor) / (1 - floor)`, which is `arggym.corrected`.

---

## 11. Known limits

Stated so nobody has to discover them.

**A frozen row is only re-scorable against the pinned engine.** Scoring the six
construction tasks runs `python-argumentation==2.0.2` at *scoring* time, not
just at generation time. `score_row` compares `pyarg_version` and refuses on a
mismatch rather than returning a quietly different number
(`arggym/core/rows.py:203-226`).

**`min_directives` is minimal among the candidates the generator produced**, not
proven globally minimal. The caveat is written into the taskset manifest; a row
separated from its manifest does not carry it.

**`formalization` measures template inversion, not argument modelling.** The
formal theory is sampled first and the prose is rendered from it, which is the
only way to have engine-verified gold for a natural-language task. But two
lexical markers give away the two hardest modelling decisions. A high score here
does not license "models can formalize natural-language argumentation".

**The twelve scorers are not consistent with each other in their diagnostics.**
Three names describe one condition -- `no_parseable_lines`, `no_parseable_pairs`,
`no_pairs` -- diagnostic key sets vary by branch and by task, and `parse("")` is
an empty value for the operation-list tasks and an `UnparseableAnswer` for the
three label tasks. A harness aggregating across tasks has to test for presence
rather than assume a shape.
