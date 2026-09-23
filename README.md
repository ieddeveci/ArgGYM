# ArgGYM

A procedural benchmark and RL environment for defeasible reasoning, over ASPIC+ under grounded
semantics. Every item is built by construction and checked against the engine, so the gold answer is
never asserted, it is verified.

Twelve tasks, fifteen levels, four strength orderings. The generator is the product; a frozen
taskset is one dump of it.

```
pip install arggym
```

## Evaluate a model

```python
import arggym

ds = arggym.create("counter_argument", level=6,
                   ordering="weakest_link_elitist", size=50)

for entry in ds:
    answer = my_solver(entry["question"])    # your prompt, your parsing
    result = ds.score(answer, entry)
    print(result.score, result.success, result.reason)
```

ArgGYM owns **what a legal answer is**. You own **how you get one**. A solver is anything that
turns a question into an answer: a bare model, an agent with tools, a symbolic procedure. Composing
the prompt, calling the thing, and pulling the answer out of what came back are all yours, so no
convention of ours is in your way.

The question states the task and what a legal answer must contain, and says nothing about where to
put it. Adding that sentence is yours; `AnswerTemplate` and `extract_answer` are there if you want
the common `<answer>` convention rather than your own:

```python
ds = arggym.create("counter_argument", level=6, size=50)
entry = ds[0]
text = my_model(entry["question"] + "\n" + arggym.XML_TAGS.instruction)
result = ds.score(arggym.extract_answer(text), entry)      # a helper, not a requirement
```

An answer does not have to be text. Every task also takes the value directly, so a solver using a
JSON schema or constrained decoding never writes a directive:

```python
# The names are the ones the question's theory uses, so read them from the item.
ops = arggym.ops_from_json([{"kind": "prefer_rule", "stronger": "xo7", "weaker": "xi7"}])
result = ds.score_value(ops, entry)

# A label map is a plain dict: every claim the question asks about, and its status.
labels = arggym.create("status_query", level=6, size=10)
result = labels.score_value({"ab1": "justified", "cd2": "undecided"}, labels[0])
```

Both take the shape of an answer rather than a correct one -- the names above belong to no
particular item. `entry["metadata"]["gold"]` holds the answer the scorer is checking against.

`ds.score_answer(text, entry)` returns the float alone, so a reasoning-gym-shaped harness or an RL
loop works unchanged.

`examples/evaluate.py` is a working reference: standard library only, reads a frozen taskset, calls
any OpenAI-compatible endpoint, writes a scored JSONL. It reaches into no ArgGYM internal, which is
the point of it.

`evals/` is the harness for a real sweep -- resumable, offline scoring, per-task reporting against
the chance floors -- and it talks to any OpenAI-compatible endpoint. Configs ship for OpenRouter,
the OpenAI API, Gemini's compatibility endpoint and a local vLLM; anything else with that protocol,
Vertex and Azure among them, is a config of its own naming a `base_url` and a key variable. It is
not part of the wheel. `docs/evaluation.md` has it.

```
uv run python -m evals.run   taskset=data/taskset.jsonl model=claude-openrouter
uv run python -m evals.score outputs/runs/<dir>
uv run python -m evals.report outputs/runs/* -o outputs/reports/latest
```

## Freeze a taskset

```
uv run arggym freeze -c tasksets/standard.yaml -o data/taskset.jsonl
```

`tasksets/standard.yaml` is the evaluated grid, as an input you can check in, cite and diff. The
manifest records the arggym and engine versions, which seeds produced the items and which were
skipped, so two exports can be compared by what they skipped and not only by their hash.
It is 12 tasks x 15 levels x 4 orderings x 10 seeds, 7200 rows. `tasksets/lite.yaml` is the same
grid at 2 seeds, 1440 rows, and its rows are the first two of every standard cell.

## A row

```jsonc
{
  "id": "status_query/L9/weakest_link_elitist/s0",
  "task": "status_query",
  "question": "<theory> <ask> <answer format>",
  "reference_answer": "<one correct answer>",
  "metadata": {
    "source_dataset": "status_query", "source_index": 0, "seed": 0, "level": 9,
    "ordering": "weakest_link_elitist", "profile": "FULL",
    "checker": "graded", "answer_shape": "label_map",
    "theory_schema": 1, "pyarg_version": "2.0.2",
    "base_ops": [ ... ],         // the theory, so the row scores without the generator
    "state":    { ... },         // what the question already gives away
    "gold":     { ... }          // everything that gives the answer away
  }
}
```

A row scores on its own: `arggym.score_row(text, row)` and `arggym.score_row_value(value, row)` need
no dataset object and no generator, so stored model outputs can be re-scored later. **Hide
`metadata.gold` from the model and you have hidden the answer** -- that is one rule, and a test
enforces it across all twelve tasks.

Six tasks have no oracle at all. `preference_construction`, both `counter_argument` variants,
`attack`, `defence` and `attack_defense` are graded by running the engine on theory + answer, so any
directive set reaching the goals is correct. `metadata.checker` says which kind a row is; comparing
a model's text to `reference_answer` is wrong on those six.

## Reading a score

Per task, never as an unweighted mean, and against the chance floor. `arggym floors <taskset>`
measures what the best uninformed answer gets, and `arggym.corrected` rescales a score so chance is zero.
`docs/dataset-card.md` says why, and what a high score does *not* license.

## Documentation

| | |
|---|---|
| `docs/dataset-contract.md` | the interface, and why each part is the way it is |
| `docs/evaluation.md` | running a sweep: providers, solvers, and reading the output |
| `docs/dataset-card.md` | what the benchmark measures and what a score licenses |
| `NOTATION.md` | the DSL, the semantics conventions, the answer formats |

## Install

```
pip install arggym                # the library: generate, render, parse, score
pip install "arggym[inspector]"   # + the browser inspector
```

`examples/evaluate.py` needs nothing beyond the library: it is standard library only. A bare
install pulls neither a web framework nor an HTTP client -- `evals/` is not in the wheel, and its
dependencies are a local group. From a checkout, `uv sync` then `uv run pytest`; `make` lists the
shortcuts.

---

# How it is built

The rest of this file is the architecture tour, for anyone changing the generator.

## `arggym/aspic/` -- the engine layer

A thin, typed wrapper over PyArg. Everything above this directory speaks in `Operation` records and
never touches PyArg directly, so the whole suite could be moved to another ASPIC+ implementation by
rewriting these files.

| file | responsibility |
|---|---|
| `engine.py` | The `Operation` record -- one directive of a theory. Also defines `contrary()`, which adds or removes a single `-`. That function IS the contrary relation the whole benchmark rests on; ASPIC+ treats contrariness as part of the system specification rather than deriving it, so this is a modelling choice made explicit in one place. |
| `dsl.py` | The `[directive: ...]` text format, parsed and rendered. The bridge between what a model reads and what the engine evaluates. |
| `api.py` | `ASPICVerifier` -- `status()`, `status_map()`, `is_consistent()`. The single point at which a theory becomes a verdict. Every gold answer in the suite is verified through this class. |

---

## `arggym/core/` -- machinery shared by every task

| file | responsibility |
|---|---|
| `registry.py` | The task table: for each of the twelve, its module, its variant, how its answers are checked, and which item fields a row carries. Data rather than a branch, so adding a task cannot silently miss the export. |
| `dataset.py` | `create(task, level, ordering, size)`. `ds[k]` is the item built from seed `start + k`, and a seed that builds nothing raises instead of being skipped, so an item depends on its coordinates and nothing else. |
| `rows.py` | An item into a row and back. What makes a frozen line scorable without the generator that wrote it, and where a row's engine version is checked before it is scored. |
| `answers.py` | `ScoreResult`, `UnparseableAnswer`, and the `AnswerTemplate`/`extract_answer` pair a harness may use for the common `<answer>` convention. No scorer calls either of those two. |
| `scoring.py` | The directive scorer, shared by the six engine-checked tasks. Parses an answer, rejects illegal additions, checks every goal against the engine, and computes efficiency against the verified minimum. It also emits `achieved_status` and `deadlock_not_defeat`, which separate "created a conflict and failed to win it" from "did nothing" -- two zeroes that mean opposite things. |
| `pairs.py` | `collect` and `pair_f1`, the F1 over (key, label) pairs that the three label-map tasks score with, including the contradiction case where an answer gives one key two labels. |
| `invariants.py` | Checked helpers for the mistakes that recur. `minimal_subset_exact` finds a provably minimal answer using a forced-element prefilter; `split_atoms_and_rules` distinguishes a literal from a rule name, which matters because an undercut's consequent is a rule; `randomize_rule_names` strips ordering and role signal from names. These live here because writing them from memory got them wrong repeatedly. |
| `minimality.py` | The minimum search for `attack_defense`, which has a different shape from the others: no single move is forced, but the task proves a lower bound of one move per chain, so the search starts there instead of at size one. |
| `curriculum.py` | The level-indexed knobs several builders share: junction density, ternary junctions, the per-task junction cap, and the four language profiles. Those numbers came out of generation sweeps rather than being chosen. A level's item shape is not here -- each builder states its own, because the one that lived here had no readers (#31). |
| `prompting.py` | The notation contract, written once: the legal forms, the naming rule, the antecedent rule and the minimality factor, stating what `scoring.py` enforces. Four blocks cover the seven variants that need one, because what an item permits is not a single axis, and each block's docstring names the scoring policy behind its clauses. A block is identical on every item of a variant, so its content leaks nothing about the theory. |
| `nlforms.py` | The natural-language surface forms `formalization` writes its prose from, taken from the ASPIC+ and argumentation-schemes literature: eighteen construct banks holding 128 forms between them, plus the connectives that join them. Vocabulary is shared across banks where it can be, so that one word rarely decides which construct a sentence encodes. |
| `spec.py`, `freeze.py` | A taskset is a spec file. `freeze` fills each cell to the size asked for, refuses a cell it cannot fill, and records every skipped seed with its reason. |
| `floors.py` | What an uninformed answer scores, per task -- one constant per item, or one per key group where the answer key has a component the task fixes. A floor is a property of the scorer, so it is measured from rows rather than written down. |
| `serialize.py` | Operations to JSON and back, with `theory_schema` versioned apart from the package. |

---

## `arggym/structures/` -- argument shapes, used only by `attack_defense`

These build the theories that the three attack_defense modes reason over. They are separate from
`tasks/` because they encode ASPIC+ structure rather than task logic.

| file | responsibility |
|---|---|
| `chains.py` | Nine chain configurations with their attack profiles verified per ordering. A chain's shape decides which attacks are even possible against it -- a strict final rule cannot be undercut, an axiom root cannot be undermined -- which is what forces per-chain discrimination rather than one repeated move. |
| `defence.py` | Deliberate construction of defence items: N attackers, each needing its own treatment, with a verified per-attacker lower bound so the minimum is proven rather than assumed. |
| `interaction.py` | Mixed items built around a shared node, where a naive attack also destroys the claim the model must defend. The interference is verified against a control, so the item genuinely requires a surgical answer. |

---

## `arggym/tasks/` -- one module per task

Each exposes `make_item(level, seed, ordering)`, `parse`, `score_value` and `score`, and owns its own
prompt, curriculum and gold construction. They share `core/` but not each other.

| file | what it asks |
|---|---|
| `status_query.py` | State the status of each named claim. The baseline: everything else presupposes it. The queried set must cover all three statuses with no status above 45% of it, because justified literals outnumber the rest and asking about a representative sample would hand over most of the score. |
| `semantics_query.py` | State the status of each claim under the semantics named beside it: grounded, sceptical or credulous preferred, stable, or eager. The same theory yields different answers under different semantics, so a model that knows only the grounded extension cannot score by default. |
| `attack_defense.py` | Three modes. **attack**: make a justified claim overruled. **defence**: make an attacked claim justified. **attack_defense**: both at once on a shared structure, where the naive attack sabotages the defence. |
| `perturbation.py` | Given a theory and a set of additions, predict which claims change status and to what. Tests prediction rather than action, and requires distinguishing a cascade from a survivor. |
| `counter_argument.py` | Make a claim's *contrary* justified -- which destroying its support does not achieve. Also provides the strict-permitted ablation, where a strict rule wins unconditionally. Its gold is a fixed two-step recipe read off the question, so read it as the contrast against the plain arm rather than as a score of its own (`docs/dataset-card.md`). |
| `preference_construction.py` | Move claims to required statuses using **only** preference directives. Isolates the preference machinery: no undercut, no new argument, only re-weighting. |
| `claim_chain.py` | Write all and only the directives forming the line that justifies a claim. Decoys reach the same claim but are defeated; the true line may itself be attacked yet reinstated, so neither "the unattacked line" nor "any path" works. |
| `defeat_diagnosis.py` | A claim is not justified -- say why. Names every failure point, its defeater, the kind of attack, and why that defeater survives. The complement to `claim_chain`: extraction has a partner in diagnosis. |
| `formalization.py` | Translate a natural-language argumentation into ASPIC+ DSL. Scored on behaviour, per-directive structure, and the type decisions separately, because axiom-versus-premise is behaviourally invisible unless something undermines the literal. |

---

## `arggym/inspector.py` -- reading items by hand

A Flask UI at `http://127.0.0.1:5000`, started with `arggym inspect`. Select a task, level, ordering
and seed; see the exact prompt a model would receive, the reference answer, the verified minimum, the
per-goal engine verdicts, and a rendered view of the theory's structure with statuses inline.
