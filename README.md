# ArgGYM

A procedural benchmark and RL environment for defeasible reasoning, over ASPIC+ under grounded
semantics. Every item is built by construction and checked against the engine, so the gold answer is
never asserted, it is verified.

Twelve tasks, five evaluated levels, four strength orderings. The generator is the product; a frozen
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
    text = my_model(entry["question"])       # the question asks for <answer> tags
    result = ds.score(arggym.extract_answer(text), entry)
    print(result.score, result.success, result.reason)
```

ArgGYM owns **what a legal answer is**; you own **how it is delivered**. The question asks for the
answer between `<answer>` and `</answer>`, matching reasoning-gym, and `arggym.extract_answer(text)`
reads that region back. Which delimiters the question names is a render-time choice, not a property
of the benchmark: `arggym.create(..., template=AnswerTemplate("boxed", r"\boxed{", "}"))` asks for
those instead, and each row records which template built its question. Scoring unwraps `<answer>`
and nothing else, so a harness that renders its own template unwraps its own answers and hands the
body to `score` -- which is what a harness with constrained decoding or a JSON schema already does.

The fence earns its place: the parser is strict, so every line inside it has to be an answer line.
Without one, a model that reasons before writing its answer would have its reasoning read as a
malformed answer.

`ds.score_answer(text, entry)` returns the float alone, so a reasoning-gym-shaped harness or an RL
loop works unchanged.

`examples/evaluate.py` is a working reference: standard library only, reads a frozen taskset, calls
any OpenAI-compatible endpoint, writes a scored JSONL. It reaches into no ArgGYM internal, which is
the point of it.

## Freeze a taskset

```
uv run arggym freeze -c tasksets/standard.yaml -o data/taskset.jsonl
```

`tasksets/standard.yaml` is the evaluated grid, as an input you can check in, cite and diff. The
manifest records the arggym and engine versions, the seeds used, and every seed skipped with its
reason — so two exports can be compared by what they skipped, not only by their hash.

## A row

```jsonc
{
  "id": "status_query/L9/weakest_link_elitist/s0",
  "task": "status_query",
  "question": "<theory> <ask> <notation>",
  "reference_answer": "<one correct answer>",
  "metadata": {
    "source_dataset": "status_query", "seed": 0, "level": 9,
    "ordering": "weakest_link_elitist",
    "checker": "graded", "answer_shape": "label_map",
    "base_ops": [ ... ],     // the theory, so the row scores without the generator
    "state":    { ... },     // what the question already gives away
    "gold":     { ... }      // everything that gives the answer away
  }
}
```

A row scores on its own: `arggym.score_row(text, row)` needs no dataset object and no generator, so
stored model outputs can be re-scored later. **Hide `metadata.gold` from the model and you have hidden
the answer** — that is one rule, and a test enforces it across all twelve tasks.

Six tasks have no oracle at all. `preference_construction`, both `counter_argument` variants,
`attack`, `defence` and `attack_defense` are graded by running the engine on theory + answer, so any
directive set reaching the goals is correct. `metadata.checker` says which kind a row is; comparing
a model's text to `reference_answer` is wrong on those six.

## Reading a score

Per task, never as an unweighted mean, and against the chance floor. `docs/dataset-card.md` says why,
and what a high score does *not* license.

## Documentation

| | |
|---|---|
| `docs/dataset-contract.md` | the interface, and why each part is the way it is |
| `docs/dataset-card.md` | what the benchmark measures and what a score licenses |
| `NOTATION.md` | the DSL, the semantics conventions, the answer formats |

## Install

```
pip install arggym                # the library: generate, render, parse, score
pip install "arggym[inspector]"   # + the browser inspector
```

`examples/evaluate.py` needs nothing beyond the library: it is standard library only.
A bare install pulls no web framework. From a checkout, `uv sync` then `uv run pytest`; `make` lists
the shortcuts.

---

# How it is built

The rest of this file is the architecture tour, for anyone changing the generator.

## `arggym/aspic/` — the engine layer

A thin, typed wrapper over PyArg. Everything above this directory speaks in `Operation` records and
never touches PyArg directly, so the whole suite could be moved to another ASPIC+ implementation by
rewriting these three files.

| file | responsibility |
|---|---|
| `engine.py` | The `Operation` record — one directive of a theory. Also defines `contrary()`, which adds or removes a single `-`. That function IS the contrary relation the whole benchmark rests on; ASPIC+ treats contrariness as part of the system specification rather than deriving it, so this is a modelling choice made explicit in one place. |
| `dsl.py` | The `[directive: ...]` text format, parsed and rendered. The bridge between what a model reads and what the engine evaluates. |
| `api.py` | `ASPICVerifier` — `status()`, `status_map()`, `is_consistent()`. The single point at which a theory becomes a verdict. Every gold answer in the suite is verified through this class. |

---

## `arggym/core/` — machinery shared by every task

| file | responsibility |
|---|---|
| `scoring.py` | The directive scorer, used by `attack_defense`, `counter_argument` and `preference_construction`. Parses an answer, rejects illegal additions, checks every goal against the engine, and computes efficiency against the verified minimum. It also emits `achieved_status` and `deadlock_not_defeat`, which separate "created a conflict and failed to win it" from "did nothing" — two zeroes that mean opposite things. |
| `invariants.py` | Checked helpers for the mistakes that recur. `minimal_subset_exact` finds a provably minimal answer using a forced-element prefilter; `split_atoms_and_rules` distinguishes a literal from a rule name, which matters because an undercut's consequent is a rule; `randomize_rule_names` strips ordering and role signal from names. These live here because writing them from memory got them wrong repeatedly. |
| `minimality.py` | The minimum search for `attack_defense`, which has a different shape from the others: no single move is forced, but the task proves a lower bound of one move per chain, so the search starts there instead of at size one. |
| `curriculum.py` | Level shapes and the measured floors behind them. The numbers here were derived from generation sweeps, not chosen. |
| `prompting.py` | The permitted-directive block, which states what `scoring.py` enforces: the legal forms, the naming rule, the antecedent rule and the minimality factor. It is identical on every item of a variant, so its content leaks nothing about the theory. `attack_defense` renders its whole prompt here; `counter_argument` renders the block. |
| `nlforms.py` | Eighty-nine natural-language surface forms for `formalization`, taken from the ASPIC+ and argumentation-schemes literature. The giveaway words are deliberately shared between the constructs they used to separate. |
| `registry.py` | The task table: for each of the twelve, its module, its variant, how its answers are checked, and which item fields a row carries. Data rather than a branch, so adding a task cannot silently miss the export. |
| `dataset.py` | `create(task, level, ordering, size)`. The index **is** the seed and a failed build raises, so an item depends on its coordinates and nothing else. |
| `rows.py` | An item into a row and back. What makes a frozen line scorable without the generator that wrote it. |
| `spec.py`, `freeze.py` | A taskset is a spec file. `freeze` fills each cell to the size asked for, refuses a cell it cannot fill, and records every skipped seed with its reason. |
| `answers.py` | One answer extractor and one `ScoreResult`. Delimiters are accepted and never required. |
| `serialize.py` | Operations to JSON and back, with `theory_schema` versioned apart from the package. |
| `export.py` | The previous per-task export, kept while callers move to `freeze`. |

---

## `arggym/structures/` — argument shapes, used only by `attack_defense`

These build the theories that the three attack_defense modes reason over. They are separate from
`tasks/` because they encode ASPIC+ structure rather than task logic.

| file | responsibility |
|---|---|
| `chains.py` | Seven chain configurations with their attack profiles verified per ordering. A chain's shape decides which attacks are even possible against it — a strict final rule cannot be undercut, an axiom root cannot be undermined — which is what forces per-chain discrimination rather than one repeated move. |
| `defence.py` | Deliberate construction of defence items: N attackers, each needing its own treatment, with a verified per-attacker lower bound so the minimum is proven rather than assumed. |
| `interaction.py` | Mixed items built around a shared node, where a naive attack also destroys the claim the model must defend. The interference is verified against a control, so the item genuinely requires a surgical answer. |

---

## `arggym/tasks/` — one module per task

Each exposes `make_item(level, seed, ordering)` and a scorer, and owns its own prompt, curriculum and
gold construction. They share `core/` but not each other.

| file | what it asks |
|---|---|
| `status_query.py` | State the status of each named claim. The baseline: everything else presupposes it. Queried claims are selected for balance, because 84% of literals in these theories are justified and asking about all of them would hand over most of the score. |
| `semantics_query.py` | State the status of each claim under the semantics named beside it: grounded, sceptical or credulous preferred, stable, or eager. The same theory yields different answers under different semantics, so a model that knows only the grounded extension cannot score by default. |
| `attack_defense.py` | Three modes. **attack**: make a justified claim overruled. **defence**: make an attacked claim justified. **attack_defense**: both at once on a shared structure, where the naive attack sabotages the defence. |
| `perturbation.py` | Given a theory and a set of additions, predict which claims change status and to what. Tests prediction rather than action, and requires distinguishing a cascade from a survivor. |
| `counter_argument.py` | Make a claim's *contrary* justified — which destroying its support does not achieve. Also provides the strict-permitted ablation, where a strict rule wins unconditionally and the question becomes whether the model finds the cheapest answer. |
| `preference_construction.py` | Move claims to required statuses using **only** preference directives. Isolates the preference machinery: no undercut, no new argument, only re-weighting. |
| `claim_chain.py` | Write all and only the directives forming the line that justifies a claim. Decoys reach the same claim but are defeated; the true line may itself be attacked yet reinstated, so neither "the unattacked line" nor "any path" works. |
| `defeat_diagnosis.py` | A claim is not justified — say why. Names every failure point, its defeater, the kind of attack, and why that defeater survives. The complement to `claim_chain`: extraction has a partner in diagnosis. |
| `formalization.py` | Translate a natural-language argumentation into ASPIC+ DSL. Scored on behaviour, per-directive structure, and the type decisions separately, because axiom-versus-premise is behaviourally invisible unless something undermines the literal. |

---

## `arggym/inspector.py` — reading items by hand

A Flask UI at `http://127.0.0.1:5000`. Select a task, level, ordering and seed; see the exact prompt a
model would receive, the reference answer, the verified minimum, the per-goal engine verdicts, and a
rendered view of the theory's structure with statuses inline.

