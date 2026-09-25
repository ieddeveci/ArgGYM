# How ArgGYM is built

A tour of the code for anyone changing the generator or the scorers. What the
package promises to its users is in `docs/dataset-contract.md`.

## `arggym/aspic/` -- the engine layer

A thin, typed wrapper over PyArg. Everything above this directory speaks in `Operation` records and
never touches PyArg directly, so the whole suite could be moved to another ASPIC+ implementation by
rewriting these files.

| file | responsibility |
|---|---|
| `aspic.py` | The theory model and the bridge to PyArg: `Theory`, its rules, facts, preferences and contrariness, the semantics and ordering names, and labellings. The only file that imports PyArg. |
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
| `scoring.py` | The directive scorer, shared by the six engine-checked tasks. Parses an answer, rejects illegal additions, checks every goal against the engine, computes efficiency against the verified minimum, and on a miss pays partial credit for how far the answer moved each goal from its status before the answer. It also emits `achieved_status` and `deadlock_not_defeat`, which separate "created a conflict and failed to win it" from "did nothing" -- two zeroes that mean opposite things. |
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
prompt, curriculum and gold construction. They share `core/` but not each other. What each task
asks is in `docs/dataset-card.md`.

| file | tasks |
|---|---|
| `status_query.py` | `status_query` |
| `semantics_query.py` | `semantics_query` |
| `attack_defense.py` | `attack`, `defence`, `attack_defense` |
| `perturbation.py` | `perturbation` |
| `counter_argument.py` | `counter_argument`, `counter_argument_strict` |
| `preference_construction.py` | `preference_construction` |
| `claim_chain.py` | `claim_chain` |
| `defeat_diagnosis.py` | `defeat_diagnosis` |
| `formalization.py` | `formalization` |

---

## `arggym/inspector.py` -- reading items by hand

A Flask UI at `http://127.0.0.1:5000`, started with `arggym inspect`. Select a task, level, ordering
and seed; see the exact prompt a model would receive, the reference answer, the verified minimum, the
per-goal engine verdicts, and a rendered view of the theory's structure with statuses inline.

---

## `arggym/cli.py` -- the `arggym` command

`arggym tasks` lists the registered tasks, `arggym freeze` builds a taskset from a spec,
`arggym floors` measures the chance floors of a frozen taskset, and `arggym inspect` starts the
inspector. `arggym --help` lists every command.

---

## Outside the package

| path | what it holds |
|---|---|
| `data/` | the two taskset specs and the two frozen tasksets built from them |
| `evals/` | the evaluation harness: `run.py` generates, `score.py` scores offline, `report.py` and `figures.py` read the scores. Not part of the wheel; `docs/evaluation.md` covers it. |
| `evals/conf/` | the Hydra configs: one file per model under `model/`, the answer templates and the elicitations |
| `hpc/` | the vLLM serving profiles and the TRUBA cluster scripts that serve the `hf-*` configs (`hpc/README.md`) |
| `scripts/` | `evaluate.py`, a reference evaluator that uses the standard library and the public API only, and `results.sh`, which packs and unpacks `results/` |
| `results/` | finished eval runs, with generations in Git LFS (`results/README.md`) |
| `tests/` | the test suite; `tests/e2e/` holds the slow full-grid cells, run with `make test-all` |
