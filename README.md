# ArgGYM

A generator and grader for defeasible-argumentation tasks in ASPIC+. Every item is built by
construction and checked against the engine, so the gold answer is never asserted, it is verified.

Ten tasks, eleven modes, fifteen levels, two strength orderings.

```
uv sync                           # install the package and its dependencies
uv run arggym export-all          # generate every task as JSONL into data/
uv run arggym export <task>       # generate one
uv run arggym tasks               # list the exportable task names
uv run arggym inspect             # read items in the browser
```

`make setup`, `make test`, `make export` and `make inspect` wrap the common ones; `make` lists them.

Without uv, `pip install -e .` then `arggym export-all`. The package also runs uninstalled from a
checkout as `python -m arggym`, given `python-argumentation==2.0.2` and `flask`.

---

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
| `prompting.py` | Prompt rendering for `attack_defense`, including the notation block that lists the permitted directive forms identically on every item so that its content leaks nothing about the theory. |
| `nlforms.py` | Eighty-nine natural-language surface forms for `formalization`, taken from the ASPIC+ and argumentation-schemes literature. The giveaway words are deliberately shared between the constructs they used to separate. |
| `export.py` | JSONL export with a manifest carrying `taskset_hash`, item counts, orderings, Python version and the minimality caveat. The hash covers prompt, reference and metadata together, so an edit that changes gold produces a new hash rather than silently overwriting a frozen taskset. |

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

