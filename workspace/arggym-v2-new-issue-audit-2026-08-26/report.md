# ArgGYM v2 net-new benchmark audit

- Audit date: 2026-08-26
- Owner snapshot: fresh clone of `main` at `1fdc9db03ddcc415abd82bef0f230ab31b8a6892`
- Scope: generation, export, prompt/scorer agreement, leakage, reproducibility, and benchmark validity
- GitHub action: none; no finding below was posted

## Outcome

I found eight standalone issue candidates that are not covered by the current GitHub ledger. Five affect benchmark installation, composition, or scoring. Three concern leakage, specification, or generation cost. I also found two concrete v2 failures that belong on existing issues #19 and #12 instead of becoming new issues.

| ID | Severity | Finding | Disposition |
|---|---|---|---|
| N1 | High | A clean v2 install is impossible from its requirements file | New issue |
| N2 | High | `export-all` omits four of the twelve modes exposed by the inspector | New issue; carried forward from the earlier audit's unfiled finding |
| N3 | High | Export silently drops failed grid cells and produces an unbalanced taskset | New issue |
| N4 | High | `claim_chain` gives full credit when the required line is reversed | New issue |
| N5 | Medium | `counter_argument` rule names identify every injected noise component | New issue |
| N6 | Medium | `semantics_query` repeatedly recomputes extension sets and can exceed 30 seconds for one official cell | New issue |
| N7 | Medium | Construction tasks run a second uncapped exponential subset search | New issue |
| N8 | Medium | The documented meaning of `overruled` contradicts the engine on undercuts | New issue |
| E1 | High | Scorers accept a rule keyword paired with the wrong arrow | Add to #19 |
| E2 | High | `formalization` gives full credit after reversing a gold preference | Add to #12 |

The fast behavioral probes are reproducible with [probes.py](probes.py). Raw commands, outputs, and limits are in [evidence.md](evidence.md).

## Standalone issue candidates

### N1 — A clean v2 install is impossible

`ArgGYM_v2/requirements.txt:1` requires `py_arg==2.0.2`. No distribution with that name and version exists on PyPI, so `uv pip install -r ArgGYM_v2/requirements.txt` fails before the benchmark can run. The package that provides the `py_arg` module is [`python-argumentation==2.0.2`](https://pypi.org/project/python-argumentation/2.0.2/); installing that exact distribution succeeds and imports `py_arg`.

This is not a Python-version compatibility problem. PyPI declares Python 3.9 or newer, and the corrected requirement installed in the fresh clone's Python 3.13 environment.

Suggested fix: restore `python-argumentation==2.0.2`, then verify installation in a clean CI environment.

### N2 — `export-all` exports eight of twelve modes

The inspector is the fullest executable inventory and lists twelve modes (`ArgGYM_v2/inspector.py:33-58`). The export registry lists eight (`ArgGYM_v2/core/export.py:16-25`). It omits:

- `attack`
- `defence`
- `attack_defense`
- `perturbation`

Both `python run.py export attack_defense ...` and `python run.py export perturbation ...` fail as unknown tasks. This contradicts `python run.py export-all # generate every task as JSONL` (`ArgGYM_v2/README.md:8-11`). The nearby claim of “Ten tasks, eleven modes” also disagrees with the twelve-mode inspector, and `semantics_query` is absent from the README task table (`ArgGYM_v2/README.md:57-71`).

The result is not merely a documentation defect: the public export path cannot build one third of the benchmark modes.

Suggested fix: make one authoritative registry drive the CLI, inspector, README count, and exporter. Add explicit export keys for all three attack/defence modes and perturbation.

### N3 — Export silently drops failed cells

`export_task` skips a cell whenever a generator returns `None` (`ArgGYM_v2/core/export.py:86-96`). It neither fails nor records which tuple was lost. On the default grid, both `status_query` cells at level 12 with last-link ordering and seeds 0 and 1 exhaust all 24 generation attempts (`ArgGYM_v2/tasks/status_query.py:338-344`). The advertised 20-row grid therefore exports 18 rows:

```text
wrote 18 items (18 valid) -> .../status_query.jsonl
```

The manifest still lists all five levels and both orderings, with no missing-cell field. A separate manual grid for the currently unexportable `perturbation` task also lost level 15 / last-link / seed 0.

This changes task and level weights in any aggregate score and makes absence look like successful validation.

Suggested fix: require every requested `(level, ordering, seed)` cell by default. Fail export with the missing tuples, or require an explicit `--allow-missing` mode that records them in the manifest.

### N4 — `claim_chain` ignores the required order

The prompt asks for the argumentation line “in order from the premise to the claim” (`ArgGYM_v2/tasks/claim_chain.py:305-317`). The scorer converts gold and prediction to sets before computing the score (`ArgGYM_v2/tasks/claim_chain.py:342-349`). It calculates `correct_order`, but that value is diagnostic only (`ArgGYM_v2/tasks/claim_chain.py:367-371`).

On the official level-9 / seed-0 / last-link item, reversing all 22 reference directives produces:

```text
score=1.0, correct_order=False
```

Ordering is part of the task's stated output and of its intended “trace the line” skill, so a set metric removes part of the construct.

Suggested fix: include order in the score. An exact sequence metric or normalized edit/LCS score would preserve partial credit without accepting a reversal as perfect.

### N5 — `counter_argument` names reveal every noise rule

The task adds a disconnected language-enrichment component with the fixed prefix `lx` (`ArgGYM_v2/tasks/counter_argument.py:173-176`). Unlike the other tasks that use this helper, it imports but never calls `randomize_rule_names` (`ArgGYM_v2/tasks/counter_argument.py:12-13`).

Across all eight sampled cells at levels 3 and 6, both orderings, and seeds 0 and 1, the filler rules were always exactly `lx_901` through `lx_904`; none occurred in the reference. Core rules retained `d...` names. A model can therefore discard every injected noise component from its name without testing relevance in the argument graph.

Suggested fix: call `randomize_rule_names` after appending enrichment, then remap every stored rule reference used to build candidates and gold.

### N6 — `semantics_query` recomputes expensive extensions for every pair

`status_under` constructs a fresh verifier on every call and enumerates preferred or stable extensions (`ArgGYM_v2/tasks/semantics_query.py:74-110`). `build` calls it once per candidate-and-semantics pair (`ArgGYM_v2/tasks/semantics_query.py:297-303`) and may repeat the entire build 24 times (`ArgGYM_v2/tasks/semantics_query.py:416-422`).

In this audit environment, level 12 / seed 0 / last-link completed in 3.62 seconds, while level 15 / seed 0 / last-link did not finish under a 30-second cap. The interrupted full grid was inside PyArg's preferred-extension recursion. Absolute timings depend on the machine, but the repeated computation is source-level and avoidable.

Suggested fix: construct one verifier per theory and cache each semantics' extension conclusions once. Apply every candidate query to those cached sets.

### N7 — Construction performs a second uncapped subset search

`minimal_subset_exact` has a call budget (`ArgGYM_v2/core/invariants.py:215-255`). Immediately afterward, `counter_argument` calls `assert_irredundant`, which enumerates every proper subset with no budget (`ArgGYM_v2/tasks/counter_argument.py:286-294`; `ArgGYM_v2/core/invariants.py:38-44`). `preference_construction` also performs additional singleton and pair searches after its bounded exact search (`ArgGYM_v2/tasks/preference_construction.py:236-261`).

One official `counter_argument` cell, level 12 / seed 1 / weakest-link, exceeded a 30-second cap while the other three level-12 cells took 1.87–3.45 seconds. Level 15 / seed 1 / last-link also exceeded 30 seconds. A 20-cell `preference_construction` batch was stopped after two minutes inside the post-search checks.

Suggested fix: make the bounded search return the proof needed by callers and remove the duplicate exhaustive pass. Carry one explicit call/time budget through every fallback and record exhaustion in item metadata.

### N8 — `overruled` has two incompatible definitions

The notation says that a claim is overruled when its contrary is in the extension, and undecided when neither is (`ArgGYM_v2/NOTATION.md:102-106`). The engine instead labels a conclusion overruled whenever all of its arguments are OUT (`ArgGYM_v2/aspic/engine.py:273-287`). An undercut can make that happen without deriving the contrary.

A four-directive probe builds `aa0 => qq0` and a justified undercut of that rule. The engine reports `qq0` as `OVERRULED`; the grounded extension and the only preferred extension contain neither `qq0` nor `-qq0`. Under the written definition, this is undecided.

This matters because item prompts list the labels without defining them. The repository notation is the only explicit contract a benchmark user can consult, and following it can produce a different answer from the scorer.

Suggested fix: choose one definition. If `overruled` means every argument for the claim is defeated, say that and explain that undercuts and undermines need not establish the claim's contrary. Then define the extension-based mapping for preferred, stable, and eager semantics explicitly.

## Additions to existing issues

### E1 — Extend #19: scorer accepts contradictory keyword/arrow pairs

Both scoring parsers match either `=>` or `->` but discard which arrow matched (`ArgGYM_v2/core/scoring.py:15,49-59`; `ArgGYM_v2/tasks/formalization.py:359,381-385`). The canonical DSL parser correctly rejects `[defeasible w: aa0 -> bb0]` as `defeasible_missing_arrow`, while the shared scorer parses it as a defeasible rule.

Changing one gold counter-argument line from `[defeasible w: br0 => -er3]` to `[defeasible w: br0 -> -er3]` leaves the score at 1.0. The same mutation receives 1.0 in `formalization`.

This is the same parser-contract class as #19, so it should be added there rather than filed separately.

### E2 — Extend #12: preference operands disappear from formalization shape scoring

The formalization structural key retains premise contents and rule structure but reduces every preference to `(kind,)` (`ArgGYM_v2/tasks/formalization.py:425-430`). The scorer therefore treats all preferences of one kind as identical unless the changed preference happens to alter a queried status.

For the official level-15 / seed-0 / weakest-link item, replacing gold

```text
[prefer_rule: q_14 > q_15]
```

with its reverse

```text
[prefer_rule: q_15 > q_14]
```

still produces `score=1.0`, `behavioural=1.0`, `shape_f1=1.0`, and `type_score=1.0`.

Issue #12 already records the absence of negative controls and warns that the old formalization metric was blind to semantically inert preferences. This is the v2 reproducer for that open tracking item.

## Checks that passed

- All 28 v2 Python files parse successfully.
- The `claim_chain`, `formalization`, and `defeat_diagnosis` default grids each generated 20/20 items, and every reference scored 1.0.
- The manual `perturbation` grid generated 19/20 items, and every generated reference scored 1.0.
- The fresh-main closure revalidation found no reason to reopen #1, #2, #3, #4, #5, #7, or #11; that evidence is in the earlier audit workspace.

## Limits

- I did not run model inference or estimate how much each scorer failure changes a published model ranking.
- The repository contains no v2 tests or installed `validation/` package, so there was no project test suite to run. This is already covered by #12.
- I stopped the full `export-all` sweep because high-level extension and minimality searches were consuming minutes. Candidate timings were measured while another repository evaluation process was active, so treat the exact seconds as environment-specific; the timeouts and repeated-work code paths are the evidence.
- I did not complete all 20 cells for `counter_argument`, `counter_argument_strict`, `preference_construction`, or `semantics_query`.
- I did not change source code in the clone or post to GitHub.
