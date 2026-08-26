# Evidence

## Snapshot

```text
repository: https://github.com/ieddeveci/ArgGYM.git
branch: main
commit: 1fdc9db03ddcc415abd82bef0f230ab31b8a6892
commit date: 2026-08-21T18:46:15+03:00
subject: Update requirements.txt
```

The clone is at `workspace/arggym-v2-new-issue-audit-2026-08-26/repo` and remained on `main`. Runtime probes imported v2 from that clone with Python 3.11 and `python-argumentation==2.0.2`.

## Existing-issue check

The pre-publication GitHub ledger contained 19 issues. Titles, bodies, and repository-wide issue comments were searched for the candidate topics. No issue or comment covered the install failure, export omissions, missing grid cells, claim-chain order, fixed `lx_` noise names, generation time, or the status-definition conflict.

Two findings overlap existing scopes:

- Invalid keyword/arrow pairs extend #19's parser-validity problem.
- Ignored formalization preference operands instantiate #12's negative-control and semantically-inert-preference warning.

## Installation

From a new Python 3.13 virtual environment:

```text
$ uv pip install --python .venv/bin/python -r ArgGYM_v2/requirements.txt
× No solution found when resolving dependencies:
╰─▶ Because py-arg was not found in the package registry and you require
    py-arg==2.0.2, we can conclude that your requirements are unsatisfiable.
```

Installing the actual distribution succeeds:

```text
$ uv pip install --python .venv/bin/python python-argumentation==2.0.2
Resolved 2 packages
Installed 2 packages
 + parse==1.22.1
 + python-argumentation==2.0.2

{'module': 'py_arg', 'distribution': 'python-argumentation', 'version': '2.0.2'}
```

PyPI evidence:

- `https://pypi.org/pypi/python-argumentation/2.0.2/json` returns the 2.0.2 release and declares Python `>=3.9`.
- `https://pypi.org/pypi/py-arg/2.0.2/json` returns 404.

## Executable inventory

`ArgGYM_v2/inspector.py:33-58` lists twelve modes:

```text
semantics_query
status_query
formalization
defeat_diagnosis
claim_chain
preference_construction
counter_argument_strict
counter_argument
perturbation
attack
defence
attack_defense
```

`ArgGYM_v2/core/export.py:16-25` registers the first eight only. Direct CLI checks:

```text
$ python run.py export attack_defense /tmp/attack_defense.jsonl
unknown task attack_defense; known: [...]

$ python run.py export perturbation /tmp/perturbation.jsonl
unknown task perturbation; known: [...]
```

## Fast behavioral probes

Command:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  workspace/arggym-v2-new-issue-audit-2026-08-26/probes.py
```

Important results:

| Probe | Result |
|---|---|
| Reversed `claim_chain` reference | 22 lines, score 1.0, `correct_order=False` |
| Defeasible keyword with strict arrow | Canonical DSL: 0 operations, `defeasible_missing_arrow`; scorer: 1 operation |
| Wrong-arrow counter-argument reference | Score 1.0 |
| Reversed formalization preference | Score, behavior, shape, and type all 1.0 |
| `status_query` default grid | Missing `(12, last_link_elitist, 0)` and `(12, last_link_elitist, 1)` |
| Counter-argument noise names | `lx_901`–`lx_904` in all 8 sampled cells; no reference uses one |
| Undercut status | `qq0=OVERRULED`; neither `qq0` nor `-qq0` occurs in the grounded or preferred extension |

## Grid coverage

Each completed grid requested levels 3, 6, 9, 12, and 15; both orderings; and seeds 0 and 1, for 20 cells.

| Task | Generated | Missing | Invalid references |
|---|---:|---|---:|
| `claim_chain` | 20/20 | none | 0 |
| `formalization` | 20/20 | none | 0 |
| `defeat_diagnosis` | 20/20 | none | 0 |
| `status_query` | 18/20 | L12 / last-link / seeds 0 and 1 | 0 |
| `perturbation` | 19/20 | L15 / last-link / seed 0 | 0 |

The incomplete construction and multi-semantics grids are listed under report limits.

A real status export confirms the silent omission:

```text
wrote 18 items (18 valid) -> /tmp/.../status_query.jsonl
taskset_hash 17cb671432026520f51d49c66c3a0019
```

Its manifest reports `n_items: 18` and lists all requested levels and orderings, but contains no missing-cell list.

## Timing probes

These elapsed times are specific to this machine and were measured while another repository evaluation was active.

| Cell | Result |
|---|---|
| `semantics_query`, L12, seed 0, last-link | generated in 3.62 s |
| `semantics_query`, L15, seed 0, last-link | exceeded 30 s; exit 124 |
| `counter_argument`, L12, seed 0, last-link | 1.87 s |
| `counter_argument`, L12, seed 1, last-link | 2.52 s |
| `counter_argument`, L12, seed 0, weakest-link | 3.45 s |
| `counter_argument`, L12, seed 1, weakest-link | exceeded 30 s; exit 124 |
| `counter_argument`, L15, seed 0, last-link | 5.35 s |
| `counter_argument`, L15, seed 1, last-link | exceeded 30 s; exit 124 |
| `counter_argument`, L15, seed 0, weakest-link | 10.36 s |
| `counter_argument`, L15, seed 1, weakest-link | 15.41 s |

The interrupted semantics stack was inside `get_preferred_extensions`. The interrupted construction stack was inside `assert_irredundant` or post-minimality `holds` checks.

## Other checks

- Parsed all 28 v2 Python files with `ast.parse`: no syntax errors.
- Source inventory found no `tests/`, `validation/`, CI configuration, or Python project configuration in v2.
- `python run.py gates` raises `ModuleNotFoundError: No module named 'validation'`; `run.py --help` labels validation optional. This is not counted as new because #12 already tracks missing validation and negative controls.
- The expanded closure reproducibility probe generated 60 items from five task families in each of three processes. All payloads had SHA-256 `20a6b479570ee3e13b244d6ac4a262ecfd62e3d624fbb72db4cfcf0b958e7678` for `PYTHONHASHSEED=0`, `1`, and `12345`.
