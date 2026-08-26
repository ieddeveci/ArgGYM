# Audit evidence

## Snapshot and issue ledger

- Owner code: `origin/main` at
  `1fdc9db03ddcc415abd82bef0f230ab31b8a6892`, fetched on 2026-08-26.
- Issue ledger: `gh issue list --repo ieddeveci/ArgGYM --state all --limit 200`.
  Result: 19 issues, 17 open and 2 closed (#16 and #17).
- Runtime: Python 3.11 with Python Argumentation/PyArg 2.0.2. The owner pin is
  `ArgGYM_v2/requirements.txt:1`.
- All behavioral probes ran from a clean archive of the owner commit, with its
  `ArgGYM_v2` directory first on `sys.path`.

## Static checks

| Check | Result |
|---|---|
| `rg '^def stable_seed' ArgGYM_v2/tasks` | All nine task modules define a stable seed; each implementation uses BLAKE2b. |
| `rg '\[reasoning\]|reason step|chain.of.thought|CoT' ArgGYM_v2` | No matches. |
| `rg 'with_content|content mode' ArgGYM_v2` | No matches. |
| Owner-tree task inventory | Nine task modules; no `claim_identification.py` or `robustness.py`. |
| Owner-tree report/test inventory | No v2 reporting module, `tests/`, or `validation/` directory. `run.py:23-26` treats validation as optional and absent. |
| `core/export.py:_EXPORTABLE` | Eight export keys. `attack_defense` and `perturbation` are absent. |

## Behavioral probes

All task probes used `last_link_elitist` and seed 0 unless stated otherwise.

| Issue | Probe | Observed result |
|---|---|---|
| #1 | Generate `(prompt, reference, metadata, theory)` for 60 items: five task families (`preference_construction`, `counter_argument`, `claim_chain`, `formalization`, `status_query`) × levels 3, 6, 9 × seeds 0, 1 × two orderings. Repeat in fresh processes with `PYTHONHASHSEED=0`, `1`, and `12345`. | All 180 generations succeeded. Each 60-row payload had SHA-256 `427a9c45e74a681f732a4345ad3fb837642ae67305f377a9039fafc53b8bf3ca`. |
| #3 | Generate formalization items at levels 1, 3, 5, 10, 15 × seeds 0-9 × two orderings. Intersect exact axiom and ordinary-premise literals in each item. | 100 items generated; zero intersections. |
| #8 | Reverse the four reference directives of a level-10 `counter_argument` item. Separately append `[prefer_rule: missing1 > missing2]` to its correct reference. | Reversed answer: 1.0. Correct answer plus unknown preference: 0.0, `engine_rejected:ValueError`. |
| #16 | Change one of eight labels in a correct level-3 `status_query` answer to another valid status. | Precision = recall = F1 = 0.875. The wrong prediction is in the denominator. |
| #17 | Append `extra: nan` to the same correct eight-label answer. Separately append `extra: justified`. | Invalid label: 0.0, `unparseable_tokens:2`. Valid ninth prediction: precision 8/9, recall 1, F1 0.9412. |
| #18 | Compare level-14 `counter_argument` prompts for `allow_strict=True` and `False` at seed 0; inspect the instruction suffixes for attack and counter-argument items. | Strict and non-strict counter-argument instructions are identical. No inspected instruction suffix mentions strict rules. |
| #19 | Rewrite all three rules in a correct level-3 attack answer first without names (`[defeasible: ...]`), then without keyword/name whitespace (`[defeasiblek1: ...]`). | Reference, omitted-name, and joined-name answers all parsed three rules and scored 1.0. Missing names became `m1`, `m2`, `m3`. |
| #20 | Parse `[defeasible _bad: aa0 => bb0]`, `[defeasible good_name: aa0 => bb0]`, and `[premise: _bad]`. | The underscore-leading rule and premise names were unparseable; `good_name` parsed. |
| #21 | Append one illegal strict rule, axiom, or positive premise to a correct three-directive attack answer. Repeat against a correct two-directive `preference_construction` answer. | All six modified answers remained 1.0. Diagnostics reported the additions as illegal; `n_used` stayed at the reference length. |
