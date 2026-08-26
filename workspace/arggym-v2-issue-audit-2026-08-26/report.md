# ArgGYM v2 GitHub issue audit

- Audit date: 2026-08-26
- Owner snapshot: `origin/main` at `1fdc9db03ddcc415abd82bef0f230ab31b8a6892`
- GitHub ledger at audit time: 19 issues, 17 open and 2 closed

## Verdict

The GitHub state understates what v2 changed. Nine issues are addressed in v2,
including seven that remain open on GitHub. Three are only partially addressed,
and seven remain unresolved.

| Audit verdict | Count | Issues |
|---|---:|---|
| Addressed in v2 | 9 | #1, #2, #3, #4, #5, #7, #11, #16, #17 |
| Partially addressed | 3 | #6, #8, #12 |
| Still open | 7 | #9, #10, #15, #18, #19, #20, #21 |

“Addressed” includes a redesign that removes the affected task or mode when the
reported failure can no longer occur. It does not mean that the original v1
implementation was repaired in place.

The exact probe matrix and outputs are in [evidence.md](evidence.md).

## Issue ledger

| Issue | GitHub | Audit | Evidence from owner v2 |
|---|---|---|---|
| [#1 Generation is not reproducible from its seed](https://github.com/ieddeveci/ArgGYM/issues/1) | Open | **Addressed** | All nine task modules derive seeds with BLAKE2b and local `random.Random` instances (`ArgGYM_v2/tasks/attack_defense.py:25-33`; static inventory in `evidence.md`). A cross-process probe generated 60 items from five task families with `PYTHONHASHSEED=0`, `1`, and `12345`; all three runs produced SHA-256 `427a9c45e74a681f732a4345ad3fb837642ae67305f377a9039fafc53b8bf3ca`. The export manifest records Python and `PYTHONHASHSEED` (`ArgGYM_v2/core/export.py:97-106`). |
| [#2 `claim_identification` exposes raw rule symbols](https://github.com/ieddeveci/ArgGYM/issues/2) | Open | **Addressed by redesign** | v2 removes `claim_identification`; it is absent from the nine-module task inventory and export registry (`ArgGYM_v2/README.md:57-71`; `ArgGYM_v2/core/export.py:16-25`; inventory in `evidence.md`). The raw-rule candidate path is absent from the owner v2 source. |
| [#3 Formalization reuses one literal as axiom and premise](https://github.com/ieddeveci/ArgGYM/issues/3) | Open | **Addressed** | The formalization generator draws fresh atom names from one iterator (`ArgGYM_v2/tasks/formalization.py:128-153`) rather than calling a helper that restarts at `a0`. A probe over 100 total items, covering five levels, ten seeds, and both orderings, found no exact axiom/premise overlap. |
| [#4 Prompts hard-code an unused CoT recipe](https://github.com/ieddeveci/ArgGYM/issues/4) | Open | **Addressed** | A source-wide search found no `[reasoning]`, `[/reasoning]`, or step-by-step instruction. Shared construction prompts request only directives inside an answer region (`ArgGYM_v2/core/prompting.py:20-53`; search result in `evidence.md`). |
| [#5 `robustness/reinstate` is never sampled](https://github.com/ieddeveci/ArgGYM/issues/5) | Open | **Addressed by redesign** | v2 removes `robustness`, but samples the missing reinstatement phenomenon directly: `status_query` adds attack towers from level 5 (`ArgGYM_v2/tasks/status_query.py:82-100,154-173`), and `claim_chain` attacks then reinstates the true line from level 8 (`ArgGYM_v2/tasks/claim_chain.py:79-97,161-162`). |
| [#6 Tied preferences occur undeclared](https://github.com/ieddeveci/ArgGYM/issues/6) | Open | **Partial** | The notation reference now states that opposite preferences coexist and produce a tie (`ArgGYM_v2/NOTATION.md:67-78`). Item prompts do not carry that explanation because shared notation is disabled by default (`ArgGYM_v2/core/prompting.py:22,50-52`). The repository documents the convention, but the benchmark input does not state it. |
| [#7 Content mode is closed-world while prompts invite open-world answers](https://github.com/ieddeveci/ArgGYM/issues/7) | Open | **Addressed for v2's scope** | A source-wide search and the task/export inventories show no content-mode generator or `with_content` path (`ArgGYM_v2/README.md:57-71`; `ArgGYM_v2/core/export.py:16-25`; `evidence.md`). The v1 content-glossary mismatch has no corresponding v2 input path. |
| [#8 One malformed preference zeroes a correct answer](https://github.com/ieddeveci/ArgGYM/issues/8) | Open | **Partial** | v2 applies rules before preferences, so directive order no longer matters (`ArgGYM_v2/core/scoring.py:106-109`); reversing a four-directive counter-argument reference still scored 1.0. The other half remains: an unknown preference raises in the engine and the scorer returns 0 (`ArgGYM_v2/core/scoring.py:137-141`; `ArgGYM_v2/aspic/engine.py:123-140`). A correct answer plus `[prefer_rule: missing1 > missing2]` scored 0.0. |
| [#9 Report chance floors; do not rank by a raw overall mean](https://github.com/ieddeveci/ArgGYM/issues/9) | Open | **Still open** | The complete v2 source inventory has no result-reporting or chance-correction module (`ArgGYM_v2/README.md:16-79`; `evidence.md`). `core/export.py` writes item rows and a manifest only (`ArgGYM_v2/core/export.py:82-112`). The measured “floors” in `core/curriculum.py` are minimum-search difficulty settings, not chance baselines. |
| [#10 Decouple scoring from `[answer]` transport](https://github.com/ieddeveci/ArgGYM/issues/10) | Open | **Still open** | The shared scorer still accepts only text, extracts an answer region, parses it, and scores it in one function (`ArgGYM_v2/core/scoring.py:28-70,112-141`). Task-specific scorers repeat the same shape, for example `status_query.score(answer_text, item)` (`ArgGYM_v2/tasks/status_query.py:298-335`). There is no parsed-value scoring API. |
| [#11 `claim_identification` coarsens `status_query`](https://github.com/ieddeveci/ArgGYM/issues/11) | Open | **Addressed by redesign** | v2 drops `claim_identification` and keeps the three-way `status_query` (`ArgGYM_v2/README.md:62-71`; `ArgGYM_v2/core/export.py:16-25`). The redundant task no longer occupies a benchmark slot. |
| [#12 Tracking issue for the gold-test-set freeze](https://github.com/ieddeveci/ArgGYM/issues/12) | Open | **Partial** | The tracker requires both the linked fixes and the listed process safeguards. v2 addresses #1-#5, #7, and #11, and its taskset hash now covers prompt, reference, and metadata (`ArgGYM_v2/core/export.py:97-105`). The owner tree still has no negative-control tests, validation directory, or prompt/reference validation; the manifest does not record the ArgGYM commit or PyArg version (`ArgGYM_v2/core/export.py:101-107`; `ArgGYM_v2/run.py:23-26`; inventory in `evidence.md`). #6 and #8 remain partial, while #9 and #10 remain unresolved. |
| [#15 Paired symbolic/content instances and repeated content renderings](https://github.com/ieddeveci/ArgGYM/issues/15) | Open | **Still open** | The source and task/export inventories contain neither content rendering nor a paired item schema (`ArgGYM_v2/README.md:57-71`; `ArgGYM_v2/core/export.py:16-25`; `evidence.md`). Removing content mode avoids the confounded comparison but does not implement the requested experiment. |
| [#16 `status_query` omits wrong predictions from precision](https://github.com/ieddeveci/ArgGYM/issues/16) | Closed | **Addressed** | Precision now divides true-positive pairs by every parsed predicted pair (`ArgGYM_v2/tasks/status_query.py:323-328`). In a probe with eight predictions and one wrong in-range label, precision, recall, and F1 were all 0.875. |
| [#17 `status_query` ignores invalid labels such as `nan`](https://github.com/ieddeveci/ArgGYM/issues/17) | Closed | **Addressed, with a stricter policy** | The parser removes valid pairs and rejects any remaining non-format tokens (`ArgGYM_v2/tasks/status_query.py:298-318`). A correct eight-label answer plus `extra: nan` scored 0.0 rather than escaping with 1.0. Replacing it with a valid ninth prediction gave precision 8/9, recall 1, and F1 0.9412. |
| [#18 State strict-rule constraints in each task prompt](https://github.com/ieddeveci/ArgGYM/issues/18) | Open | **Still open** | `counter_argument._render_prompt` accepts `allow_strict` but never uses it (`ArgGYM_v2/tasks/counter_argument.py:338-347`). Strict-permitted and strict-forbidden items generated with the same seed have identical prompt instructions. Shared attack/defence prompts also omit the constraint because the notation block is disabled (`ArgGYM_v2/core/prompting.py:22,34-52`). `preference_construction` is the exception: it explicitly says preferences only (`ArgGYM_v2/tasks/preference_construction.py:284-299`). |
| [#19 Missing or concatenated rule names are not penalized](https://github.com/ieddeveci/ArgGYM/issues/19) | Open | **Still open** | The shared rule regex makes the name optional and requires no boundary between the keyword and name (`ArgGYM_v2/core/scoring.py:15`). Rewriting every rule in an attack reference from `[defeasible k1: ...]` to either `[defeasible: ...]` or `[defeasiblek1: ...]` left the score at 1.0. |
| [#20 Specify naming rules](https://github.com/ieddeveci/ArgGYM/issues/20) | Open | **Still open** | Identifier grammar exists only in parser regexes: names must start with an ASCII letter and then use word characters (`ArgGYM_v2/core/scoring.py:14-16`). The notation reference says only that rule names and literals are distinct (`ArgGYM_v2/NOTATION.md:22-26`), and task prompts give no naming grammar. `_bad` is therefore rejected by an unstated rule. |
| [#21 Illegal strict/axiom/premise additions do not affect score](https://github.com/ieddeveci/ArgGYM/issues/21) | Open | **Still open** | Legality checking drops invalid operations (`ArgGYM_v2/core/scoring.py:73-103`), then efficiency and bloat use only `len(kept)` (`ArgGYM_v2/core/scoring.py:131-150,193-199`). Adding one illegal strict rule, axiom, or premise to a perfect attack answer left it at 1.0; the same held for all three additions in `preference_construction`. Diagnostics list the additions as illegal, but the score ignores them. |

## What should happen to the GitHub issues

- Close #1, #2, #3, #4, #5, #7, and #11 with a short note that v2 addresses
  them. For #2, #5, #7, and #11, state that the affected task or mode was
  removed or replaced rather than repaired.
- Keep #6, #8, #9, #10, #12, #15, and #18-#21 open.
- Leave #16 and #17 closed.

## Unfiled finding

The owner snapshot says that `python run.py export-all` exports every task
(`ArgGYM_v2/README.md:8-12`), but the export registry omits both
`attack_defense` and `perturbation` (`ArgGYM_v2/core/export.py:16-25`). This is
separate from the 19 existing issues and warrants its own issue.

## Limits of this audit

- The verdicts apply only to owner commit `1fdc9db`.
- The cross-process reproducibility probe covered 60 items from
  `preference_construction`, `counter_argument`, `claim_chain`, `formalization`,
  and `status_query`. It did not cover `attack_defense`, `perturbation`,
  `defeat_diagnosis`, or `semantics_query`.
- Parser and scorer probes used the owner-pinned PyArg 2.0.2.
- I did not run model inference, rebuild a complete frozen v2 taskset, or inspect
  screenshots attached to issues.
