# Closed issues

Date: 2026-08-26

Repository: `ieddeveci/ArgGYM`

GitHub account: `bdsaglam`

The following issues were open and classified as fully addressed by the ArgGYM v2 audit. Each received the corresponding reviewed note from `closures/` and was then closed:

| Issue | Resolution in v2 | Closing comment | Final state |
|---|---|---|---|
| #1 | Direct fix | https://github.com/ieddeveci/ArgGYM/issues/1#issuecomment-5428049814 | Closed |
| #2 | Affected `claim_identification` surface removed | https://github.com/ieddeveci/ArgGYM/issues/2#issuecomment-5428053428 | Closed |
| #3 | Direct fix | https://github.com/ieddeveci/ArgGYM/issues/3#issuecomment-5428054101 | Closed |
| #4 | Direct fix | https://github.com/ieddeveci/ArgGYM/issues/4#issuecomment-5428054564 | Closed |
| #5 | Affected robustness surface removed; intended behavior retained elsewhere | https://github.com/ieddeveci/ArgGYM/issues/5#issuecomment-5428055150 | Closed |
| #7 | Affected content mode removed | https://github.com/ieddeveci/ArgGYM/issues/7#issuecomment-5428055630 | Closed |
| #11 | Affected `claim_identification` task removed | https://github.com/ieddeveci/ArgGYM/issues/11#issuecomment-5428056208 | Closed |

Issues #16 and #17 were also classified as fully addressed, but they were already closed before this action and were not changed.

Verification after the changes:

- The seven target issues report `CLOSED` through `gh issue view`.
- Each latest issue comment matches its reviewed draft byte-for-byte after shell newline normalization.
- Partially addressed issues #6, #8, and #12 remain open.
- Unresolved issues #9, #10, #15, #18, #19, #20, and #21 remain open.
- Previously closed issues #16 and #17 remain closed.

## Fresh-main revalidation

After the closures, the seven decisions were checked again inside a fresh clone of owner `main` at `1fdc9db03ddcc415abd82bef0f230ab31b8a6892`, rather than in the `baris-benchmark` worktree.

- #1: a 60-item matrix from five task families produced the same SHA-256, `20a6b479570ee3e13b244d6ac4a262ecfd62e3d624fbb72db4cfcf0b958e7678`, in fresh processes with `PYTHONHASHSEED=0`, `1`, and `12345`. All nine task modules also derive seeds with BLAKE2b and local RNG instances.
- #2, #7, and #11: `claim_identification` and content mode remain absent from the v2 task and export inventories.
- #3: 100 formalization items across five levels, ten seeds, and both orderings contained no literal used as both an axiom and an ordinary premise.
- #4: a source-wide search found no reasoning tags or step-by-step instruction in v2.
- #5: sampled `claim_chain` items contain reinstatement towers at levels 9, 12, and 15; `status_query` enables attack towers from level 6 in the export grid.

The fresh clone could not install `ArgGYM_v2/requirements.txt` from the configured package index because `python-argumentation==2.0.2` was unavailable there. Behavioral checks therefore used the existing Python 3.11 environment with PyArg 2.0.2, while imports and source came from the fresh clone.

Outcome: all seven closure decisions hold on owner `main`; no issue was reopened.
