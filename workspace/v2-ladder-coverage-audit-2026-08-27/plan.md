# V2 ladder and coverage audit plan

## Objective

Check every claim in `workspace/benchmark/issues/v2-ladder-and-coverage.md`
against the current repository owner's `main`, the frozen tasksets, the stored
evaluation results, and the live GitHub issue ledger. Determine which findings
remain valid and whether they overlap existing issues. Do not publish anything.

## Fixed scope

- Owner repository: `ieddeveci/ArgGYM`
- Owner-main snapshot: `1fdc9db03ddcc415abd82bef0f230ab31b8a6892`
- Historical fix snapshot: `df453dc04e9b510ae3595132e457c5f13d858842`
- Pilot taskset: `v2-pilot-20260804T220828Z-e89f0b8a`
- Fixed taskset: `v2-fixed-20260817T143923Z-ea3fd430`
- Issue ledger checked through issue 29 on 2026-08-27

## Completed checks

1. Pin a fresh clone to the live owner-main commit.
2. Trace A1-A5 and B1-B2 to source and historical fix commits.
3. Recompute taskset metadata profiles and run focused current-main generators.
4. Run the historical ladder regression suite at the exact fix commit.
5. Reproduce the sampling-payload and high-error-gate changes.
6. Re-run the four-arm Qwen sampler comparison.
7. Search all live issue titles, bodies, and comments for exact matches, then
   inspect broad matches manually.
8. Classify legitimacy, current-main status, overlap, and publication action.

## Completion criterion

The report gives one evidence-backed disposition for every A, B, and C claim,
identifies the correct number of current-main issue candidates, records all
claim corrections, and makes no external changes.
