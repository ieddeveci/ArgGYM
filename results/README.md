# Shared eval runs

Finished runs that teammates share through the repo. Each one sits in
`results/<run_id>/`, where `run_id` is its directory name under `outputs/runs/`:

```
run.json               the manifest, as a plain file
metrics.json           the scores, as a plain file, so they show in diffs
generations.jsonl.zst  the model outputs, zstd -19, stored in Git LFS
```

Nothing else is kept. `samples.jsonl` is rebuilt by `evals.score`,
`prompts.jsonl` by the taskset and template, and `run.lock`, `run.log` and
`.hydra/` are local. The one edit made while packing is the `taskset` path in
`run.json`: it becomes `data/<name>.jsonl`, the copy committed in this repo, and
the `taskset_hash` beside it is kept as recorded.

## Setup

Install `zstd` and `git-lfs`, then run `git lfs install` once per machine.

## Add a run

```
make results-add RUN=<run_id>
make results-add RUN="<run_id> <run_id> ..."
```

This packs `outputs/runs/<run_id>`. It refuses a run whose `run.json` does not
say `"status": "completed"`, and a run already in `results/` unless you pass
`FORCE=1`. Score the run before packing it, because `metrics.json` is required.

## Unpack and rescore

```
make results-unpack                  # every run in results/
make results-unpack RUN=<run_id>
uv run python -m evals.score outputs/runs/<run_id>
uv run python -m evals.report outputs/runs/* -o outputs/reports/latest
```

Unpacking writes `generations.jsonl`, `run.json` and `metrics.json` into
`outputs/runs/<run_id>/`. It refuses to overwrite an existing
`generations.jsonl` unless you pass `FORCE=1`. If the `.zst` file is still an
LFS pointer, it runs `git lfs pull` for that file first.

## LFS quota

GitHub Free gives 10 GiB of LFS storage and 10 GiB of bandwidth a month,
counted against the repo owner. Every clone that pulls these files uses
bandwidth, and a replaced file keeps using storage. Commit only the compressed
generations. Never commit uncompressed `generations.jsonl`, `prompts.jsonl` or
`samples.jsonl`.
