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
`.hydra/` are local. Packing edits only paths: `taskset` in `run.json` and
`metrics.json` becomes `data/<name>.jsonl`, the copy committed in this repo,
and `run_dir` in `metrics.json` becomes `outputs/runs/<run_id>`. The
`taskset_hash` beside them is kept as recorded.

## Setup

Install `zstd` and `git-lfs`, then run `git lfs install` once per machine.

## Pull only what you need

A plain `git pull` downloads the generations of every run. To get the scores
alone, skip the LFS download, and the `.zst` files stay small pointer files:

```
GIT_LFS_SKIP_SMUDGE=1 git pull
```

To keep every later pull pointer-only too, set this once per clone:

```
git config lfs.fetchexclude "results/**"
```

Then fetch the generations of just the runs you want to rescore:

```
git lfs pull --include="results/<run_id>/*"
make results-unpack RUN=<run_id>
```

`make results-unpack RUN=<run_id>` fetches a pointer by itself as well, one
file at a time. `make results-unpack` with no `RUN` unpacks every run in
`results/`, so it downloads all of their generations.

## Share a run

With `git-lfs` installed and `git lfs install` run once, score the run, pack
it, and commit the packed directory:

```
uv run python -m evals.score outputs/runs/<run_id>
make results-add RUN=<run_id>
git add results/<run_id> && git commit -m "Add <run_id> to results" && git push
```

`make results-add RUN="<run_id> <run_id> ..."` packs several runs. It checks
every run before it packs any, and refuses:

- when Git LFS is not set up, because git would then commit the `.zst` as a
  plain file;
- a run whose `run.json` does not say `"status": "completed"`, or that has no
  `metrics.json`;
- a run whose taskset is not directly under a `data/` folder, or whose
  `data/<name>.jsonl` is missing from this repo or has a different
  `taskset_hash` than the run;
- a run already in `results/`, unless you pass `FORCE=1`.

## Unpack and rescore

```
make results-unpack RUN=<run_id>
uv run python -m evals.score outputs/runs/<run_id>
uv run python -m evals.report outputs/runs/* -o outputs/reports/latest
```

Unpacking writes `generations.jsonl`, `run.json` and `metrics.json` into
`outputs/runs/<run_id>/`. It refuses to overwrite an existing
`generations.jsonl` unless you pass `FORCE=1`. If the `.zst` file is still an
LFS pointer, it runs `git lfs pull --include` for that one file first. A `.zst`
that does not decompress leaves `outputs/runs/<run_id>/` untouched.

## LFS quota

GitHub Free gives 10 GiB of LFS storage and 10 GiB of bandwidth a month,
counted against the repo owner. Every clone that pulls these files uses
bandwidth, and a replaced file keeps using storage. Commit only the compressed
generations. Never commit uncompressed `generations.jsonl`, `prompts.jsonl` or
`samples.jsonl`.
