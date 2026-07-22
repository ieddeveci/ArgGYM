# ArgGYM tasksets

A **taskset** is a frozen set of evaluation items drawn from the ArgGYM
generator. ArgGYM is procedural — it can produce unlimited tasks — so comparing
models requires pinning a fixed sample and shipping it, which is what lives here.

## What is in a taskset

```
pilot-20260722T114138Z-247eca21/
  taskset.jsonl.gz   # the items, one JSON object per line (gzipped, ~19x smaller)
  config.yaml        # the grid it was generated from
  manifest.json      # provenance: build time, commit, hashes, per-cell counts, checks
```

Each row:

| Field | Meaning |
|---|---|
| `sample_id` | `<task>__<mode>__L<level>__<idx>`, e.g. `attack__content__L05__003` |
| `task`, `mode`, `level`, `idx` | grid coordinates (`mode` is `symbolic` or `content`) |
| `cell_seed` | RNG seed for this (task, mode, level) cell |
| `prompt` | the complete prompt to send a model — already includes the formalism intro, the theory, and the answer-format block |
| `entry` | the ArgGYM entry: `question`, `answer` (gold), and `metadata` (ops, ordering, kind, task-specific fields) |

The directory name is `<name>-<UTC-stamp>-<hash8>`, e.g.
`pilot-20260722T114138Z-247eca21`. The stamp sorts builds chronologically (newest
last), so `ls` shows the build order; the trailing hash covers **every prompt plus
the `kb.json` they were drawn from**, giving a stable content identity — the same
code, config, and KB reproduce the same hash, and a taskset built against a
different KB lands in a different directory and cannot be silently confused with
this one.

`manifest.json` records when and from what the taskset was built:

| Field | Meaning |
|---|---|
| `built_at` | UTC timestamp of the build |
| `git_sha` | repository HEAD at build time |
| `git_dirty` | whether tracked files had uncommitted changes — if `true`, `git_sha` alone does not fully reproduce the build |
| `taskset_hash`, `kb_sha256` | content hash of the prompts, and of the KB they were drawn from |
| `gold_self_check`, `generator_skips` | build-integrity results (see below) |

## Using it

```python
from evals.taskset import load_taskset
from aspic_gym import score_answer

rows = load_taskset("data/tasksets/pilot-20260722T114138Z-247eca21")  # reads .jsonl or .jsonl.gz

for row in rows:
    answer = my_model(row["prompt"])                  # prompt needs no further assembly
    score  = score_answer(answer, row["entry"])       # float in [0, 1]
```

Gold answers are computed symbolically by PyArg under grounded semantics — no
LLM judges anything. `score_answer` is the only scorer; partial credit is
task-specific (per-claim accuracy, F1 over sets, and so on).

**Scoring caveat.** `aspic_gym._answer_region` matches the **first**
`[answer]` tag in the text. Reasoning models routinely draft an answer
mid-chain-of-thought and then revise it, so score the model's *final* output
only — never chain-of-thought concatenated with the answer. In one measured
sample of 13, that distinction flipped a score from 0.00 to 1.00.

## The `pilot` taskset

| | |
|---|---|
| Samples | 1200 |
| Grid | 12 tasks × 2 modes × 5 levels × 10 samples = 120 cells |
| Tasks | every registered task except `syntax`, which is excluded from the benchmark |
| Modes | `symbolic` (DSL atoms) and `content` (plain language, drawn from `kb.json`) |
| Levels | 1, 3, 5, 10, 15 |
| Base seed | 20260721 |

Levels are not evenly spaced on purpose. `levels.py` unlocks every *qualitative*
feature by level 5 — conflict and negation at 2, axioms, side branches and
derived attacks at 3, strict rules and attack chains at 4, undercuts at 5 —
while levels 6–15 only scale knobs (more atoms, more rules) without adding new
phenomena. So 1/3/5 walk the feature ladder and 10/15 isolate pure scale at a
saturated feature set.

## How it is generated

```bash
python -m evals.taskset                              # uses evals/conf/taskset/pilot.yaml
python -m evals.taskset taskset.n=20 taskset.name=big # override any field
python -m evals.taskset build_workers=8              # cap parallelism (default: min(cells, cores, 32))
```

Output goes to `data/tasksets/<name>-<UTC-stamp>-<hash8>/`. The build writes a
plain `taskset.jsonl` there, which is gitignored (regenerable, ~19x larger); to
publish, gzip it to `taskset.jsonl.gz` and commit that alongside `manifest.json`
and `config.yaml`. A local rebuild's plain file, when present, is preferred over
the committed snapshot, so regenerating overrides what shipped.

Cells are independent and each seeds its RNG from `(base_seed, task, mode,
level)` alone, so they build in parallel across processes; rows are reassembled
in grid order, and generation is `PYTHONHASHSEED`-independent, so the content
hash is identical whatever the worker count. The generator walks every (task,
mode, level) cell and pulls `n` items:

1. **Seeding.** Each cell gets `cell_seed = blake2b(base_seed, task, mode, level)`.
   A hash rather than a running counter, so adding or removing a task from the
   grid does not renumber every other cell.
2. **Sampling.** One `ASPICDataset` per cell. Because `ASPICDataset.__getitem__`
   derives its RNG from `idx` alone, **raising `n` extends a cell rather than
   reshuffling it** — the first `n` items stay byte-identical, so you can grow a
   taskset without invalidating results already collected on it.
3. **Skips.** A few (task, level) combinations occasionally fail to produce a
   valid theory at a given index even after the generator's own 60 internal
   retries. Those indices are skipped deterministically and counted in
   `manifest.json` under `generator_skips`, rather than shrinking the cell. In
   `pilot`, 2 indices were skipped, both in `counter_argumentation/symbolic/L15`.
4. **Gold self-check.** Every item's own gold answer is scored against itself and
   must return exactly 1.0. A single failure aborts the build. If the parsers
   cannot read their own gold, no model score from that taskset would mean
   anything. All 1200 items in `pilot` pass.

## Reproducing and verifying

Generation is deterministic given the same seed, generator code, and `kb.json`:

```bash
python -m evals.verify_taskset data/tasksets/pilot-20260722T114138Z-247eca21
```

This regenerates the grid from the taskset's own `config.yaml` and compares the
content hash, reporting which prompts drifted if they do not match. Exit codes:
`0` match, `1` mismatch, `2` local `kb.json` differs from the one the taskset was
built against (regeneration is impossible; the shipped file is still usable).

Content-mode prompts are drawn from `kb.json`, so **widening the KB changes
them**. That is intended: a new KB produces a new taskset id, old results stay
attributable to the KB that produced them, and results from the two are never
pooled by accident.

## Known issues (scorer-side; re-scorable without regenerating)

Some cells under-score for **notation** reasons rather than reasoning. These live
in `score_answer`, not in the taskset, so they can be corrected by fixing the
scorer and replaying over stored generations — no new taskset or model run.

- **Construct tasks, content mode** (`attack`, `counter_argumentation`,
  `evidence_construction`, `preference_construction`): a newly added rule has no
  disclosed label, yet the gold refers to it by an auto-assigned ordinal
  (`Rule 5`), and an explicit label makes the whole answer fail parsing.
- **`formalization`, symbolic mode**: the prompt presents source rules as
  `Rule 1`, `Rule 2`, … while the gold names them `d1`, `d2`, …, with the mapping
  undisclosed. A model that echoes the prompt's own
  `[prefer_rule: Rule 3 > Rule 2]` scores ~0.22 instead of 1.0.

Tasks that only name existing elements (`attackers_of`, `claim_identification`)
are unaffected.
