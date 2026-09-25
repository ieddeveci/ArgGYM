# ArgGYM

A procedural benchmark and RL environment for defeasible reasoning, over ASPIC+ under grounded
semantics. Every item is built by construction and checked against the engine, so the gold answer is
never asserted, it is verified. Half the tasks ask what a theory says; the other half ask the model
to change what it says, and have no single reference answer: the scorer runs the engine on the
theory plus the answer, so any answer that reaches the goals counts.

Twelve tasks, fifteen levels, four strength orderings. The generator is the product; a frozen
taskset is one dump of it. ArgGYM owns what a legal answer is, and you own how you get one: the
question states the task and says nothing about where to put the answer, so any solver works.

## Tasks

- `status_query`: state the status of each named claim.
- `semantics_query`: state each claim's status under the semantics named beside it.
- `claim_chain`: write the line of directives that justifies a claim, among defeated decoys.
- `defeat_diagnosis`: say why a claim is not justified: each failure point and its defeater.
- `perturbation`: predict which claims change status when the theory changes.
- `formalization`: translate a natural-language argument into the ASPIC+ DSL.
- `preference_construction`: move claims to required statuses with preferences only.
- `counter_argument`: make a claim's contrary justified.
- `counter_argument_strict`: the same with strict rules allowed, an ablation of the one above.
- `attack`: make a justified claim overruled.
- `defence`: make an attacked claim justified.
- `attack_defense`: both at once, where the naive attack breaks the defence.

`docs/dataset-card.md` describes each task and what its score means.

## Install

```
pip install arggym                # the library: generate, render, parse, score
pip install "arggym[inspector]"   # + the browser inspector
```

For the evaluation harness, the tests and the shipped tasksets, work from a checkout with
[uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/ieddeveci/ArgGYM && cd ArgGYM
uv sync --python 3.13
uv run pytest -q
```

`make` lists the shortcuts.

## Quickstart

Score answers from your own code:

```python
import arggym

ds = arggym.create("counter_argument", level=6, ordering="weakest_link_elitist", size=50)
for entry in ds:
    answer = my_solver(entry["question"])    # your prompt, your parsing
    result = ds.score(answer, entry)
    print(result.score, result.success, result.reason)
```

Or evaluate a model end to end with the harness in `evals/`. The two frozen tasksets ship in
`data/`: `taskset-lite.jsonl` (1,440 rows, the one evals run on) and `taskset.jsonl` (7,200 rows).
Rebuild them from their specs only after a generator change:

```
make freeze-lite        # data/taskset-lite.yaml -> data/taskset-lite.jsonl
make freeze             # data/taskset.yaml      -> data/taskset.jsonl
```

Then generate, score and report. Generation calls any OpenAI-compatible endpoint; copy
`.env.example` to `.env` and fill in the key the model config names.

```
uv run python -m evals.run   taskset=data/taskset-lite.jsonl model=openrouter-claude-sonnet-4.5-high
uv run python -m evals.score outputs/runs/openrouter-claude-sonnet-4.5-high__xml_tags__cot
uv run python -m evals.report outputs/runs/* -o outputs/reports/latest
```

`make eval MODEL=<config>` runs the first two. `scripts/evaluate.py` is a smaller reference
evaluator that uses only the standard library and the public API.

Read scores per task and against the chance floor that `arggym floors <taskset>` measures; the
dataset card says why.

## Reproducing the paper's results

The finished runs are in `results/`, with generations in Git LFS. To rebuild every table from them
without calling a model:

```
make results-unpack
for run in outputs/runs/*/; do uv run python -m evals.score "$run"; done
uv run python -m evals.report outputs/runs/* -o outputs/reports/paper
```

`docs/evaluation.md` ("Reproducing the results") gives the environment, the taskset hash, the model
configs, the vLLM serving setup and the sampling rules needed to regenerate the runs themselves.

## Documentation

| | |
|---|---|
| [`docs/dataset-card.md`](docs/dataset-card.md) | the tasks, what the benchmark measures, and what a score licenses |
| [`docs/dataset-contract.md`](docs/dataset-contract.md) | the interface: rows, answers, scoring, and why each part is the way it is |
| [`docs/evaluation.md`](docs/evaluation.md) | running a sweep, reading its output, and reproducing the results |
| [`docs/notation.md`](docs/notation.md) | the DSL, the semantics conventions, the answer formats |
| [`docs/architecture.md`](docs/architecture.md) | the code, module by module |
| [`docs/defeasible-reasoning-primer.md`](docs/defeasible-reasoning-primer.md) | ASPIC+ and grounded semantics from scratch, with a worked example |
| [`hpc/README.md`](hpc/README.md) | serving the open-weight models with vLLM on the TRUBA cluster |
| [`results/README.md`](results/README.md) | the layout of the shared runs |

## License

MIT; see [LICENSE](LICENSE).
