# Running an evaluation

ArgGYM generates rows and scores answers. Getting an answer out of a model is
this directory's job, and `evals/` is one way to do it -- not the only one, and
not part of the package. `pip install arggym` still gets three dependencies and
no HTTP client.

```
uv sync                                    # the evals group is included in dev
uv run arggym freeze -c tasksets/standard.yaml -o data/taskset.jsonl
uv run python -m evals.run   taskset=data/taskset.jsonl model=gpt-5-openrouter
uv run python -m evals.score outputs/runs/<dir>
uv run python -m evals.report outputs/runs/* -o outputs/reports/latest
```

Three commands because they fail differently. Generating costs money and hours
and can be interrupted. Scoring is free, offline, and changes whenever a scorer
does. Reporting reads what scoring wrote. Keeping them apart means a parser fix
costs seconds instead of a sweep.

## Pointing it at a provider

The harness speaks OpenAI-compatible HTTP and nothing else. Every provider we
use serves that: OpenRouter, OpenAI, Gemini's compatibility endpoint, Vertex's,
Azure's, Anthropic's shim, and any local vLLM. Switching between them is a
config file and an environment variable.

```yaml
# evals/conf/model/claude-openrouter.yaml
name: claude-openrouter
model: anthropic/claude-sonnet-4.5
base_url: https://openrouter.ai/api/v1
api_key_env: OPENROUTER_API_KEY
sampling:
  temperature: 0.0
  max_tokens: 32768
extra_body:
  reasoning:
    effort: high
```

```bash
export OPENROUTER_API_KEY=...
uv run python -m evals.run model=claude-openrouter taskset=data/taskset.jsonl
```

`api_key_env` names the variable rather than holding the key, so the whole
config can be written into a run manifest without writing down a secret.

`extra_body` is for anything one provider understands and the rest do not:
OpenRouter's `reasoning` and `provider`, vLLM's `chat_template_kwargs`, Gemini's
`thinking_config`. It is passed through untouched, because validating it would
mean this harness knowing every provider, which is the thing we are avoiding.

Five configs ship as examples, one per provider. Adding a sixth is copying one.

### What the compatibility layers cost

Worth knowing before a number goes in a paper.

**Anthropic direct drops things silently.** Their own documentation says the
OpenAI SDK "doesn't return Claude's detailed thought process", and that
`response_format`, `reasoning_effort`, `seed` and `logprobs` are ignored, `n`
must be 1, `temperature` is capped at 1, and `usage.completion_tokens_details`
is always empty. "Most unsupported fields are silently ignored rather than
producing errors." That is why `claude-openrouter.yaml` goes through OpenRouter:
it returns the reasoning trace over a plain OpenAI-compatible call.

**Gemini's compatibility endpoint** is still labelled beta and carries the same
silent-drop warning.

**So read the request back.** Every generation records the exact body that was
sent, minus the prompt, in its `request` field. A sampling parameter that a
provider ignored is visible there and nowhere else.

Unknown sampling keys are refused rather than forwarded. A run that silently
dropped one would look configured, record it in the manifest, and generate as if
it were never set: a whole sweep on the previous harness was scored under a
`repetition_penalty` that never reached the server.

## Writing your own solver

The harness holds one object, and it is four lines:

```python
class Solver(Protocol):
    def __call__(self, row: Dict[str, Any]) -> Attempt: ...
```

An agent with tools, a constrained decoder, a symbolic procedure with no model
in it: anything that returns an `Attempt` works, and `evals/solver.py` is the
one shipped implementation. If you would rather call models through litellm or
pydantic-ai, write a solver that does; the harness will not know.

A solver that produced the answer as a *value* rather than as text sets
`Attempt.value` and leaves `completion` empty. Scoring then routes to
`arggym.score_row_value` and extracts nothing, which is the point of that
function existing: asking a structured-output solver to render ArgGYM's DSL so
we could parse it back would be the dataset dictating a serialization.

Values travel to `score.py` as JSON, so two of the four answer shapes need an
encoding -- an operation list is `Operation` objects, and `semantics_query` keys
its map by a pair. `evals/values.py` has both directions.

## Reading the output

A run directory holds what happened, in the order it happened:

```
run.json           the manifest; status running -> completed | failed
prompts.jsonl      written before any call, so a dead run still says what it asked
generations.jsonl  appended as each result lands
samples.jsonl      one scored record per item, written by score.py
metrics.json       the aggregates
```

`generations.jsonl` is the expensive artifact. Everything else can be rebuilt
from it, and nothing can rebuild it. **Copy a finished run somewhere durable.**
`outputs/` is gitignored scratch, and a completed four-model sweep was once
deleted from it by something outside this repository, leaving no raw generation
anywhere on disk. Nothing here can prevent that; only moving the directory can.

### The three outcomes

An API error, a truncated generation and a wrong answer are different events,
and the scored record keeps them apart.

| field | means |
|---|---|
| `api_error` | the request did not succeed. `score` is `null`, never `0.0`. |
| `truncated` | the generation hit its token cap. |
| `no_answer_region` | nothing in the completion was inside the fence. |
| `answer_in_cot` | the answer was only in the reasoning trace, never submitted. |
| `zero_with_region` | a well-formed answer that scored zero. |
| `scorer_refused` | the scorer would not grade the row at all. |

The last two are the ones to look at. `zero_with_region` is either a real
reasoning failure -- which is a result -- or a scorer bug, and counting them is
how the second gets noticed. `scorer_refused` means a mismatched engine version
or a missing field; the previous harness caught that case with a broad `except`,
called it `0.0`, and published a cell of forty items scoring exactly 0.000.

**Read `truncated_rate` before any mean.** On the August sweep three of seven
models lost between 74% and 89% of their items to the token cap at every level.
Half of all 4,312 truncated generations ended in a repetition loop; one burned
61,440 tokens repeating a single vacuous line 1,654 times and never wrote an
answer. A mean over what survives that is a measurement of the token cap.

### Why there is no single number

The report gives a per-task table and refuses to average it. Twelve metrics of
four different kinds, with chance floors spanning half the range, do not add up
to a quantity: the number that comes out moves mostly with which tasks are in
the basket.

Floors are printed beside every score for the same reason. At level 3
`semantics_query` sits at 0.490 and `status_query` at 0.375, so a model scoring
0.45 on the first is doing worse than answering the same thing every time. Where
one figure per task is wanted, the chance-corrected column is
`(score - floor) / (1 - floor)`, which is at least the same quantity across
tasks.

## Resuming, filtering, and what is refused

A rerun into the same directory skips ids already generated without an error, so
an interruption costs the item in flight. Errored items are retried and the
retry supersedes the failure; both lines stay on disk.

Slice a frozen taskset by coordinate rather than freezing a second one:

```bash
uv run python -m evals.run model=stub taskset=data/taskset.jsonl \
    filter.levels=[3,9,15] filter.tasks=[status_query,claim_chain] filter.limit=40
```

A filter that matches nothing raises. Matching nothing silently would report a
missing sweep as a clean run of zero items.

Two more refusals, both for the same reason -- a wrong number that looks right is
worse than an error:

- **`score.py` refuses a taskset whose hash is not the one the run recorded.**
  Scoring answers against questions they were not asked is silent.
- **A run whose API error rate passes `max_error_rate` is marked `failed`, not
  `completed`,** and exits non-zero. A dead endpoint and a model that answers
  badly produce the same low score, and only one of them is a finding.

## Testing it without spending anything

`tests/evals/` stands up a real HTTP server speaking the OpenAI protocol and
points the harness at it. The central test replies to every row with that row's
own reference answer and asserts every task scores 1.0: if anything between
composing a prompt and reading a score loses the answer, that test says so
before a sweep finds out.
