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

Configs ship for OpenRouter, OpenAI, Gemini, a local vLLM and Qwen3.8 on vLLM.
Adding another is copying one.

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
and the scored record keeps them apart. Every field below is present on every
record, whatever happened, so a reader never has to guess whether a missing key
means `false` or means "not applicable".

| field | means |
|---|---|
| `api_error` | the request did not succeed. `score` is `null`, never `0.0`. |
| `api_error_kind` | which failure it was. `null` beside a `null` `api_error` means nothing failed; `null` beside a real one means the cause was never recorded, which is true only of generations made before this field existed. `score.py` calls that case `unclassified` and refuses to guess the cause back out of the message. |
| `truncated` | the generation hit its token cap. |
| `no_answer_region` | no fenced answer was found anywhere, completion or reasoning. |
| `answer_in_cot` | a fenced answer was found only in the reasoning, never submitted. |
| `zero_with_region` | a well-formed answer that scored zero. |
| `scorer_refused` | the scorer would not grade the row at all. |
| `latency_s` | what the whole row cost in wall time, every retry and backoff included. `null` when nothing timed it -- a solver need not have a clock. |
| `attempt_latency_s` | the slowest single request made for the row, which is what `timeout_s` bounds. `null` on the same terms. |
| `requests_timed_out` | how many of the row's requests expired, including on a row that then answered. |

The last two of the first group are the ones to look at. `zero_with_region` is
either a real reasoning failure -- which is a result -- or a scorer bug, and
counting them is how the second gets noticed. `scorer_refused` means a
mismatched engine version or a missing field; the previous harness caught that
case with a broad `except`, called it `0.0`, and published a cell of forty items
scoring exactly 0.000.

**Read `truncated_rate` before any mean.** On the August sweep three of seven
models lost between 74% and 89% of their items to the token cap at every level.
Half of all 4,312 truncated generations ended in a repetition loop; one burned
61,440 tokens repeating a single vacuous line 1,654 times and never wrote an
answer. A mean over what survives that is a measurement of the token cap.

### Whether the timeout was big enough

`endpoint.timeout_s` is 5400 seconds. It was 1800, and on the August sweep 1800
expired on 9.8%, 14.8% and 21.8% of `qwen3.6-27b`'s requests at levels 3, 6 and
9 -- the rate rising with the level, because a harder item is a longer
generation. Nothing measured whether the new number was enough, so the report
now does.

`metrics.json` carries two kinds of number about it, and they are not
substitutes.

The **latency percentiles** (`latency_s_p50`, `latency_s_p95`, `latency_s_max`,
with `timeout_s` and `near_timeout_s` beside them) are over the rows that
answered, and each row contributes its *slowest* request -- the one the timeout
was up against, not the one that happened to succeed. `n_near_timeout` counts
the rows whose slowest request came within 80% of the wall.

**`n_requests_timed_out`** counts every request that expired, whatever became of
its row. The two are not redundant. A row that hit the wall and recovered is in
both: it answered, so it has percentiles, and its slowest request expired. A row
that expired on all of its attempts is in neither percentile -- it is an error,
and an error's latency is a measurement of the timeout rather than of the
model -- so it is counted here and nowhere else. And this is per *request*: three
expired attempts on one row are one lost row and three completions the server
generated for nothing.

A row carries `api_error_kind: "timeout"`, and so appears in `n_api_timeout`,
only when *all* of its requests expired -- three of them, at the shipped
`retries: 2`. So the ordinary sequence is: `n_requests_timed_out` rises first,
while `n_api_error` is still zero and the run still looks healthy, and
`n_api_timeout` only starts moving once rows begin running out of retries.
Watching the second alone means watching the point at which data is already
lost.

`n_api_error` is split by cause in `api_errors_by_kind`. Seven labels are fixed:
`timeout`, `connection`, `malformed_response`, `refusal` and `caller_error` from
the client, plus `solver_raised` and `solver_value_not_json` for the two
failures that happen outside any client. Two more are open-ended -- a status
code names itself as `http_401`, and a provider's stop reason as
`finish_reason_content_filter`. Leaving those open is deliberate: normalising
them would mean this harness knowing every provider's vocabulary, and a provider
we have not met would be filed under whichever of our buckets was least wrong.
The cost is that two providers' error tables have to be read by label rather
than compared cell for cell.

Every one of these numbers is `null`, and prints as `-` rather than `0`, on a
run that predates the field. An absent measurement and a measurement of zero are
different findings, and a `0` in the timeout column of a run nobody measured is
the more comfortable of the two.

The yardstick is read from the generations, not from the manifest. `run.py`
rewrites `run.json` on every invocation and `refuse_a_changed_run` does not
treat a changed `timeout_s` as a changed run -- a different deadline is not a
different question, so refusing to resume over one would cost a sweep its
generations for nothing. Each generation therefore records the deadline it ran
under, and a directory holding two of them reports no headroom at all rather
than measuring old generations against a new wall.

### Why there is no single number

The report gives a per-task table and refuses to average it. Twelve metrics of
four different kinds, with chance floors spanning half the range, do not add up
to a quantity: the number that comes out moves mostly with which tasks are in
the basket.

Floors are printed beside every mean for the same reason. At level 3
`semantics_query` sits at 0.806 and `status_query` at 0.375, so a model scoring
0.45 on the first is doing worse than answering the same thing every time. Where
one figure per task is wanted, the chance-corrected column is
`(score - floor) / (1 - floor)`, which is at least the same quantity across
tasks.

Two things follow from a floor being *measured* rather than given. Each
breakdown measures its own — a level-15 group is corrected by a level-15 floor,
not by one averaged over the grid. And a floor is measured over the rows a run
actually scored, so a filtered run's floors are estimates from that filter. The
report marks a task's floor with `*` when two runs disagree about it, and also
when one of them could not measure it at all: `(score - floor) / (1 - floor)`
then cannot be reproduced from the single number in the column. A floor that
could not be measured -- `arggym.floors` refuses a row it cannot grade -- is
named in `_meta.floors_unmeasured` rather than left as a blank cell that reads
like a task with no floor.

There is no floor column beside the success-rate table. A floor is the mean
*score* of the best uninformed answer, not its success rate, and printing it
there would invite exactly the comparison it exists to prevent.

## Resuming, filtering, and what is refused

**A run directory is named by what identifies the run**, not by the clock:

```yaml
run_id: ${model.name}__${template.name}__${elicitation.name}
hydra.run.dir: outputs/runs/${run_id}
```

So running the same command twice resumes rather than starting a second copy.
That is the whole mechanism; there is nothing to pass. Give `run_id=` a value of
your own for a deliberate second run of one configuration -- and for a filtered
one, or a second token cap, since the name carries only the model, the template
and the elicitation while the guard below compares everything that changes what
the model was asked.

A rerun skips ids that already have a good generation, so an interruption costs
the item in flight. A failure and its retry both stay on disk and the good one
is the one that counts, in either order — so a transient failure after a
success never discards work already paid for. `resume=false` deletes the
directory's generations and starts over, because the writer appends and leaving
them would be resuming under a flag that says otherwise.

Slice a frozen taskset by coordinate rather than freezing a second one:

```bash
uv run python -m evals.run model=stub taskset=data/taskset.jsonl \
    filter.levels=[3,9,15] filter.tasks=[status_query,claim_chain] filter.limit=40
```

A filter that matches nothing raises. Matching nothing silently would report a
missing sweep as a clean run of zero items.

`filter.limit=0` is refused for the same reason: it is not "no limit", it is a
filter that matches nothing, and it would be written down as a completed run
with an error rate of zero.

The rest of the refusals are all the same reason -- a wrong number that looks
right is worse than an error:

- **A taskset whose contents do not match its own manifest is refused on load.**
  The hash is recomputed from the rows, not read out of the manifest and
  compared against a copy of itself: a file truncated to eight lines loads seven
  rows and would otherwise pass, then get the full grid's hash printed beside
  seven items' numbers.
- **A taskset with no manifest at all is refused too.** It names no hash, so
  `score.py` compares the run's `None` against the file's `None`, finds them
  equal, and scores yesterday's completions against today's gold. Row ids are
  stable across regenerations, so nothing else catches it either.
- **`score.py` refuses a taskset whose hash is not the one the run recorded.**
  Scoring answers against questions they were not asked is silent.
- **A directory will not take a second configuration.** Resume keys on the row
  id, so reusing one with a different model, template, elicitation, filter or
  sampling would mix two configurations' completions under one manifest, all
  extracted with whichever template was named last — and every other guard
  would pass. The elicitation is compared by its text, not by its name.
- **A directory will not take two runs at once.** It is named by the
  configuration, so launching the same command twice is an ordinary accident;
  both invocations would read the resume list before either wrote to it. The
  second one exits saying the first holds the directory.
- **A provider that did not answer is an error, not a zero.** A
  `finish_reason` of `content_filter`, or a refusal beside a null content,
  leaves an empty completion that would otherwise score zero on every task,
  publishing the provider's policy as the model's reasoning.
- **A run whose API error rate passes `max_error_rate` is marked `failed`, not
  `completed`,** and exits non-zero. A dead endpoint and a model that answers
  badly produce the same low score, and only one of them is a finding.

## Testing it without spending anything

`tests/evals/` stands up a real HTTP server speaking the OpenAI protocol and
points the harness at it. The central test replies to every row with that row's
own reference answer and asserts every task scores 1.0: if anything between
composing a prompt and reading a score loses the answer, that test says so
before a sweep finds out.
