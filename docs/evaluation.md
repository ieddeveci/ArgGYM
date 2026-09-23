# Running an evaluation

ArgGYM generates rows and scores answers. Getting an answer out of a model is
this directory's job, and `evals/` is one way to do it -- not the only one, and
not part of the package. `pip install arggym` still gets three dependencies and
no HTTP client.

```
uv sync                                    # the evals group is included in dev
uv run arggym freeze -c tasksets/standard.yaml -o data/taskset.jsonl
uv run python -m evals.run   taskset=data/taskset.jsonl model=openrouter-gpt-5-medium
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
# evals/conf/model/openrouter-claude-sonnet-4.5-high.yaml
name: openrouter-claude-sonnet-4.5-high
model: anthropic/claude-sonnet-4.5
base_url: https://openrouter.ai/api/v1
api_key_env: OPENROUTER_API_KEY
sampling:
  max_tokens: 64000
extra_body:
  reasoning:
    effort: high
```

```bash
cp .env.example .env        # then fill in OPENROUTER_API_KEY
uv run python -m evals.run model=openrouter-claude-sonnet-4.5-high taskset=data/taskset.jsonl
```

`api_key_env` names the variable rather than holding the key, so the whole
config can be written into a run manifest without writing down a secret. The
harness loads `.env` from the working directory upward at startup; a variable
already exported in the shell wins over the file. `.env.example` lists the
credentials and endpoints the shipped model configs read.

`extra_body` is for anything one provider understands and the rest do not:
OpenRouter's `reasoning` and `provider`, vLLM's `chat_template_kwargs`, Gemini's
`thinking_config`. It is passed through untouched, because validating it would
mean this harness knowing every provider, which is the thing we are avoiding.

Configs ship for Claude and GPT-5 on OpenRouter, GPT-5 on OpenAI, Gemini and
Gemma on AI Studio, and every model the HPC lane serves on vLLM (`hf-*`), plus a
stub for running the pipeline without spending a token. Adding another is
copying one. An `hf-*` config also runs against a vLLM you started yourself:
set `VLLM_BASE_URL`, and serve with the `max_model_len` of the profile of the
same name under `hpc/vllm/models/`, or the config's `max_tokens` will not fit.

The two layers hold different facts and never the same one. The eval config
says what each request carries, the checkpoint id included; the serving
profile says how to serve it (GPU counts, dtype, context length, parser, extra
vLLM flags). They pair by file stem, and `hpc/vllm/verify_bundle.py` checks
that they fit.
Running the `hf-*` configs on the TRUBA cluster is in [hpc/](../hpc/README.md).

A config is named `<provider>-<model>[-<level>]`. The provider is the endpoint
(`hf` for the vLLM lane, `openrouter`, `openai`, `aistudio`), the model is the
checkpoint name after the last `/` in lowercase, and the level is the reasoning
effort the config sends: `openrouter-claude-sonnet-4.5-high`,
`openai-gpt-5-minimal`, `hf-qwen3.8-27b-xhigh`, `hf-gemma-4-31b-it`.

An RL fine-tune is named after the model it was trained from, with an RL tag:
`<provider>-<base model>-rl-<tag>[-<level>]`. Its `model` is the fine-tune's
own repo, which is what gets served, and a `base_model` field names the
checkpoint it started from:

```yaml
name: hf-qwen3-8b-rl-arggym-40k
model: your-org/qwen3-8b-arggym-grpo   # served, at the profile's revision if pinned
base_model: Qwen/Qwen3-8B              # recorded in run.json, never sent
```

A config with `base_model` must take the `-rl-<tag>` form and one without must
not; `tests/evals` and `hpc/vllm/verify_bundle.py` both check this.

A model that takes a reasoning-effort level has one config per level it accepts
and no level-less one. The level changes what the model is asked as much as the
prompt does, so it is part of the name, and therefore of the run directory. A
model whose only switch is thinking on or off (Qwen3, Qwen3.6, Gemma 4 on vLLM)
has one config, with thinking on.

Every config runs under the `cot` elicitation, which asks the model to reason
before it answers. `elicitation=none` sends the question with nothing added; it
is the ablation that measures what `cot` buys.

### What the compatibility layers cost

Worth knowing before a number goes in a paper.

**Anthropic direct drops things silently.** Their own documentation says the
OpenAI SDK "doesn't return Claude's detailed thought process", and that
`response_format`, `reasoning_effort`, `seed` and `logprobs` are ignored, `n`
must be 1, `temperature` is capped at 1, and `usage.completion_tokens_details`
is always empty. "Most unsupported fields are silently ignored rather than
producing errors." That is why `openrouter-claude-sonnet-4.5-*.yaml` goes
through OpenRouter: it returns the reasoning trace over a plain
OpenAI-compatible call.

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
a real reasoning failure -- which is a result -- or a scorer bug, and counting
them is how the second gets noticed. On the construction tasks it also counts
a third kind, an answer the bloat gate zeroed, which `bloat_rate` below
separates out. `scorer_refused` means a
mismatched engine version or a missing field; the previous harness caught that
case with a broad `except`, called it `0.0`, and published a cell of forty items
scoring exactly 0.000.

**Read `truncated_rate` before any mean.** On the August sweep three of seven
models lost between 74% and 89% of their items to the token cap at every level.
Half of all 4,312 truncated generations ended in a repetition loop; one burned
61,440 tokens repeating a single vacuous line 1,654 times and never wrote an
answer. A mean over what survives that is a measurement of the token cap.

**`bloat_rate` says how many zeros the directive budget decided.** The six
construction tasks zero an answer that uses more than twice the minimum number
of directives, and the record's `reason` then starts with `bloated:`.
`metrics.json` gives the share of scored records with that reason for every
task, level and ordering; `results.csv` and `report.md` carry it, and
`score.py` prints it per task. API errors and refused rows are left out of the
denominator, as for `no_answer_region_rate`, and a task with no directive
budget reads 0. Where the minimum is 2 a correct answer of five lines scores
zero, the same as a wrong one, so read this rate beside a construction task's
mean.

### Whether the timeout was big enough

`endpoint.timeout_s` is 28,800 seconds, and it is a safety net for a hung
connection. What ends a long generation is the model's token cap. A completion
stopped by the cap comes back truncated, and `truncated_rate` shows it. A
completion stopped by the clock comes back as an API error, which leaves the
denominator, so a model that loops slowly would score better for looping. The
number is therefore the largest cap in `evals/conf/model/` (253,952 tokens) at a
deliberately slow 10 tokens/s, rounded up to the hour. The report measures
whether it held.

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

`n_requests_timed_out` is `null` whenever *any* generation in the directory was
written without the counter, not only when an error was. A resumed directory can
hold a stale successful generation beside newer ones, and reading that as having
hit the wall zero times would be a measurement it never made. So a dash in that
column means "something here was not counted", which is a generation as often as
it is an unlabelled error.

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

Floors are printed beside every mean for the same reason. On the level-3 rows of
the lite grid `semantics_query` sits at 0.727 and `status_query` at 0.375, so a
model scoring 0.45 on the first is doing worse than answering the same thing
every time. Where one figure per task is wanted, the chance-corrected column is
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

## Figures

```
uv run --extra report python -m evals.figures outputs/runs/* -o outputs/figures
```

The same runs `report.py` tabulates, drawn: contamination by level first, then
the macro-average against difficulty, `status_query` against the construction
tasks, a per-task profile of the least contaminated run, and output length
against truncation. Scores come from `metrics.json` so a point and a table cell
cannot disagree; coverage per level and completion length are counted from
`samples.jsonl`, which is where the per-item record lives.

`matplotlib` is an extra rather than a dependency -- scoring model outputs in CI
should not pull a plotting library -- so the command carries `--extra report`. A
dev checkout has it already.

Two runs on different tasksets are refused rather than drawn together. The table
prints that as a warning because its rows are named; a curve through both points
has nowhere to say that the two models sat different exams.

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
