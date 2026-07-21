# ArgGYM

Procedural, engine-verified benchmark and RL environment for **defeasible reasoning**, built on the **ASPIC+** structured argumentation framework under **grounded semantics**. Gold answers are computed symbolically with PyArg (`python-argumentation`), never by an LLM.

## Project context

Research collaboration repo. Current phase: turn the task generator into a fixed benchmark (gold test set, content + symbolic splits) and evaluate frontier models. A later method project (self-play RL) will build on it.

## Architecture

- `aspic_engine.py` — `ASPICFramework`: wraps PyArg. Builds an ASPIC+ theory (axioms, ordinary premises, strict/defeasible rules, preferences), computes the grounded extension and per-claim statuses: `JUSTIFIED / OVERRULED / UNDECIDED / UNSATISFIABLE`.
- `aspic_dsl.py` — parses the bracket-directive DSL (`[premise: x]`, `[defeasible d1: a AND b => c]`, `[prefer_rule: d1 > d2]`, …) into `Operation`s; malformed directives are dropped with reasons.
- `aspic_api.py` — `ASPICVerifier`, convenience layer over the engine.
- `aspic_gym.py` — core: random theory generation (`TheoryConfig`), dataset creation (`create_dataset`), answer scoring (`score_answer`).
- `tasks/` — one framer per task, registered in `tasks/registry.py` (13 tasks: status_query, attack, enthymeme, formalization, …).
- `levels.py` — difficulty schedule (levels 1–15): per-task knobs/gates/recipes; `FEATURES` maps feature → unlock level.
- `aspic_content.py` — natural-language ("content") rendering; tasks come in two modes: *symbolic* (DSL atoms) and *content* (plain-language statements from the KB/templates).
- `prompting.py` — shared prompt intro (explains the formalism to the model), answer-format block, exemplars.
- KB pipeline: `build_kb.py` + `generate_argumentations.py` build a content KB from `claims.json` via a local LLM server → `kb.json`; `merge_kb.py` merges KBs from multiple models; `audit_kb.py` engine-verifies records. The chat client (`_chat`) speaks both Ollama and OpenAI-compatible (vLLM) APIs, auto-detected from `--host` (override with `LLM_API=ollama|openai`).
- `app.py` — Flask playground to browse/serve tasks.
- `evals/` — model evaluation harness. Four file-connected stages so re-scoring never needs a GPU: `taskset.py` freezes the task grid into `outputs/tasksets/<name>-<hash>/` (shared by every model, so prompts are byte-identical across the roster); `runner.py` does inference only against a local vLLM server; `scoring.py` replays `score_answer` over stored generations; `report.py` builds cross-model tables. `serve.sh` / `run_all.sh` serve models serially on GPU 2. Config is Hydra, with `conf/model/*.yaml` as the model registry — each file carries both the vLLM serve flags and the sampling params so the two cannot drift.

## Conventions

- Default ordering: `last_link_elitist` (see `ORDERINGS` in `aspic_engine.py`).
- `workspace/` is untracked personal working area (notes, todos, coordination files); don't commit it.
- New to the domain? Read `workspace/benchmark/defeasible-reasoning-primer.md`.
