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
- KB pipeline: `build_kb.py` + `generate_argumentations.py` build a content KB from `claims.json` via Ollama models → `kb.json`; `merge_kb.py` merges KBs from multiple models; `audit_kb.py` engine-verifies records.
- `app.py` — Flask playground to browse/serve tasks.

## Conventions

- Default ordering: `last_link_elitist` (see `ORDERINGS` in `aspic_engine.py`).
- `workspace/` is untracked personal working area (notes, todos, coordination files); don't commit it.
- New to the domain? Read `workspace/benchmark/defeasible-reasoning-primer.md`.
