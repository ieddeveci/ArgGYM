"""Generate. This program never scores.

It reads a frozen taskset, composes a prompt for every row, calls a solver, and
writes what came back. Scoring is `score.py`, offline, from these files -- so a
parser fix, a template change or a scorer bug costs a rerun of a few seconds
rather than the hours of inference that produced the completions.

    uv run python -m evals.run taskset=data/taskset.jsonl model=gpt-5-openai-medium
    uv run python -m evals.run model=qwen3.8-27b-vllm-medium filter.levels=[3,9] filter.limit=20
"""
from __future__ import annotations

import dataclasses
import os
import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from evals import artifacts, taskset, values
from evals.client import Endpoint
from evals.prompt import Elicitation
from evals.solver import ChatSolver
from evals.types import SOLVER_RAISED, SOLVER_VALUE_NOT_JSON, Attempt, Solver


def _plain(node: Any) -> Dict[str, Any]:
    """An omegaconf node as an ordinary dict, tolerating an absent key.

    `cfg.get(k, {})` hands back a bare dict when the key is missing, and
    `to_container` refuses one.
    """
    if node is None:
        return {}
    return dict(node) if isinstance(node, dict) else OmegaConf.to_container(node,
                                                                           resolve=True)


def refuse_a_changed_run(run_dir: str, meta: Dict[str, Any]) -> None:
    """Refuse to add generations from one configuration to another's directory.

    Resume keys on the row id alone, so a directory reused under a different
    model, template, elicitation, filter or token cap ends up holding two
    configurations' completions under one manifest, all extracted with whichever
    template was named last. Every other guard passes: same taskset, same ids,
    same hash.
    """
    path = os.path.join(run_dir, artifacts.RUN)
    if not os.path.exists(path):
        return
    import json

    with open(path) as f:
        before = json.load(f)
    # `elicitation` is a name and `filter` decides which rows exist, so both
    # belong here: `elicitation.system=...` changes the system prompt while
    # leaving the name alone, and a second invocation under a narrower filter
    # leaves a manifest describing two items over a directory holding twelve.
    changed = [k for k in ("taskset_hash", "template", "elicitation",
                           "elicitation_config", "filter")
               if before.get(k) != meta.get(k)]
    # Sampling too, not just the model. Temperature and the token cap change
    # what the model was asked as surely as the prompt does -- on this box a
    # 24576-token cap scored eight of twelve tasks at exactly 0.000 and a larger
    # one did not -- so generations made under two caps must not pool.
    was, now = before.get("endpoint") or {}, meta["endpoint"]
    changed += [f"endpoint.{k}" for k in ("model", "base_url", "sampling",
                                          "extra_body")
                if was.get(k) != now.get(k)]
    if changed:
        raise SystemExit(
            f"{run_dir} already holds a run whose {', '.join(changed)} "
            f"differ(s) from this one. Resuming into it would mix generations "
            f"from two configurations under one manifest. Pass a different "
            f"run_id=, or resume=false to start this directory over.")


class RunFailed(SystemExit):
    """More of the run failed to reach the provider than the config allows."""


def quiet_http() -> None:
    """One INFO line per request is thousands of lines of noise on a full taskset."""
    import logging

    for name in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


def build_solver(cfg: DictConfig) -> ChatSolver:
    m = cfg.model
    endpoint = Endpoint(
        model=m.model,
        base_url=m.get("base_url"),
        api_key_env=m.get("api_key_env", "OPENAI_API_KEY"),
        sampling=_plain(m.get("sampling")),
        extra_body=_plain(m.get("extra_body")),
        timeout_s=cfg.endpoint.timeout_s,
        retries=cfg.endpoint.retries,
    )
    elicit = Elicitation(**_plain(cfg.elicitation))
    return ChatSolver(endpoint, template=cfg.template.name, elicitation=elicit)


def git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001 - a tarball checkout has no git
        return None


def manifest(cfg: DictConfig, solver: ChatSolver, ts_manifest: Dict[str, Any],
             ts_path: str, rows: List[Dict[str, Any]], n_total: int,
             n_resumed: int) -> Dict[str, Any]:
    """Everything needed to say what this run was, and nothing secret.

    The endpoint is recorded by the *name* of its key variable, never its value.
    """
    return {
        "status": "running",
        "taskset": ts_path,
        # The identity of the questions. `score.py` refuses a run whose hash
        # does not match the taskset it is handed.
        "taskset_hash": ts_manifest.get("taskset_hash"),
        "taskset_versions": ts_manifest.get("versions", {}),
        "endpoint": solver.client.endpoint.redacted(),
        "template": cfg.template.name,
        "elicitation": solver.elicitation.name,
        # The text, not just the name. A name is what a config file happens to
        # call a system prompt, and `refuse_a_changed_run` compares this so two
        # different prompts under one name cannot pool into one directory.
        "elicitation_config": dataclasses.asdict(solver.elicitation),
        "filter": _plain(cfg.filter),
        # Both numbers, so a report can tell a filtered run from a short one.
        "n_taskset_total": n_total,
        "n_selected": len(rows),
        "n_resumed": n_resumed,
        "concurrency": cfg.generation.concurrency,
        "git_sha": git_sha(),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def generate(rows: List[Dict[str, Any]], solver: Solver, run_dir: str,
             concurrency: int, progress: bool = True) -> Dict[str, int]:
    """Call the solver on every row, writing each result the moment it lands."""
    path = os.path.join(run_dir, artifacts.GENERATIONS)
    counts = {"generated": 0, "errors": 0, "truncated": 0}
    with artifacts.Appender(path) as out:
        def one(row: Dict[str, Any]) -> None:
            try:
                attempt = solver(row)
            except Exception as e:  # noqa: BLE001
                # A solver that raises is still infrastructure. One bad row must
                # not cost the rows already paid for.
                attempt = Attempt(error=f"solver raised: {type(e).__name__}: {e}",
                                  error_kind=SOLVER_RAISED)
            if attempt.value is not None:
                try:
                    values.check_json(attempt.value, row["id"])
                except TypeError as e:
                    # Inside the pool, and the completion is kept. Raising here
                    # killed the run *and* threw away the generation it was
                    # complaining about, which was already paid for. The record
                    # carries an error, so the item is unmeasured rather than
                    # zero, and `max_error_rate` ends a sweep whose solver does
                    # this on every row.
                    attempt = dataclasses.replace(
                        attempt, value=None, error=str(e),
                        error_kind=SOLVER_VALUE_NOT_JSON)
            out.write({"id": row["id"], "task": row["task"],
                       "level": row["metadata"]["level"],
                       "ordering": row["metadata"]["ordering"],
                       **attempt.to_dict()})
            counts["generated"] += 1
            counts["errors"] += bool(attempt.error)
            counts["truncated"] += bool(attempt.truncated)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            it = pool.map(one, rows)
            if progress:
                try:
                    from tqdm import tqdm

                    it = tqdm(it, total=len(rows), unit="item")
                except ImportError:
                    pass
            list(it)
    return counts


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Hydra has already made the run directory and chdir'd into it.
    execute(cfg, os.getcwd(), hydra.utils.get_original_cwd())


def execute(cfg: DictConfig, run_dir: str, origin: str = ".") -> Dict[str, Any]:
    """One run, start to finish, returning its manifest.

    Separated from `main` so it can be tested. Everything documented about a run
    -- the manifest, the prompts written before any call, resume, and the
    error-rate gate -- lives here, and none of it was reachable from a test
    while it sat inside a `@hydra.main` entry point.

    A relative taskset path means what the caller typed, hence `origin`: Hydra
    chdirs into the run directory before `main` runs.
    """
    quiet_http()
    ts_path = os.path.join(origin, cfg.taskset)
    ts_manifest, all_rows = taskset.load(ts_path)
    rows = taskset.select(
        all_rows,
        tasks=cfg.filter.get("tasks"), levels=cfg.filter.get("levels"),
        orderings=cfg.filter.get("orderings"), limit=cfg.filter.get("limit"))

    # Held for the whole run. The directory is named by the configuration, so
    # two invocations of one command land in the same place and would each skip
    # the ids the other is still generating.
    with artifacts.exclusive(run_dir):
        solver = build_solver(cfg)
        meta = manifest(cfg, solver, ts_manifest, ts_path, rows, len(all_rows), 0)

        if cfg.resume:
            refuse_a_changed_run(run_dir, meta)
        else:
            artifacts.restart(run_dir)
        done = artifacts.completed_ids(run_dir) if cfg.resume else set()
        todo = [r for r in rows if r["id"] not in done]
        meta["n_resumed"] = len(done)
        artifacts.write_json(os.path.join(run_dir, artifacts.RUN), meta)

        # Before any call, so a run that dies mid-flight still says what it asked.
        with artifacts.Appender(os.path.join(run_dir, artifacts.PROMPTS)) as p:
            for row in todo:
                system, user = solver.prompt_of(row)
                p.write({"id": row["id"], "system": system, "user": user})

        print(f"{len(todo)} of {len(rows)} items -> {run_dir}", file=sys.stderr)
        started = time.monotonic()
        try:
            counts = generate(todo, solver, run_dir, cfg.generation.concurrency)
        except BaseException:
            # A manifest left at "running" cannot be told from a run still in
            # flight, so a crash reads as work in progress for as long as
            # anyone is willing to wait for it. The generations are on disk and
            # the run resumes; only the status is corrected here.
            meta["status"] = "crashed"
            meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            artifacts.write_json(os.path.join(run_dir, artifacts.RUN), meta)
            raise
        elapsed = time.monotonic() - started

        # Counted over everything on disk, not over this invocation. A resumed
        # run that generated one item would otherwise write `n_generated: 0` and
        # an error rate measured on a single row over the record of a whole
        # sweep -- and this manifest is what the "not measured must not look
        # like measured as failing" argument rests on.
        final = artifacts.best_per_id(run_dir)
        n_errors = sum(1 for r in final.values() if r.get("error"))
        n_truncated = sum(1 for r in final.values() if r.get("truncated"))
        error_rate = n_errors / len(final) if final else 0.0
        meta.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    elapsed_s=round(elapsed, 1),
                    n_generated=len(final), n_errors=n_errors,
                    n_truncated=n_truncated,
                    n_this_invocation=counts["generated"],
                    # The selected rows that have no generation, counted by id.
                    # `len(rows) - len(final)` counts one set against another and
                    # goes negative the moment the directory holds an id this
                    # invocation did not select.
                    n_missing=sum(1 for r in rows if r["id"] not in final),
                    error_rate=round(error_rate, 4))
        # A run that mostly failed to reach the provider is not a measurement of
        # a model. Saying so in the manifest keeps "not measured" from reading as
        # "measured as failing" -- the previous harness lost two cells that way
        # and the report could not tell.
        meta["status"] = "failed" if error_rate > cfg.max_error_rate else "completed"
        artifacts.write_json(os.path.join(run_dir, artifacts.RUN), meta)

        print(f"{meta['status']}: {len(final)} of {len(rows)} generated, "
              f"{n_errors} errors ({error_rate:.1%}), "
              f"{n_truncated} truncated, {elapsed:.0f}s "
              f"({counts['generated']} this run)", file=sys.stderr)
        if meta["status"] == "failed":
            raise RunFailed(
                f"error rate {error_rate:.1%} is above max_error_rate "
                f"{cfg.max_error_rate}. The generations are kept and the run is "
                f"resumable; read the errors in {artifacts.GENERATIONS} before "
                f"rerunning, because a dead endpoint and a rejected request look "
                f"the same in a score.")
        return meta


if __name__ == "__main__":
    main()
