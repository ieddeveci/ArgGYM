"""Generate. This program never scores.

It reads a frozen taskset, composes a prompt for every row, calls a solver, and
writes what came back. Scoring is `score.py`, offline, from these files -- so a
parser fix, a template change or a scorer bug costs a rerun of a few seconds
rather than the hours of inference that produced the completions.

    uv run python -m evals.run taskset=data/taskset.jsonl model=gpt-5-openai
    uv run python -m evals.run model=qwen-vllm filter.levels=[3,9] filter.limit=20
"""
from __future__ import annotations

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

from evals import artifacts, taskset
from evals.client import Endpoint
from evals.prompt import Elicitation
from evals.solver import ChatSolver
from evals.types import Attempt, Solver


def _plain(node: Any) -> Dict[str, Any]:
    """An omegaconf node as an ordinary dict, tolerating an absent key.

    `cfg.get(k, {})` hands back a bare dict when the key is missing, and
    `to_container` refuses one.
    """
    if node is None:
        return {}
    return dict(node) if isinstance(node, dict) else OmegaConf.to_container(node,
                                                                           resolve=True)


def quiet_http() -> None:
    """One INFO line per request is 480 lines of noise on a full taskset."""
    import logging

    for name in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


def build_solver(cfg: DictConfig) -> ChatSolver:
    m = cfg.model
    endpoint = Endpoint(
        model=m.model, base_url=m.get("base_url"),
        api_key_env=m.get("api_key_env", "OPENAI_API_KEY"),
        sampling=_plain(m.get("sampling")), extra_body=_plain(m.get("extra_body")),
        timeout_s=cfg.endpoint.timeout_s, retries=cfg.endpoint.retries)
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
        "elicitation_summary": solver.elicitation.summary(),
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
                attempt = Attempt(error=f"solver raised: {type(e).__name__}: {e}")
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
    # Hydra has already made the run directory and chdir'd into it, so a
    # relative taskset path in the config means what the caller typed, not what
    # it would resolve to from in here.
    quiet_http()
    ts_path = os.path.join(hydra.utils.get_original_cwd(), cfg.taskset)
    ts_manifest, all_rows = taskset.load(ts_path)
    rows = taskset.select(
        all_rows,
        tasks=cfg.filter.get("tasks"), levels=cfg.filter.get("levels"),
        orderings=cfg.filter.get("orderings"), limit=cfg.filter.get("limit"))

    run_dir = os.getcwd()  # Hydra has already made and entered it.
    done = artifacts.completed_ids(run_dir) if cfg.resume else set()
    todo = [r for r in rows if r["id"] not in done]

    solver = build_solver(cfg)
    meta = manifest(cfg, solver, ts_manifest, ts_path, rows, len(all_rows),
                    len(done))
    artifacts.write_json(os.path.join(run_dir, artifacts.RUN), meta)

    # Before any call, so a run that dies mid-flight still says what it asked.
    with artifacts.Appender(os.path.join(run_dir, artifacts.PROMPTS)) as p:
        for row in todo:
            system, user = solver.prompt_of(row)
            p.write({"id": row["id"], "system": system, "user": user})

    print(f"{len(todo)} of {len(rows)} items -> {run_dir}", file=sys.stderr)
    started = time.monotonic()
    counts = generate(todo, solver, run_dir, cfg.generation.concurrency)
    elapsed = time.monotonic() - started

    n = max(counts["generated"], 1)
    error_rate = counts["errors"] / n
    meta.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                elapsed_s=round(elapsed, 1), **{f"n_{k}": v for k, v in counts.items()},
                error_rate=round(error_rate, 4))
    # A run that mostly failed to reach the provider is not a measurement of a
    # model. Saying so in the manifest keeps "not measured" from reading as
    # "measured as failing" -- the previous harness lost two cells that way and
    # the report could not tell.
    meta["status"] = "failed" if error_rate > cfg.max_error_rate else "completed"
    artifacts.write_json(os.path.join(run_dir, artifacts.RUN), meta)

    print(f"{meta['status']}: {counts['generated']} generated, "
          f"{counts['errors']} errors ({error_rate:.1%}), "
          f"{counts['truncated']} truncated, {elapsed:.0f}s", file=sys.stderr)
    if meta["status"] == "failed":
        raise SystemExit(
            f"error rate {error_rate:.1%} is above max_error_rate "
            f"{cfg.max_error_rate}. The generations are kept and the run is "
            f"resumable; read the errors in {artifacts.GENERATIONS} before "
            f"rerunning, because a dead endpoint and a rejected request look "
            f"the same in a score.")


if __name__ == "__main__":
    main()
