"""Run one model over one taskset. Inference only -- no scoring, no prompting."""
from __future__ import annotations

import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import artifacts, client, elicitation  # noqa: E402
from evals.taskset import git_sha, load_taskset  # noqa: E402


def resolve_taskset_dir(root: Path, cfg: DictConfig) -> Path:
    """Locate the taskset directory, preferring an explicit id."""
    ts_root = root / cfg.taskset_root
    explicit = cfg.get("taskset_id")
    if explicit:
        d = ts_root / explicit
        if not d.exists():
            raise FileNotFoundError(f"taskset {explicit} not found under {ts_root}")
        return d
    matches = sorted(p for p in ts_root.glob(f"{cfg.taskset.name}-*") if p.is_dir())
    if not matches:
        raise FileNotFoundError(
            f"no taskset matching '{cfg.taskset.name}-*' under {ts_root}; "
            f"run: python -m evals.taskset")
    # Directory names carry a UTC build stamp, so the last one sorted is the
    # newest. Default to it, but announce the choice: within a sweep every model
    # must be pinned to the SAME taskset (run_all.sh sets taskset_id), or a newer
    # build appearing mid-sweep would compare models across different questions.
    chosen = matches[-1]
    if len(matches) > 1:
        print(f"note: {len(matches)} tasksets match '{cfg.taskset.name}-*'; using the "
              f"newest ({chosen.name}). Pass taskset_id=<dir> to pin a specific one.",
              flush=True)
    return chosen


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    root = Path(__file__).resolve().parent.parent
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ts_dir = resolve_taskset_dir(root, cfg)
    rows = load_taskset(ts_dir)
    manifest = artifacts.read_json(ts_dir / "manifest.json") or {}

    # Evaluate a slice of the canonical taskset (e.g. only low difficulty) instead
    # of building a derived taskset. Recorded in run.json so a run is traceable to
    # exactly the subset it covered.
    eval_filter = OmegaConf.to_container(cfg.get("eval_filter") or {}, resolve=True)
    n_total = len(rows)
    for field, key in (("levels", "level"), ("tasks", "task"), ("modes", "mode")):
        allowed = eval_filter.get(field)
        if allowed:
            allowed = set(allowed)
            rows = [r for r in rows if r[key] in allowed]
    if not rows:
        raise RuntimeError(f"eval_filter {eval_filter} left 0 of {n_total} rows")

    sampling = OmegaConf.to_container(cfg.model.sampling, resolve=True)
    elicit = OmegaConf.to_container(cfg.get("elicitation") or {}, resolve=True)
    done = artifacts.completed_sample_ids(run_dir)
    todo = [r for r in rows if r["sample_id"] not in done]

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    artifacts.write_json(run_dir / "run.json", {
        "status": "running",
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "sampling": sampling,
        "generation": OmegaConf.to_container(cfg.generation, resolve=True),
        "endpoint": OmegaConf.to_container(cfg.endpoint, resolve=True),
        # Which prompting method produced these generations. The taskset carries
        # only the task and the [answer] contract, so without this a run is not
        # traceable to how it was elicited.
        "elicitation": elicit,
        "elicitation_summary": elicitation.describe(elicit),
        "taskset_id": ts_dir.name,
        "taskset_hash": manifest.get("taskset_hash"),
        "kb_sha256": manifest.get("kb_sha256"),
        "eval_filter": eval_filter,
        "n_taskset_total": n_total,
        "n_samples": len(rows),
        "n_resumed": len(done),
        "sweep_id": cfg.get("sweep_id"),
        "git_sha": git_sha(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vllm_image": cfg.model.image,
        "started_utc": started,
    })

    print(f"model={cfg.model.name}  taskset={ts_dir.name}  "
          f"elicitation={elicit.get('name', 'none')}  "
          f"todo={len(todo)}/{len(rows)} (resumed {len(done)})", flush=True)

    for r in rows:
        artifacts.write_sample_input(run_dir, r)

    state = {"n": 0, "errors": 0, "t": time.time()}

    enable_thinking = cfg.model.get("enable_thinking")

    # max_tokens must fit the model's context. A model with a per-model
    # max_model_len (gemma-4 caps at 20480 for KV headroom) needs a smaller
    # max_tokens than the roster default, or every request is rejected with
    # "max_tokens cannot be greater than max_model_len".
    max_tokens = int(cfg.model.get("max_tokens") or cfg.generation.max_tokens)

    def work(row: dict) -> dict:
        prompt, system = elicitation.apply(row["prompt"], elicit)
        gen = client.complete(
            cfg.endpoint.base_url, cfg.model.name, prompt, sampling,
            max_tokens, int(cfg.endpoint.timeout_s),
            int(cfg.endpoint.retries), enable_thinking, system)
        gen["sample_id"] = row["sample_id"]
        return gen

    def on_result(gen: dict) -> None:
        sid = gen["sample_id"]
        artifacts.write_generation(run_dir, sid, gen)
        artifacts.append_jsonl(run_dir / "generations.jsonl", gen)
        state["n"] += 1
        if gen.get("error"):
            state["errors"] += 1
        if state["n"] % 25 == 0 or state["n"] == len(todo):
            el = time.time() - state["t"]
            rate = state["n"] / el if el else 0
            eta = (len(todo) - state["n"]) / rate / 60 if rate else 0
            print(f"  {state['n']}/{len(todo)}  errors={state['errors']}  "
                  f"{rate*60:.1f}/min  eta={eta:.0f}m", flush=True)

    if todo:
        client.run_batch(todo, work, int(cfg.generation.concurrency), on_result)

    elapsed = round(time.time() - t0, 1)
    run = artifacts.read_json(run_dir / "run.json") or {}
    run.update({
        "status": "completed",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed,
        "n_generated": state["n"],
        "n_errors": state["errors"],
    })
    artifacts.write_json(run_dir / "run.json", run)
    print(f"done in {elapsed/60:.1f}m; {state['n']} generated, "
          f"{state['errors']} errors -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
