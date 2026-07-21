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

from evals import artifacts, client  # noqa: E402
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
    matches = sorted(ts_root.glob(f"{cfg.taskset.name}-*"))
    if not matches:
        raise FileNotFoundError(
            f"no taskset matching '{cfg.taskset.name}-*' under {ts_root}; "
            f"run: python -m evals.taskset")
    if len(matches) > 1:
        raise RuntimeError(
            f"{len(matches)} tasksets match '{cfg.taskset.name}-*': "
            f"{[m.name for m in matches]}. Pass taskset_id=<dir> to disambiguate -- "
            f"guessing would risk comparing models across different question sets.")
    return matches[0]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    root = Path(__file__).resolve().parent.parent
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ts_dir = resolve_taskset_dir(root, cfg)
    rows = load_taskset(ts_dir)
    manifest = artifacts.read_json(ts_dir / "manifest.json") or {}

    sampling = OmegaConf.to_container(cfg.model.sampling, resolve=True)
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
        "taskset_id": ts_dir.name,
        "taskset_hash": manifest.get("taskset_hash"),
        "kb_sha256": manifest.get("kb_sha256"),
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
          f"todo={len(todo)}/{len(rows)} (resumed {len(done)})", flush=True)

    for r in rows:
        artifacts.write_sample_input(run_dir, r)

    state = {"n": 0, "errors": 0, "t": time.time()}

    enable_thinking = cfg.model.get("enable_thinking")

    def work(row: dict) -> dict:
        gen = client.complete(
            cfg.endpoint.base_url, cfg.model.name, row["prompt"], sampling,
            int(cfg.generation.max_tokens), int(cfg.endpoint.timeout_s),
            int(cfg.endpoint.retries), enable_thinking)
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
