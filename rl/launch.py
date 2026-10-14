"""One command for a run: pool, vLLM on one GPU, the trainer on the others, export.

    uv run --group rl python -m rl.launch rl/configs/smoke.yaml
    uv run --group rl python -m rl.launch rl/configs/qwen3-4b.yaml grpo.learning_rate=2e-5

In order: build the pool if `data` names one not built yet; start `vllm serve`
on `vllm.gpu` and wait for it; run `accelerate launch -m rl.train` on
`trainer_gpus`; merge the selected checkpoint and check it (`rl.export`). The
server is torn down on every exit path, Ctrl-C and SIGTERM included, by killing
its whole process group: vLLM forks an engine process that would otherwise
outlive the parent and hold the GPU.

`--no-export` stops after training. Logs go to `<run>/vllm.log` and the
terminal.
"""
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from rl import config as rl_config
from rl.train import output_dir


def vllm_command(cfg: Dict[str, Any]) -> List[str]:
    """The server TRL's client expects (`trl vllm-serve`'s `build_command`, TRL 1.15).

    `--logprobs-mode processed_logprobs` makes the returned log-probs those of
    the distribution actually sampled from. The weight-transfer config enables
    the NCCL endpoint the trainer pushes weights through; it needs
    `VLLM_SERVER_DEV_MODE=1`, which also exposes the prefix-cache reset the
    trainer calls after every sync.
    """
    v, m = cfg["vllm"], cfg["model"]
    return [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", m["name"],
            "--revision", m["revision"], "--dtype", "bfloat16",
            "--host", "127.0.0.1", "--port", str(v["port"]),
            "--max-model-len", str(v["max_model_len"]),
            "--gpu-memory-utilization", str(v["gpu_memory_utilization"]),
            "--weight-transfer-config", '{"backend": "nccl"}',
            "--logprobs-mode", "processed_logprobs", "--max-logprobs", "-1",
            *v.get("extra_args", [])]


def wait_ready(port: int, proc: subprocess.Popen, timeout_s: float) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"vLLM exited with {proc.returncode} before it was ready")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(5)
    raise SystemExit(f"vLLM not ready on port {port} after {timeout_s:.0f} s")


def _stop(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def _run(cmd: List[str], env: Dict[str, str]) -> None:
    print("+", shlex.join(cmd), flush=True)
    rc = subprocess.call(cmd, env=env)
    if rc:
        raise SystemExit(f"{shlex.join(cmd[:4])} ... exited with {rc}")


def _interrupt(*_: object) -> None:
    raise KeyboardInterrupt


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("overrides", nargs="*")
    p.add_argument("--no-export", action="store_true")
    a = p.parse_args(argv)
    cfg = rl_config.load(a.config, a.overrides)
    out = output_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    py = sys.executable

    # SIGTERM (tmux kill-session, a scheduler) becomes an exception, so `finally` runs.
    signal.signal(signal.SIGTERM, _interrupt)

    _run([py, "-m", "rl.pool", a.config, *a.overrides], env)

    server = trainer = None
    try:
        cmd = vllm_command(cfg)
        print("+", shlex.join(cmd), f"> {out / 'vllm.log'}", flush=True)
        with open(out / "vllm.log", "a") as log:
            server = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**env, "CUDA_VISIBLE_DEVICES": str(cfg["vllm"]["gpu"]),
                     "VLLM_SERVER_DEV_MODE": "1"})
        wait_ready(cfg["vllm"]["port"], server, cfg["vllm"]["ready_timeout_s"])

        gpus = str(cfg["trainer_gpus"])
        cmd = [py, "-m", "accelerate.commands.launch",
               "--num_processes", str(len(gpus.split(","))), "--num_machines", "1",
               "--mixed_precision", "bf16", "--dynamo_backend", "no",
               "-m", "rl.train", a.config, *a.overrides]
        print("+", shlex.join(cmd), flush=True)
        trainer = subprocess.Popen(cmd, start_new_session=True,
                                   env={**env, "CUDA_VISIBLE_DEVICES": gpus})
        if trainer.wait():
            raise SystemExit(f"trainer exited with {trainer.returncode}")
    finally:
        _stop(trainer)
        _stop(server)

    if not a.no_export:
        _run([py, "-m", "rl.export", str(out)], {**env, "CUDA_VISIBLE_DEVICES": ""})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
