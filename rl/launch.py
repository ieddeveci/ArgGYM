"""One command for a run: pool, vLLM on one GPU, the trainer on the others, export.

    uv run --group rl python -m rl.launch rl/configs/smoke.yaml
    uv run --group rl python -m rl.launch rl/configs/qwen3-4b.yaml grpo.learning_rate=2e-5

In order: build the pool if `data` names one not built yet; start `vllm serve`
on `vllm.gpu` and wait for it; run `accelerate launch -m rl.train` on
`trainer_gpus`; merge the selected checkpoint and check it (`rl.export`). The
server is torn down on every exit path, Ctrl-C and SIGTERM included, by killing
its whole process group: vLLM forks an engine process that would otherwise
outlive the parent and hold the GPU.

`--no-export` stops after training. Logs go to `<run>/vllm-<start time>.log` and the
terminal.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from rl import config as rl_config
from rl.train import SELECTION, output_dir


def vllm_command(cfg: Dict[str, Any]) -> List[str]:
    """The server TRL's client expects (`trl vllm-serve`'s `build_command`, TRL 1.15).

    `--logprobs-mode processed_logprobs` makes the returned log-probs those of
    the distribution actually sampled from. The weight-transfer config enables
    the NCCL endpoint the trainer pushes weights through; it needs
    `VLLM_SERVER_DEV_MODE=1`, which also exposes the prefix-cache reset the
    trainer calls after every sync. The served name is the eval config's `model`,
    so the eval harness can query this server, or one serving the export, under
    the name it already sends.
    """
    v, m = cfg["vllm"], cfg["model"]
    return [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", m["name"],
            "--served-model-name", rl_config.eval_model(cfg)["model"],
            "--revision", m["revision"], "--dtype", "bfloat16",
            "--host", "127.0.0.1", "--port", str(v["port"]),
            "--max-model-len", str(v["max_model_len"]),
            "--gpu-memory-utilization", str(v["gpu_memory_utilization"]),
            "--weight-transfer-config", '{"backend": "nccl"}',
            "--logprobs-mode", "processed_logprobs", "--max-logprobs", "-1",
            *v.get("extra_args", [])]


def vllm_env(cfg: Dict[str, Any], env: Dict[str, str]) -> Dict[str, str]:
    """The server's environment.

    FlashInfer's top-k/top-p sampler compiles itself at first use with the
    system `nvcc` and `ninja`. On this machine `/usr/bin/nvcc` is too old for its
    flags (`Unknown option '--compress-mode=size'`), so dev evaluation, the first
    request with top-k on, would crash the server; vLLM's own sampler is used
    instead. The venv's `bin` goes first on PATH because that is where `ninja`
    is when anything else does compile.
    """
    bin_dir = os.path.dirname(sys.executable)
    return {**env, "CUDA_VISIBLE_DEVICES": str(cfg["vllm"]["gpu"]),
            "VLLM_SERVER_DEV_MODE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "PATH": os.pathsep.join([bin_dir, env.get("PATH", "")])}


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
        # One log per launch, so a rerun of the same config does not append to
        # the previous run's server log.
        log_path = out / f"vllm-{time.strftime('%Y%m%d-%H%M%S')}.log"
        print("+", shlex.join(cmd), f"> {log_path}", flush=True)
        with open(log_path, "w") as log:
            server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, env=vllm_env(cfg, env))
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

    selection = json.loads((out / SELECTION).read_text())
    print(f"selected {selection['best_model_checkpoint']}: {selection['metric']} "
          f"{selection['best_metric']} against the base model's {selection['base_metric']} "
          f"(improved_over_base={selection['improved_over_base']})", flush=True)
    if not a.no_export:
        _run([py, "-m", "rl.export", str(out)], {**env, "CUDA_VISIBLE_DEVICES": ""})
        print(f"serve it for the eval harness: vllm serve {out / 'merged'} "
              f"--served-model-name {rl_config.eval_model(cfg)['model']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
