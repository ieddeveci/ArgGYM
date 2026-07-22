"""Emit shell-evaluable serve settings for one model config.

serve.sh sources this so the docker flags and the runner's sampling params come
from the same YAML file and cannot drift apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir


def main() -> None:
    model = sys.argv[1]
    conf_dir = str(Path(__file__).resolve().parent / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config", overrides=[f"model={model}"])

    m = cfg.model
    args = [
        "--port", "8900",
        "--model", str(m.model_path),
        "--served-model-name", str(m.name),
        "--gpu-memory-utilization", str(m.gpu_memory_utilization),
        # Per-model override: a model whose KV footprint per token is large (e.g.
        # gemma-4) can only hold a workable number of concurrent requests if its
        # max sequence length is capped nearer to what it actually generates.
        "--max-model-len", str(m.get("max_model_len") or 65536),
        "--data-parallel-size", str(cfg.serve.data_parallel_size),
    ]
    if m.get("reasoning_parser"):
        args += ["--reasoning-parser", str(m.reasoning_parser)]
    args += [str(x) for x in (m.get("extra_args") or [])]

    print(f"MODEL_NAME={m.name}")
    print(f"IMAGE={m.image}")
    print(f"GPUS={cfg.serve.gpus}")
    print(f"VLLM_ARGS='{' '.join(args)}'")


if __name__ == "__main__":
    main()
