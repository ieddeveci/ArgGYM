"""Count how often the base model solves each GSM8K question, to keep the ones it sometimes does.

    CUDA_VISIBLE_DEVICES=0 python -m rl.gsm8k_screen rl/configs/gsm8k-hard-qwen3-0.6b.yaml --shard 0/3

A question the base model always solves, or never does, gives a group of
identical rewards and no gradient; on plain GSM8K Qwen3-0.6B scores about 0.8
and two thirds of the groups are such. Each question of `train` and `test` is
sampled `screen.samples` times as training samples it (the config's
`temperature`, `top_p`, `max_completion_length`, the eval model's chat-template
arguments) and scored by the training reward. One JSON line per question,
`{"id", "passes", "samples"}`, goes to `<screen.path>/shard-<i>.jsonl`;
`rl.gsm8k.load_splits` reads the shards and keeps the questions whose passes
lie in `screen.passes`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rl import config as rl_config
from rl import gsm8k
from rl.launch import vllm_env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--shard", default="0/1", help="i/n: this process screens every n-th question")
    a = ap.parse_args()
    cfg = rl_config.load(a.config)
    i, n = (int(x) for x in a.shard.split("/"))
    # The server's sampler and PATH (`rl.launch.vllm_env`); its GPU is CUDA_VISIBLE_DEVICES's.
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ.update(vllm_env(cfg, dict(os.environ)), CUDA_VISIBLE_DEVICES=gpu)
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    data, m, g, s = cfg["data"], cfg["model"], cfg["grpo"], cfg["data"]["screen"]
    ds = load_dataset(data["dataset"], data["config"], revision=data["revision"])
    rows = [{"id": f"{split}-{k}", **r} for split in ("train", "test")
            for k, r in enumerate(ds[split])][i::n]
    llm = LLM(m["name"], revision=m["revision"], dtype="bfloat16", seed=cfg["seed"] + i,
              max_model_len=cfg["vllm"]["max_model_len"], gpu_memory_utilization=0.9)
    params = SamplingParams(n=s["samples"], temperature=g["temperature"], top_p=g["top_p"],
                            max_tokens=g["max_completion_length"], seed=cfg["seed"] + i)
    prompts = [r["prompt"] for r in gsm8k.trl_rows(rows)]
    outs = llm.chat(prompts, params,
                    chat_template_kwargs=rl_config.eval_model(cfg)["chat_template_kwargs"])
    path = Path(s["path"])
    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"shard-{i}.jsonl", "w") as f:
        for r, out in zip(rows, outs, strict=True):
            # As the training reward: a truncated completion pays 0.
            passes = sum(c.finish_reason == "stop" and gsm8k.correct(c.text, gsm8k.gold(r["answer"]))
                         for c in out.outputs)
            f.write(json.dumps({"id": r["id"], "passes": passes, "samples": len(out.outputs)}) + "\n")


if __name__ == "__main__":
    main()
