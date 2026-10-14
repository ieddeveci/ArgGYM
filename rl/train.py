"""GRPO on ArgGYM with TRL, rollouts from a vLLM server. Run it through `rl.launch`.

    CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes 2 -m rl.train CONFIG [key=value ...]

Every hyperparameter is in the config. `grpo:` goes to `GRPOConfig` as written,
so a field TRL renames fails at start-up rather than being dropped. What this
file adds to TRL's trainer is small and listed in `ArgGymGRPOTrainer`.

Checkpoints are selected on the dev split's strict success,
`eval_arggym_success_rate`, never on reward. With `load_best_model_at_end` the
best checkpoint survives `save_total_limit`; `selection.json` names it when the
run ends, and `rl.export` merges it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from rl import config as rl_config
from rl import pool
from rl.reward import make_reward

SELECTION = "selection.json"


def output_dir(cfg: Dict[str, Any]) -> Path:
    return pool.ROOT / cfg["output_root"] / "runs" / rl_config.run_name(cfg)


def _versions() -> Dict[str, Optional[str]]:
    out = {}
    for name in ("arggym", "trl", "transformers", "peft", "torch", "vllm", "accelerate"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=pool.ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def merged_params(gen):
    """`(vllm_name, tensor)` for every base weight, the LoRA delta added out of place.

    TRL's sync calls PEFT's `merge_adapter()` and `unmerge_adapter()`, which add
    the delta into the BF16 base weight and subtract it again, in place. Each
    round trip rounds twice, so the base weights the trainer keeps drift a
    little at every sync: in a CPU simulation of 300 syncs on one 2560x2560
    layer the drift reached a third of the size of the LoRA update itself. The
    policy being trained is then not the base model plus the adapter, and not
    what `rl.export` would serve. Here the base weight is read and never written,
    and the merged tensor is rounded once, exactly as a merge on export rounds
    it. Covers the DDP case only, where every rank holds whole parameters.
    """
    import torch

    model = gen.model
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, param in model.named_parameters():
            if model.prefix in name or "original_module" in name:
                continue
            tensor = param.data
            if name.endswith(".base_layer.weight"):
                owner = modules[name[: -len(".base_layer.weight")]]
                if not owner.merged and not owner.disable_adapters:
                    delta = sum(owner.get_delta_weight(a).float() for a in owner.active_adapters)
                    tensor = (param.data.float() + delta).to(param.dtype)
            vllm_name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            yield gen._fix_param_name_to_vllm(vllm_name,
                                              extra_prefixes=["modules_to_save.default."]), tensor


def _trainer_class():
    import types

    from accelerate.utils import is_peft_model
    from trl import GRPOTrainer

    class ArgGymGRPOTrainer(GRPOTrainer):
        """TRL's trainer plus four things it does not do.

        - Weight sync to vLLM merges the adapter out of place (`merged_params`).
        - The completions table goes to wandb every `completions_every` steps and
          on every eval, rather than on every logging step.
        - Dev evaluation samples with `eval_sampling` (the eval harness's model-card
          values) instead of the training temperature, so dev success is measured
          the way the eval harness measures it. Eval's IS and loss metrics are
          meaningless under this swap and should be ignored.
        - The share of importance weights the vLLM correction zeroed and their
          effective sample size, which TRL does not log. Under `sequence_mask` a
          long completion whose summed log-prob gap is large loses all its
          gradient, and nothing else on the dashboard says so.
        """

        def __init__(self, *args, completions_every: int, eval_sampling: Dict[str, Any],
                     **kwargs):
            super().__init__(*args, **kwargs)
            self.completions_every = completions_every
            self.eval_sampling = eval_sampling
            gen = self.vllm_generation
            if not is_peft_model(gen.model) or gen._dist.is_fsdp or gen._dist.is_zero3:
                raise SystemExit("rl.train syncs LoRA weights under DDP only")
            gen._iter_named_params = types.MethodType(merged_params, gen)

        def log(self, logs, start_time=None):
            training = self.model.training
            self.log_completions = self.args.log_completions and (
                not training or self.state.global_step % self.completions_every == 0)
            super().log(logs, start_time)

        def evaluate(self, *args, **kwargs):
            gen = self.vllm_generation
            saved = {k: getattr(gen, k) for k in self.eval_sampling}
            for k, v in self.eval_sampling.items():
                setattr(gen, k, v)
            try:
                return super().evaluate(*args, **kwargs)
            finally:
                for k, v in saved.items():
                    setattr(gen, k, v)

        def _generate_and_score_completions(self, inputs):
            import torch

            out = super()._generate_and_score_completions(inputs)
            ratio = out.get("importance_sampling_ratio")
            if ratio is None:
                return out
            mode = "train" if self.model.training else "eval"
            mask = out["completion_mask"].bool()
            if ratio.shape[-1] == 1:  # one weight per sequence
                w = ratio.squeeze(-1)[mask.any(dim=-1)].float()
            else:
                w = ratio[mask].float()
            local = torch.stack([torch.tensor(float(w.numel()), device=w.device),
                                 (w == 0).sum().float(), w.sum(), (w ** 2).sum()])
            n, zeroed, s1, s2 = self.accelerator.gather(local).view(-1, 4).sum(0).tolist()
            if n:
                self._metrics[mode]["sampling/is_zeroed_fraction"].append(zeroed / n)
                self._metrics[mode]["sampling/is_ess"].append(s1 ** 2 / max(n * s2, 1e-12))
            return out

    return ArgGymGRPOTrainer


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args(argv)
    cfg = rl_config.load(a.config, a.overrides)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig

    out = output_dir(cfg)
    name = rl_config.run_name(cfg)
    os.environ.setdefault("WANDB_PROJECT", cfg["wandb_project"])
    # wandb writes its run files under the working directory unless told otherwise.
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(out))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    m = cfg["model"]
    g = dict(cfg["grpo"])

    per_step = g["per_device_train_batch_size"] * world * g["gradient_accumulation_steps"]
    if per_step % g["num_generations"] or per_step // g["num_generations"] != cfg[
            "prompts_per_step"]:
        raise SystemExit(
            f"{g['per_device_train_batch_size']} per device x {world} ranks x "
            f"{g['gradient_accumulation_steps']} accumulation = {per_step} completions, "
            f"which is not {cfg['prompts_per_step']} prompts of {g['num_generations']}. "
            f"Change the batch fields, or prompts_per_step if the change is meant.")

    tokenizer = AutoTokenizer.from_pretrained(m["name"], revision=m["revision"])
    stop_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    elicit = pool.elicitation(cfg["prompt"]["elicitation"])
    template = cfg["prompt"]["template"]
    train = Dataset.from_list(pool.trl_rows(pool.load_split(cfg, "train"), template, elicit))
    dev = Dataset.from_list(pool.trl_rows(pool.load_split(cfg, "dev"), template, elicit))
    n_eval = g.get("num_generations_eval") or g["num_generations"]
    eval_batch = g["per_device_eval_batch_size"] * world
    if len(dev) * n_eval % eval_batch:
        raise SystemExit(
            f"dev split: {len(dev)} rows x {n_eval} samples is not a multiple of the global eval "
            f"batch {eval_batch}; the last batch would be padded with repeated rows and bias "
            f"eval_arggym_success_rate. Change per_device_eval_batch_size.")

    args = GRPOConfig(
        output_dir=str(out), run_name=name, seed=cfg["seed"], report_to=["wandb"],
        model_init_kwargs={"dtype": torch.bfloat16, "revision": m["revision"]},
        chat_template_kwargs={"enable_thinking": cfg["prompt"]["enable_thinking"]},
        use_vllm=True, vllm_mode="server",
        vllm_server_base_url=f"http://127.0.0.1:{cfg['vllm']['port']}",
        metric_for_best_model="eval_arggym_success_rate", greater_is_better=True,
        load_best_model_at_end=True, remove_unused_columns=False,
        **g)
    lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                      lora_dropout=cfg["lora"]["dropout"], bias="none",
                      target_modules=cfg["lora"]["target_modules"], task_type="CAUSAL_LM")

    trainer = _trainer_class()(
        model=m["name"], args=args, train_dataset=train, eval_dataset=dev,
        reward_funcs=make_reward(stop_ids, template), processing_class=tokenizer,
        peft_config=lora, completions_every=cfg["completions_every"],
        eval_sampling=cfg["eval_sampling"])

    # The optimizer updates the adapter, so the adapter is what must be FP32: a
    # BF16 weight near 1.0 cannot take a step smaller than 2^-7.
    trainable = {str(p.dtype) for p in trainer.model.parameters() if p.requires_grad}
    if trainable != {"torch.float32"}:
        raise SystemExit(f"trainable parameters are {trainable}, expected float32 only")

    if trainer.accelerator.is_main_process:
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_name": name, "config": cfg, "argv": sys.argv, "git_sha": _git_sha(),
            "versions": _versions(), "world_size": world,
            "pool": json.loads((pool.pool_dir(cfg) / "summary.json").read_text()),
            "n_train": len(train), "n_dev": len(dev),
            "trainable_params": sum(p.numel() for p in trainer.model.parameters()
                                    if p.requires_grad),
        }
        (out / "run.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        # Started here rather than by TRL's callback, which reuses a live run, so
        # the run carries the whole config and not only the GRPOConfig half.
        import wandb

        wandb.init(project=cfg["wandb_project"], name=name, config={"arggym": manifest})

    trainer.train()

    if trainer.accelerator.is_main_process:
        selection = {"metric": args.metric_for_best_model,
                     "best_metric": trainer.state.best_metric,
                     "best_model_checkpoint": trainer.state.best_model_checkpoint,
                     "global_step": trainer.state.global_step}
        (out / SELECTION).write_text(json.dumps(selection, indent=2) + "\n")
        print(json.dumps(selection, indent=2))
        import wandb

        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
