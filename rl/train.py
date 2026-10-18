"""GRPO on ArgGYM with TRL, rollouts from a vLLM server. Run it through `rl.launch`.

    CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes 2 -m rl.train CONFIG [key=value ...]

Every hyperparameter is in the config. `grpo:` goes to `GRPOConfig` as written,
so a field TRL renames fails at start-up rather than being dropped. What this
file adds to TRL's trainer is small and listed in `ArgGymGRPOTrainer`.

Checkpoints are selected on the dev split's strict success,
`eval_arggym_success_rate`, never on reward; `selection_metric` names another
for `task: gsm8k`, the positive control in `rl.gsm8k`. With
`load_best_model_at_end` the best checkpoint survives `save_total_limit`;
`selection.json` names it when the run ends, beside the base model's step-0
score and whether the checkpoint beats it, and `rl.export` merges it. Dev success is a selection signal over a few
hundred samples, picked as the maximum of many evaluations; quote the selected
checkpoint's score on the eval taskset, not its dev score.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from rl import config as rl_config
from rl import gsm8k, pool
from rl.curriculum import Curriculum, base_pass_counts, group_outcomes, summary
from rl.export import SELECTION, checkpoint_keys
from rl.reward import make_reward

#: Reference PyTorch kernels a run may fall back to. `causal_conv1d` ships no
#: wheel for torch 2.13 and builds only against a CUDA 13 toolkit this machine
#: lacks; its fallback is one cuDNN depthwise conv, measured at 0.6 s of the
#: ~21 s forward and backward of a 46k-token Qwen3.5-4B sequence. Any other
#: fallback (Gated DeltaNet without `fla`: more than ten times slower) stops the
#: run at its first logged step.
ALLOWED_FALLBACKS = ("causal_conv1d",)


class FallbackGuard(logging.Handler):
    """Records transformers' "falling back to its reference PyTorch implementation" warnings."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.seen: list = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "falling back to its reference PyTorch" in msg and not any(
                f"because `{name}` is not installed" in msg for name in ALLOWED_FALLBACKS):
            self.seen.append(msg)

    def check(self) -> None:
        if self.seen:
            raise RuntimeError(f"slow reference kernel in use: {self.seen[0]}")


@contextlib.contextmanager
def dev_sampling_swap(trainer: Any, gen: Any, sampling: Dict[str, Any]):
    """Dev evaluation's sampling on `gen`, and the training sampling back afterwards.

    `presence_penalty` (Qwen3.5's card asks for 1.5) goes into
    `generation_kwargs` for the duration of the eval, for configs whose training
    rollouts run without it (`check_penalties`). The
    importance-sampling correction is off during eval, which with aligned
    accumulation skips the trainer's old-log-prob forward pass.
    """
    sampling = dict(sampling)
    penalty = sampling.pop("presence_penalty", None)
    swap = [(gen, k, v) for k, v in sampling.items()]
    if penalty:
        swap.append((gen, "generation_kwargs",
                     {**(gen.generation_kwargs or {}), "presence_penalty": penalty}))
    swap.append((trainer, "vllm_importance_sampling_correction", False))
    saved = [(obj, k, getattr(obj, k)) for obj, k, _ in swap]
    for obj, k, v in swap:
        setattr(obj, k, v)
    try:
        yield
    finally:
        for obj, k, v in saved:
            setattr(obj, k, v)


def output_dir(cfg: Dict[str, Any]) -> Path:
    return pool.ROOT / cfg["output_root"] / "runs" / rl_config.run_name(cfg)


def select(state: Any, metric: str) -> Dict[str, Any]:
    """The selected checkpoint, and whether it beats the base model on the same metric.

    Transformers never compares the step-0 evaluation against the checkpoints,
    so the first checkpoint saved is "best" against minus infinity even when
    every checkpoint is worse than the model training started from. The base
    score is read back from the log history instead.
    """
    base = next((e[metric] for e in state.log_history
                 if e.get("step") == 0 and metric in e), None)
    best = state.best_metric
    return {"metric": metric, "best_metric": best,
            "best_model_checkpoint": state.best_model_checkpoint,
            "base_metric": base,
            "improved_over_base": (None if base is None or best is None else best > base),
            "global_step": state.global_step}


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


def check_penalties(grpo: Dict[str, Any], eval_presence: float) -> None:
    """Refuse a sampling penalty on training rollouts other than the eval's own.

    vLLM returns the penalised distribution's log-probs and the trainer computes
    the unpenalised one; the truncated importance weights correct the gap toward
    the policy the trainer optimises. A presence penalty is allowed only at the
    eval config's value, so training samples the policy the eval measures:
    without it, Qwen3.5-4B looped into the length cap on about half its rollouts.
    Frequency and repetition penalties are refused.
    """
    extra = grpo.get("generation_kwargs") or {}
    presence = extra.get("presence_penalty", grpo.get("presence_penalty", 0.0))
    if presence not in (0.0, eval_presence):
        raise SystemExit(f"grpo sets presence_penalty {presence}; the eval config uses "
                         f"{eval_presence}")
    for key, neutral in (("frequency_penalty", 0.0), ("repetition_penalty", 1.0)):
        if extra.get(key, neutral) != neutral or grpo.get(key, neutral) != neutral:
            raise SystemExit(f"grpo sets {key}; training rollouts take no {key}")


def lora_targets(model: Any) -> Dict[str, int]:
    """How many modules of each kind the adapter trains, by the module's last name."""
    out: Dict[str, int] = {}
    for name, _ in model.named_parameters():
        if ".lora_A." in name:
            kind = name.split(".lora_A.")[0].rsplit(".", 1)[-1]
            out[kind] = out.get(kind, 0) + 1
    return dict(sorted(out.items()))


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

    Names are the base checkpoint's (`rl.export.to_checkpoint`), which is what
    vLLM's loader maps from: Qwen3.5 trains as a text-only model whose tensors
    the checkpoint, a vision-language one, keeps under `model.language_model.`.
    A tensor with no checkpoint name stops the sync instead of being dropped.

    The rounding to BF16 drops any delta smaller than half the BF16 spacing of
    the weight it lands on: at LR 1e-5, 40 steps in, 7.9% of the adapted
    weights had changed and the served model carried 23% of the update.
    With `vllm.dtype: float32` the server holds FP32 weights and the adapted
    tensors are sent in FP32, so nothing is rounded away.
    `gen.sync_stats` keeps, for the last sync, how many adapted elements moved
    and the share of the delta that survived the rounding (its projection on
    the intended delta), which the trainer logs as `sync/*`.
    """
    import torch

    from rl.export import to_checkpoint

    model = gen.model
    keys = gen.checkpoint_keys
    tied = bool(model.config.get_text_config().tie_word_embeddings)
    modules = dict(model.named_modules())
    moved = total = served = intended = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if model.prefix in name or "original_module" in name:
                continue
            tensor = param.data
            if name.endswith(".base_layer.weight"):
                owner = modules[name[: -len(".base_layer.weight")]]
                if not owner.merged and not owner.disable_adapters:
                    delta = sum(owner.get_delta_weight(a).float() for a in owner.active_adapters)
                    tensor = (param.data.float() + delta).to(getattr(gen, "sync_dtype", None)
                                                              or param.dtype)
                    kept = tensor.float() - param.data.float()
                    moved += (kept != 0).sum().item()
                    total += kept.numel()
                    served += (kept * delta).sum().item()
                    intended += (delta * delta).sum().item()
            plain = gen._fix_param_name_to_vllm(
                name.removeprefix("base_model.model.").replace(".base_layer", ""),
                extra_prefixes=["modules_to_save.default."])
            ckpt = to_checkpoint(plain, keys)
            if ckpt is None:
                if plain == "lm_head.weight" and tied:
                    continue
                raise RuntimeError(f"{plain} has no tensor in the base checkpoint; vLLM "
                                   f"would not load it")
            yield ckpt, tensor
    if total:
        gen.sync_stats = {"sync/moved_fraction": moved / total}
        if intended:
            gen.sync_stats["sync/served_share"] = served / intended


def build_curriculum(cfg: Dict[str, Any], train_rows: Sequence[Dict[str, Any]]) -> Curriculum:
    """The sampler `curriculum:` describes, over the training rows' (task, level) cells."""
    c, g = cfg["curriculum"], cfg["grpo"]
    if g.get("num_iterations", 1) != 1 or g.get("steps_per_generation") or g.get(
            "generation_batch_size"):
        raise SystemExit("the curriculum draws one block per optimizer step; it needs "
                         "num_iterations 1 and the default steps_per_generation")
    cells: Dict[Any, list] = {}
    for i, r in enumerate(train_rows):
        cells.setdefault((r["task"], r["level"]), []).append(i)
    return Curriculum(cells, base_pass_counts(str(pool.ROOT / c["prior"])),
                      prompts_per_step=cfg["prompts_per_step"],
                      num_generations=g["num_generations"],
                      steps_per_generation=g["gradient_accumulation_steps"], seed=cfg["seed"],
                      half_life=c["half_life"], floor=c["floor"])


def _save_curriculum(curriculum: Curriculum):
    """A callback that saves the curriculum's state into every checkpoint."""
    from transformers import TrainerCallback

    class SaveCurriculum(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                curriculum.save(ckpt, state.global_step)

    return SaveCurriculum()


def signal_tokens(advantages: Any, completion_mask: Any) -> Any:
    """Loss tokens of the completions with a nonzero advantage.

    A group whose scored completions all got the same reward, all 0 or all 1,
    has zero advantage everywhere and so no gradient, as does a completion
    whose reward is `None`. Counting their tokens in DAPO's denominator would
    scale every update down by the share of such tokens.
    """
    return (completion_mask * (advantages != 0).unsqueeze(1)).sum()


def _trainer_class():
    import types

    from accelerate.utils import is_peft_model
    from trl import GRPOTrainer

    class ArgGymGRPOTrainer(GRPOTrainer):
        """TRL's trainer plus what it does not do.

        - Weight sync to vLLM merges the adapter out of place (`merged_params`).
        - The completions table goes to trackio every `completions_every` steps and
          on every eval, rather than on every logging step. A training step that
          is also an eval step logs only the eval table: TRL writes both to
          `completions_<step>.parquet` and the tracker key `completions`, so the
          second would overwrite the first.
        - Dev evaluation is generation and scoring only, under `dev_sampling`:
          the eval config's temperature, top-p, top-k, min-p, penalties and token
          cap (`dev_sampling_swap`). TRL
          would also run two trainer forward passes over every eval batch, the
          importance-sampling one and the loss one; both measure nothing at a
          sampling temperature the trainer does not divide by, and at a 32k cap
          they are the largest forward passes of the run. Without them the eval
          batch size sets only how many sequences vLLM gets at once.
        - The share of importance weights the vLLM correction zeroed and their
          effective sample size, which TRL does not log.
        - With `curriculum`, training prompts come from `rl.curriculum`, which
          learns from every step's gathered rewards and is logged every step.
        - The loss is normalized over the tokens of completions with a nonzero
          advantage (`signal_tokens`), so groups that were all 0 or all 1 do
          not dilute the update. They still go through forward and backward;
          dropping them would leave ranks with uneven batches.
        """

        def __init__(self, *args, completions_every: int, dev_sampling: Dict[str, Any],
                     checkpoint_keys: set, guard: FallbackGuard, sync_dtype: Any = None,
                     curriculum: Optional[Curriculum] = None, curriculum_log: Any = None,
                     table_every: int = 5, **kwargs):
            self.curriculum = curriculum
            self.curriculum_log = curriculum_log
            self.table_every = table_every
            super().__init__(*args, **kwargs)
            self.completions_every = completions_every
            self.dev_sampling = dev_sampling
            self.guard = guard
            gen = self.vllm_generation
            if not is_peft_model(gen.model) or gen._dist.is_fsdp or gen._dist.is_zero3:
                raise SystemExit("rl.train syncs LoRA weights under DDP only")
            gen.checkpoint_keys = checkpoint_keys
            gen.sync_dtype = sync_dtype
            gen._iter_named_params = types.MethodType(merged_params, gen)

        def log(self, logs, start_time=None):
            import torch

            self.guard.check()
            step = self.state.global_step
            if self.model.training:
                # Rank 0's peak since the previous log: whether a longer cap or a
                # larger micro-batch would fit.
                logs["memory/peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                torch.cuda.reset_peak_memory_stats()
                eval_step = self.args.eval_steps and step % self.args.eval_steps == 0
                self.log_completions = self.args.log_completions and (
                    step % self.completions_every == 0 and not eval_step)
                logs.update(getattr(self.vllm_generation, "sync_stats", {}))
                if self.curriculum is not None:
                    logs.update(self._log_curriculum(step))
            else:
                self.log_completions = self.args.log_completions
            super().log(logs, start_time)

        def _log_curriculum(self, step: int) -> Dict[str, float]:
            """The step's draw and outcomes; every `table_every` steps, every cell's estimate."""
            draw = self.curriculum.draws.get(step - 1)
            if not self.accelerator.is_main_process or draw is None:
                return summary(draw)
            import trackio

            with open(self.curriculum_log, "a") as f:
                f.write(json.dumps({"step": step, **draw}) + "\n")
            cols = ["step", "task", "level", "pass_rate", "signal"]
            tables = {"curriculum/drawn": trackio.Table(
                columns=cols, data=[[step] + [c[k] for k in cols[1:]] for c in draw["cells"]])}
            if step % self.table_every == 0 or step == 1:
                rows = self.curriculum.table()
                tables["curriculum/cells"] = trackio.Table(
                    columns=list(rows[0]), data=[list(r.values()) for r in rows])
            trackio.log(tables, step=step)
            return summary(draw)

        def _get_train_sampler(self, dataset=None):
            if self.curriculum is not None:
                return self.curriculum
            return super()._get_train_sampler(dataset)

        def evaluate(self, *args, **kwargs):
            with dev_sampling_swap(self, self.vllm_generation, self.dev_sampling):
                return super().evaluate(*args, **kwargs)

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            import torch

            self._prepare_inputs(inputs)  # generates, scores and logs the metrics
            return torch.zeros((), device=self.accelerator.device), None, None

        def _generate_and_score_completions(self, inputs):
            import torch

            out = super()._generate_and_score_completions(inputs)
            if self.curriculum is not None and self.model.training:
                # Every rank, the same gathered batch, once per optimizer step.
                self.curriculum.update(group_outcomes(self.reward_funcs[0].last_batch))
            if self.model.training:
                total = out["num_items_in_batch"]
                out["num_items_in_batch"] = self.accelerator.gather(
                    signal_tokens(out["advantages"], out["completion_mask"])).sum()
                self._metrics["train"]["sampling/signal_token_fraction"].append(
                    (out["num_items_in_batch"] / total.clamp(min=1)).item())
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
    p.add_argument("--resume", default=None,
                   help="a checkpoint directory of this run to continue from")
    a = p.parse_args(argv)
    cfg = rl_config.load(a.config, a.overrides)
    # Trackio stores every run of every project in one local SQLite directory,
    # read at import, so this comes before anything that imports it.
    # `trackio show --project <project>` opens the dashboard on it.
    os.environ.setdefault("TRACKIO_DIR", str(pool.ROOT / cfg["output_root"] / "trackio"))

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig

    guard = FallbackGuard()
    logging.getLogger("transformers").addHandler(guard)

    out = output_dir(cfg)
    name = rl_config.run_name(cfg)
    out.mkdir(parents=True, exist_ok=True)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    m = cfg["model"]
    g = dict(cfg["grpo"])
    check_penalties(g, rl_config.dev_sampling(cfg)["presence_penalty"])
    if g.get("beta", 0.0) != 0.0:
        raise SystemExit("grpo sets beta; the KL term would move completions that "
                         "signal_tokens leaves out of the loss's denominator")
    text_config = AutoConfig.from_pretrained(m["name"], revision=m["revision"]).get_text_config()
    if "linear_attention" in (getattr(text_config, "layer_types", None) or []):
        try:
            import fla.ops.gated_delta_rule  # noqa: F401
        except ImportError as e:
            raise SystemExit(f"{m['name']} has Gated DeltaNet layers and fla does not "
                             f"import ({e}); transformers would run them ten times slower")

    per_step = g["per_device_train_batch_size"] * world * g["gradient_accumulation_steps"]
    if per_step % g["num_generations"] or per_step // g["num_generations"] != cfg[
            "prompts_per_step"]:
        raise SystemExit(
            f"{g['per_device_train_batch_size']} per device x {world} ranks x "
            f"{g['gradient_accumulation_steps']} accumulation = {per_step} completions, "
            f"which is not {cfg['prompts_per_step']} prompts of {g['num_generations']}. "
            f"Change the batch fields, or prompts_per_step if the change is meant.")

    tokenizer = AutoTokenizer.from_pretrained(m["name"], revision=m["revision"])
    # TRL decodes completions with `parse_response` when the tokenizer carries a
    # response template, which moves the think block out of `content` and would
    # hide an answer left in the reasoning from the reward, while the eval
    # harness still finds it there.
    for attr in ("response_template", "response_schema"):
        if getattr(tokenizer, attr, None) is not None:
            raise SystemExit(f"the tokenizer sets {attr}; rl.reward reads raw content")
    stop_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    template_kwargs = rl_config.eval_model(cfg)["chat_template_kwargs"]
    dev_sampling = rl_config.dev_sampling(cfg)
    task = cfg.get("task", "arggym")
    if task == "gsm8k":
        train_split, dev_split = gsm8k.load_splits(cfg["data"])
        train_rows, dev_rows = gsm8k.trl_rows(train_split), gsm8k.trl_rows(dev_split)
        reward = gsm8k.make_reward(stop_ids)
        data_summary = {"data": cfg["data"]}
    elif task == "arggym":
        elicit = pool.elicitation(cfg["prompt"]["elicitation"])
        template = cfg["prompt"]["template"]
        train_rows = pool.trl_rows(pool.load_split(cfg, "train"), template, elicit)
        dev_rows = pool.trl_rows(pool.load_split(cfg, "dev"), template, elicit)
        reward = make_reward(stop_ids, template)
        data_summary = json.loads((pool.pool_dir(cfg) / "summary.json").read_text())
    else:
        raise SystemExit(f"task {task!r}: rl.train knows arggym and gsm8k")
    texts = [tokenizer.apply_chat_template(r["prompt"], add_generation_prompt=True,
                                           tokenize=False, **template_kwargs)
             for r in train_rows + dev_rows]
    longest = max(len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"])
    need = longest + max(g["max_completion_length"], dev_sampling["max_completion_length"])
    if need > cfg["vllm"]["max_model_len"]:
        raise SystemExit(f"vllm.max_model_len {cfg['vllm']['max_model_len']} is below the "
                         f"longest prompt ({longest}) plus the larger completion cap: {need}")
    train, dev = Dataset.from_list(train_rows), Dataset.from_list(dev_rows)
    n_eval = g.get("num_generations_eval") or g["num_generations"]
    eval_batch = g["per_device_eval_batch_size"] * world
    if len(dev) * n_eval % eval_batch:
        raise SystemExit(
            f"dev split: {len(dev)} rows x {n_eval} samples is not a multiple of the global eval "
            f"batch {eval_batch}; the last batch would be padded with repeated rows and bias "
            f"the dev success rate. Change per_device_eval_batch_size.")

    args = GRPOConfig(
        output_dir=str(out), run_name=name, seed=cfg["seed"], report_to=["trackio"],
        project=cfg["project"],
        chat_template_kwargs=template_kwargs,
        use_vllm=True, vllm_mode="server",
        vllm_server_base_url=f"http://127.0.0.1:{cfg['vllm']['port']}",
        metric_for_best_model=cfg.get("selection_metric", "eval_arggym_success_rate"),
        greater_is_better=True,
        load_best_model_at_end=True, remove_unused_columns=False,
        # On resume Transformers would replay the consumed batches to skip them,
        # which would advance the curriculum; its state is restored instead.
        ignore_data_skip=bool(cfg.get("curriculum")),
        **g)
    lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                      lora_dropout=cfg["lora"]["dropout"], bias="none",
                      target_modules=cfg["lora"]["target_modules"], task_type="CAUSAL_LM")

    # The text-only class, also for a vision-language checkpoint such as Qwen3.5's:
    # the vision tower is never used, and all-linear LoRA would adapt it too. Its
    # tensor names differ from the checkpoint's; `merged_params` maps them.
    model = AutoModelForCausalLM.from_pretrained(m["name"], revision=m["revision"],
                                                 dtype=torch.bfloat16)
    curriculum = build_curriculum(cfg, train_rows) if cfg.get("curriculum") else None
    if curriculum is not None and a.resume:
        curriculum.load(Path(a.resume))
    trainer = _trainer_class()(
        model=model, args=args, train_dataset=train, eval_dataset=dev,
        reward_funcs=reward, processing_class=tokenizer,
        peft_config=lora, completions_every=cfg["completions_every"],
        dev_sampling=dev_sampling, checkpoint_keys=checkpoint_keys(m["name"], m["revision"]),
        sync_dtype=getattr(torch, cfg["vllm"].get("dtype", "bfloat16")),
        guard=guard, curriculum=curriculum, curriculum_log=out / "curriculum.jsonl",
        table_every=(cfg.get("curriculum") or {}).get("table_every", 5))
    if curriculum is not None:
        trainer.add_callback(_save_curriculum(curriculum))

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
            "pool": data_summary,
            "n_train": len(train), "n_dev": len(dev),
            "trainable_params": sum(p.numel() for p in trainer.model.parameters()
                                    if p.requires_grad),
            "lora_targets": lora_targets(trainer.model),
        }
        (out / "run.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        # Started here so the run carries the whole config and not only the
        # GRPOConfig half; Transformers' callback resumes it by name.
        import trackio

        trackio.init(project=cfg["project"], name=name, config={"arggym": manifest},
                     resume="allow")

    trainer.train(resume_from_checkpoint=a.resume)

    if trainer.accelerator.is_main_process:
        selection = select(trainer.state, args.metric_for_best_model)
        (out / SELECTION).write_text(json.dumps(selection, indent=2) + "\n")
        print(json.dumps(selection, indent=2))
        import trackio

        trackio.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
