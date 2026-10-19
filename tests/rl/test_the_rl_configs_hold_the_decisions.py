"""The configs say what was decided, and their batch arithmetic closes.

The first-run config is the record of the choices made for it; a change to one
of them should be a visible edit to this test, not a quiet edit to a YAML file.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rl import config as rl_config

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ("rl/configs/qwen3-4b.yaml", "rl/configs/smoke.yaml", "rl/configs/qwen3.5-4b.yaml",
           "rl/configs/qwen3.5-4b-curriculum.yaml", "rl/configs/gsm8k-qwen3-0.6b.yaml",
           "rl/configs/gsm8k-hard-qwen3-0.6b.yaml", "rl/configs/gsm8k-hard-qwen3-0.6b-adapter.yaml")
REAL = (CONFIGS[0], CONFIGS[2], CONFIGS[3], CONFIGS[4], CONFIGS[5], CONFIGS[6])
#: The longest prompt in data/taskset.jsonl: 4,066 tokens with the Qwen3
#: tokenizer, 5,336 with Qwen3.5's (their eval configs). rl.train measures the
#: real number at start-up and refuses a context that does not fit. Qwen3-0.6B
#: trains on GSM8K only, whose longest prompt (train and test) is 235 tokens.
LONGEST_PROMPT = {"Qwen/Qwen3-4B": 4066, "Qwen/Qwen3.5-4B": 5336, "Qwen/Qwen3-0.6B": 235}


def test_the_qwen35_config_holds_the_decisions():
    cfg = rl_config.load(str(ROOT / "rl/configs/qwen3.5-4b.yaml"))
    g = cfg["grpo"]
    assert cfg["model"] == {"name": "Qwen/Qwen3.5-4B",
                            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}
    assert cfg["eval_model"] == "hf-qwen3.5-4b"
    assert rl_config.eval_model(cfg)["chat_template_kwargs"] == {"enable_thinking": True}
    assert cfg["data"]["tasks"] == "all" and cfg["data"]["levels"] == [1, 2, 3, 4]
    assert (cfg["prompts_per_step"], g["num_generations"]) == (8, 8)
    assert (g["per_device_train_batch_size"], g["gradient_accumulation_steps"]) == (1, 32)
    assert g["max_completion_length"] == 40960 and g["gradient_checkpointing"]
    assert (g["temperature"], g["top_p"], g["top_k"]) == (1.0, 0.95, 20)
    assert "generation_kwargs" not in g and "repetition_penalty" not in g
    assert (g["learning_rate"], g["lr_scheduler_type"]) == (1e-5, "constant_with_warmup")
    assert (g["loss_type"], g["beta"], g["epsilon"], g["epsilon_high"]) == ("dapo", 0.0, 0.2, 0.28)
    assert g["scale_rewards"] == "none" and g["mask_truncated_completions"]
    assert (g["eval_steps"], g["save_steps"], g["max_steps"]) == (20, 20, 100)
    assert cfg["data"]["dev"]["per_task_level"] == 1 and g["num_generations_eval"] == 4
    assert cfg["lora"] == {"r": 32, "alpha": 32, "dropout": 0.0, "target_modules": "all-linear"}
    dev = rl_config.dev_sampling(cfg)
    assert (dev["presence_penalty"], dev["max_completion_length"]) == (1.5, 65536)
    # A collective must outlast the longest generation rank 0 waits on.
    assert g["ddp_timeout"] >= 3 * 3600


def test_the_gsm8k_control_holds_the_decisions():
    cfg = rl_config.load(str(ROOT / "rl/configs/gsm8k-qwen3-0.6b.yaml"))
    first = rl_config.load(str(ROOT / "rl/configs/qwen3-4b.yaml"))
    g, f = cfg["grpo"], first["grpo"]
    assert cfg["task"] == "gsm8k" and cfg["selection_metric"] == "eval_gsm8k_success_rate"
    assert cfg["model"] == {"name": "Qwen/Qwen3-0.6B",
                            "revision": "c1899de289a04d12100db370d81485cdf75e47ca"}
    assert rl_config.eval_model(cfg)["chat_template_kwargs"] == {"enable_thinking": True}
    assert (cfg["data"]["dataset"], cfg["data"]["config"]) == ("openai/gsm8k", "main")
    assert cfg["data"]["dev_size"] == 200 and "curriculum" not in cfg
    # The loss, IS, LoRA and rollout sampling of the first run, so the control
    # tests that pipeline; LR and batch follow TRL's Qwen3-0.6B LoRA reproduction.
    assert cfg["lora"] == first["lora"]
    for key in ("loss_type", "beta", "epsilon", "epsilon_high", "scale_rewards",
                "importance_sampling_level",
                "vllm_importance_sampling_correction", "num_generations", "num_iterations",
                "ddp_find_unused_parameters",
                "temperature", "top_p", "top_k", "lr_scheduler_type"):
        assert g[key] == f[key], key
    assert "generation_kwargs" not in g and "presence_penalty" not in g
    # Truncation pays 0 and is trained on; masking it rewarded never finishing.
    assert g["mask_truncated_completions"] is False
    assert (g["learning_rate"], cfg["prompts_per_step"]) == (2e-5, 16)
    # The server holds FP32 weights so no part of the update rounds away.
    assert cfg["vllm"]["dtype"] == "float32"
    assert g["max_completion_length"] == 8192
    assert (g["eval_steps"], g["save_steps"], g["max_steps"]) == (10, 10, 200)
    assert (cfg["vllm"]["gpu"], cfg["trainer_gpus"]) == (0, "1,2")
    # An override reaches the learning rate.
    assert rl_config.load(str(ROOT / "rl/configs/gsm8k-qwen3-0.6b.yaml"),
                          ["grpo.learning_rate=1e-4"])["grpo"]["learning_rate"] == 1e-4


def test_an_arggym_config_names_no_task_and_selects_on_arggym_success():
    for path in CONFIGS[:4]:
        cfg = rl_config.load(str(ROOT / path))
        assert "task" not in cfg and "selection_metric" not in cfg


@pytest.mark.parametrize("path", CONFIGS)
def test_vllm_runs_with_the_flags_the_eval_lane_serves_with(path):
    cfg = rl_config.load(str(ROOT / path))
    lane = yaml.safe_load(
        (ROOT / "evals/conf/model" / f"{cfg['eval_model']}.yaml").read_text())
    assert set(lane.get("extra_args") or []) <= set(cfg["vllm"]["extra_args"])


def test_the_first_run_config_holds_the_decisions():
    cfg = rl_config.load(str(ROOT / "rl/configs/qwen3-4b.yaml"))
    g = cfg["grpo"]
    assert cfg["model"]["name"] == "Qwen/Qwen3-4B"
    assert cfg["eval_model"] == "hf-qwen3-4b"
    assert rl_config.eval_model(cfg)["chat_template_kwargs"] == {"enable_thinking": True}
    assert cfg["lora"] == {"r": 32, "alpha": 32, "dropout": 0.0, "target_modules": "all-linear"}
    assert (g["learning_rate"], g["lr_scheduler_type"]) == (1e-5, "constant_with_warmup")
    assert (g["loss_type"], g["beta"], g["epsilon"], g["epsilon_high"]) == ("dapo", 0.0, 0.2, 0.28)
    assert (g["num_generations"], cfg["prompts_per_step"], g["scale_rewards"]) == (8, 32, "none")
    assert (g["temperature"], g["top_p"], g["top_k"]) == (1.0, 1.0, 0)
    assert g["mask_truncated_completions"] and g["vllm_importance_sampling_correction"]
    assert g["max_completion_length"] == 16384
    assert cfg["data"]["levels"] == list(range(1, 9))
    assert cfg["project"] == "arggym-rl"
    assert g["save_total_limit"] <= 2


@pytest.mark.parametrize("path", CONFIGS)
def test_importance_sampling_is_truncated_per_token(path):
    g = rl_config.load(str(ROOT / path))["grpo"]
    assert g["vllm_importance_sampling_mode"] == "token_truncate"
    assert g["vllm_importance_sampling_clip_max"] == 2.0


@pytest.mark.parametrize("path", REAL)
def test_a_real_run_evaluates_dev_at_the_eval_cap(path):
    cfg = rl_config.load(str(ROOT / path))
    assert "dev_max_tokens" not in cfg
    cap = rl_config.dev_sampling(cfg)["max_completion_length"]
    assert cap == rl_config.eval_model(cfg)["max_tokens"]


@pytest.mark.parametrize("path", CONFIGS)
def test_the_batch_arithmetic_closes_on_two_trainer_gpus(path):
    cfg = rl_config.load(str(ROOT / path))
    g = cfg["grpo"]
    world = len(str(cfg["trainer_gpus"]).split(","))
    assert world == 2 and str(cfg["vllm"]["gpu"]) not in str(cfg["trainer_gpus"]).split(",")
    completions = g["per_device_train_batch_size"] * world * g["gradient_accumulation_steps"]
    assert completions == cfg["prompts_per_step"] * g["num_generations"]
    if cfg.get("task") == "gsm8k":
        dev_rows = cfg["data"]["dev_size"]
    else:
        dev_rows = 12 * len(cfg["data"]["levels"]) * cfg["data"]["dev"]["per_task_level"]
    assert dev_rows * g["num_generations_eval"] % (g["per_device_eval_batch_size"] * world) == 0
    # A checkpoint at every eval and an eval at every checkpoint.
    assert g["save_steps"] == g["eval_steps"]
    longest_cap = max(g["max_completion_length"],
                      rl_config.dev_sampling(cfg)["max_completion_length"])
    assert cfg["vllm"]["max_model_len"] >= LONGEST_PROMPT[cfg["model"]["name"]] + longest_cap


def test_an_override_is_parsed_and_a_typo_is_refused():
    cfg = rl_config.load(str(ROOT / CONFIGS[1]),
                         ["grpo.learning_rate=2e-5", "data.levels=[1,2,3]", "seed=3"])
    assert cfg["grpo"]["learning_rate"] == 2e-5 and isinstance(cfg["grpo"]["learning_rate"],
                                                                float)
    assert cfg["data"]["levels"] == [1, 2, 3] and cfg["seed"] == 3
    with pytest.raises(SystemExit, match="no such key"):
        rl_config.load(str(ROOT / CONFIGS[1]), ["grpo.learnig_rate=1"])


def test_the_run_name_encodes_model_loss_lr_and_seed():
    names = {p: rl_config.run_name(rl_config.load(str(ROOT / p), ["seed=7"])) for p in CONFIGS}
    assert names == {CONFIGS[0]: "qwen3-4b-dapo-lr1e-05-s7",
                     CONFIGS[1]: "smoke-qwen3-4b-dapo-lr1e-05-s7",
                     CONFIGS[2]: "qwen3.5-4b-dapo-lr1e-05-s7",
                     CONFIGS[3]: "qwen3.5-4b-curriculum-dapo-lr1e-05-s7",
                     CONFIGS[4]: "gsm8k-qwen3-0.6b-dapo-lr2e-05-s7",
                     CONFIGS[5]: "gsm8k-hard-qwen3-0.6b-dapo-lr2e-05-s7",
                     CONFIGS[6]: "gsm8k-hard-qwen3-0.6b-adapter-dapo-lr2e-05-s7"}


def test_an_eval_model_the_run_cannot_follow_is_refused(tmp_path, monkeypatch):
    cfg = rl_config.load(str(ROOT / CONFIGS[0]))
    conf = tmp_path / "evals/conf/model"
    conf.mkdir(parents=True)
    base = yaml.safe_load((ROOT / "evals/conf/model/hf-qwen3-4b.yaml").read_text())
    monkeypatch.setattr(rl_config, "ROOT", tmp_path)
    (conf / "hf-qwen3-4b.yaml").write_text(yaml.safe_dump(
        {**base, "sampling": {**base["sampling"], "frequency_penalty": 0.5}}))
    with pytest.raises(SystemExit, match="frequency_penalty"):
        rl_config.eval_model(cfg)
    (conf / "hf-qwen3-4b.yaml").write_text(yaml.safe_dump({**base, "model": "Qwen/Qwen3-8B"}))
    with pytest.raises(SystemExit, match="Qwen3-8B"):
        rl_config.eval_model(cfg)


def test_the_curriculum_config_differs_from_qwen35_only_where_it_says():
    base = rl_config.load(str(ROOT / "rl/configs/qwen3.5-4b.yaml"))
    cur = rl_config.load(str(ROOT / "rl/configs/qwen3.5-4b-curriculum.yaml"))
    assert cur["data"]["levels"] == list(range(1, 9)) and cur["data"]["tasks"] == "all"
    assert cur["grpo"]["num_generations_eval"] == 2
    assert cur["grpo"]["generation_kwargs"] == {"presence_penalty": 1.5}
    assert cur["curriculum"] == {"prior": "results/hf-qwen3.5-4b__xml_tags__cot/metrics.json",
                                 "half_life": 10, "floor": 0.01, "table_every": 5}
    for c in (base, cur):
        del c["name"], c["data"]["levels"], c["grpo"]["num_generations_eval"]
    del cur["curriculum"], cur["grpo"]["generation_kwargs"]
    assert base == cur
