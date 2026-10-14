"""The configs say what was decided, and their batch arithmetic closes.

The first-run config is the record of the choices made for it; a change to one
of them should be a visible edit to this test, not a quiet edit to a YAML file.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rl import config as rl_config

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ("rl/configs/qwen3-4b.yaml", "rl/configs/smoke.yaml")


def test_the_first_run_config_holds_the_decisions():
    cfg = rl_config.load(str(ROOT / "rl/configs/qwen3-4b.yaml"))
    g = cfg["grpo"]
    assert cfg["model"]["name"] == "Qwen/Qwen3-4B"
    assert cfg["lora"] == {"r": 32, "alpha": 32, "dropout": 0.0, "target_modules": "all-linear"}
    assert (g["learning_rate"], g["lr_scheduler_type"]) == (1e-5, "constant_with_warmup")
    assert (g["loss_type"], g["beta"], g["epsilon"], g["epsilon_high"]) == ("dapo", 0.0, 0.2, 0.28)
    assert (g["num_generations"], cfg["prompts_per_step"], g["scale_rewards"]) == (8, 32, "none")
    assert (g["temperature"], g["top_p"], g["top_k"]) == (1.0, 1.0, 0)
    assert g["mask_truncated_completions"] and g["vllm_importance_sampling_correction"]
    assert g["max_completion_length"] == 16384
    assert cfg["prompt"]["enable_thinking"] is True
    assert cfg["data"]["levels"] == list(range(1, 9))
    assert cfg["wandb_project"] == "arggym-rl"
    assert g["save_total_limit"] <= 2


@pytest.mark.parametrize("path", CONFIGS)
def test_the_batch_arithmetic_closes_on_two_trainer_gpus(path):
    cfg = rl_config.load(str(ROOT / path))
    g = cfg["grpo"]
    world = len(str(cfg["trainer_gpus"]).split(","))
    assert world == 2 and str(cfg["vllm"]["gpu"]) not in str(cfg["trainer_gpus"]).split(",")
    completions = g["per_device_train_batch_size"] * world * g["gradient_accumulation_steps"]
    assert completions == cfg["prompts_per_step"] * g["num_generations"]
    dev_rows = 12 * len(cfg["data"]["levels"]) * cfg["data"]["dev"]["per_task_level"]
    assert dev_rows * g["num_generations_eval"] % (g["per_device_eval_batch_size"] * world) == 0
    # load_best_model_at_end needs a checkpoint at every eval.
    assert g["save_steps"] % g["eval_steps"] == 0
    assert cfg["vllm"]["max_model_len"] > g["max_completion_length"]


def test_an_override_is_parsed_and_a_typo_is_refused():
    cfg = rl_config.load(str(ROOT / CONFIGS[1]),
                         ["grpo.learning_rate=2e-5", "data.levels=[1,2,3]", "seed=3"])
    assert cfg["grpo"]["learning_rate"] == 2e-5 and isinstance(cfg["grpo"]["learning_rate"],
                                                                float)
    assert cfg["data"]["levels"] == [1, 2, 3] and cfg["seed"] == 3
    with pytest.raises(SystemExit, match="no such key"):
        rl_config.load(str(ROOT / CONFIGS[1]), ["grpo.learnig_rate=1"])


def test_the_run_name_encodes_model_loss_lr_and_seed():
    cfg = rl_config.load(str(ROOT / CONFIGS[0]), ["seed=7"])
    assert rl_config.run_name(cfg) == "qwen3-4b-dapo-lr1e-05-s7"
    cfg = rl_config.load(str(ROOT / CONFIGS[1]))
    assert rl_config.run_name(cfg) == "smoke-qwen3-4b-dapo-lr1e-05-s0"
