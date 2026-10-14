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
CONFIGS = ("rl/configs/qwen3-4b.yaml", "rl/configs/qwen3-4b-nothink.yaml",
           "rl/configs/smoke.yaml")
REAL = CONFIGS[:2]
#: The longest prompt in data/taskset.jsonl with the Qwen3 tokenizer is 4,066
#: tokens (evals/conf/model/hf-qwen3-4b.yaml); levels 1-8 are shorter.
LONGEST_PROMPT = 4096


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
    assert cfg["wandb_project"] == "arggym-rl"
    assert g["save_total_limit"] <= 2


@pytest.mark.parametrize("path", CONFIGS)
def test_importance_sampling_is_truncated_per_token(path):
    g = rl_config.load(str(ROOT / path))["grpo"]
    assert g["vllm_importance_sampling_mode"] == "token_truncate"
    assert g["vllm_importance_sampling_clip_max"] == 2.0


def test_the_thinking_off_config_differs_only_where_it_says():
    on = rl_config.load(str(ROOT / "rl/configs/qwen3-4b.yaml"))
    off = rl_config.load(str(ROOT / "rl/configs/qwen3-4b-nothink.yaml"))
    assert rl_config.eval_model(off)["chat_template_kwargs"] == {"enable_thinking": False}
    assert off["grpo"]["max_completion_length"] == 8192
    for c in (on, off):
        del c["name"], c["eval_model"], c["grpo"]["max_completion_length"]
    assert on == off


@pytest.mark.parametrize("path", REAL)
def test_a_real_run_evaluates_dev_at_the_eval_cap(path):
    cfg = rl_config.load(str(ROOT / path))
    assert "dev_max_tokens" not in cfg
    cap = rl_config.dev_sampling(cfg)["max_completion_length"]
    assert cap == rl_config.eval_model(cfg)["max_tokens"] == 32768


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
    # A checkpoint at every eval and an eval at every checkpoint.
    assert g["save_steps"] == g["eval_steps"]
    longest_cap = max(g["max_completion_length"],
                      rl_config.dev_sampling(cfg)["max_completion_length"])
    assert cfg["vllm"]["max_model_len"] >= LONGEST_PROMPT + longest_cap


def test_an_override_is_parsed_and_a_typo_is_refused():
    cfg = rl_config.load(str(ROOT / CONFIGS[2]),
                         ["grpo.learning_rate=2e-5", "data.levels=[1,2,3]", "seed=3"])
    assert cfg["grpo"]["learning_rate"] == 2e-5 and isinstance(cfg["grpo"]["learning_rate"],
                                                                float)
    assert cfg["data"]["levels"] == [1, 2, 3] and cfg["seed"] == 3
    with pytest.raises(SystemExit, match="no such key"):
        rl_config.load(str(ROOT / CONFIGS[2]), ["grpo.learnig_rate=1"])


def test_the_run_name_encodes_model_loss_lr_and_seed():
    names = {p: rl_config.run_name(rl_config.load(str(ROOT / p), ["seed=7"])) for p in CONFIGS}
    assert names == {CONFIGS[0]: "qwen3-4b-dapo-lr1e-05-s7",
                     CONFIGS[1]: "qwen3-4b-nothink-dapo-lr1e-05-s7",
                     CONFIGS[2]: "smoke-qwen3-4b-dapo-lr1e-05-s7"}


def test_an_eval_model_the_run_cannot_follow_is_refused(tmp_path, monkeypatch):
    cfg = rl_config.load(str(ROOT / CONFIGS[0]))
    conf = tmp_path / "evals/conf/model"
    conf.mkdir(parents=True)
    base = yaml.safe_load((ROOT / "evals/conf/model/hf-qwen3-4b.yaml").read_text())
    monkeypatch.setattr(rl_config, "ROOT", tmp_path)
    (conf / "hf-qwen3-4b.yaml").write_text(yaml.safe_dump(
        {**base, "sampling": {**base["sampling"], "presence_penalty": 1.5}}))
    with pytest.raises(SystemExit, match="presence_penalty"):
        rl_config.eval_model(cfg)
    (conf / "hf-qwen3-4b.yaml").write_text(yaml.safe_dump({**base, "model": "Qwen/Qwen3-8B"}))
    with pytest.raises(SystemExit, match="Qwen3-8B"):
        rl_config.eval_model(cfg)
