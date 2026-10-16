"""`selection.json` puts the selected checkpoint beside the base model's step-0 dev score.

Transformers selects among checkpoints only, so without this a run in which
every checkpoint is worse than the base still names a "best" one.
"""
from __future__ import annotations

import types
from pathlib import Path

from rl import config as rl_config
from rl.launch import vllm_command, vllm_env
from rl.train import select

ROOT = Path(__file__).resolve().parents[2]
METRIC = "eval_arggym_success_rate"


def state(history, best, ckpt="checkpoint-20"):
    return types.SimpleNamespace(log_history=history, best_metric=best,
                                 best_model_checkpoint=ckpt, global_step=20)


def test_a_checkpoint_below_the_base_is_marked():
    # The smoke run's numbers: base 0.104, both checkpoints 0.083.
    history = [{"step": 0, METRIC: 0.104}, {"step": 1, "loss": 0.0},
               {"step": 10, METRIC: 0.083}, {"step": 20, METRIC: 0.083}]
    s = select(state(history, 0.083), METRIC)
    assert s["base_metric"] == 0.104 and s["improved_over_base"] is False


def test_a_checkpoint_above_the_base_is_marked():
    s = select(state([{"step": 0, METRIC: 0.1}, {"step": 10, METRIC: 0.2}], 0.2), METRIC)
    assert s["improved_over_base"] is True


def test_no_base_evaluation_means_no_claim():
    s = select(state([{"step": 10, METRIC: 0.2}], 0.2), METRIC)
    assert s["base_metric"] is None and s["improved_over_base"] is None


def test_the_server_answers_to_the_eval_configs_model_name():
    cmd = vllm_command(rl_config.load(str(ROOT / "rl/configs/qwen3-4b.yaml")))
    assert cmd[cmd.index("--served-model-name") + 1] == "Qwen/Qwen3-4B"
    assert cmd[cmd.index("--max-model-len") + 1] == "40960"
    cmd = vllm_command(rl_config.load(str(ROOT / "rl/configs/qwen3.5-4b.yaml")))
    assert cmd[cmd.index("--served-model-name") + 1] == "Qwen/Qwen3.5-4B"
    assert "--language-model-only" in cmd


def test_the_server_avoids_the_flashinfer_jit_sampler_and_finds_the_venv_tools():
    import sys

    cfg = rl_config.load(str(ROOT / "rl/configs/smoke.yaml"))
    env = vllm_env(cfg, {"PATH": "/usr/bin"})
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["PATH"].split(":")[0] == str(Path(sys.executable).parent)
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and env["VLLM_SERVER_DEV_MODE"] == "1"
