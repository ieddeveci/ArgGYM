"""Dev evaluation samples as the eval config says; training rollouts take no other penalty.

Qwen3.5's card asks for presence_penalty 1.5, and dev evaluation must match the
final eval. A training rollout may take that same penalty and no other.
"""
from __future__ import annotations

import logging
import types
from pathlib import Path

import pytest

from rl import config as rl_config
from rl.train import FallbackGuard, check_penalties, dev_sampling_swap

ROOT = Path(__file__).resolve().parents[2]


def fake_gen():
    """The `VLLMGeneration` attributes the swap touches, at their training values."""
    return types.SimpleNamespace(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
                                 repetition_penalty=1.0, max_completion_length=40960,
                                 generation_kwargs=None)


def test_the_penalty_is_on_during_dev_evaluation_and_off_after_it():
    cfg = rl_config.load(str(ROOT / "rl/configs/qwen3.5-4b.yaml"))
    gen, trainer = fake_gen(), types.SimpleNamespace(vllm_importance_sampling_correction=True)
    before = dict(vars(gen))
    with dev_sampling_swap(trainer, gen, rl_config.dev_sampling(cfg)):
        assert gen.generation_kwargs == {"presence_penalty": 1.5}
        assert (gen.temperature, gen.top_p, gen.top_k) == (1.0, 0.95, 20)
        assert gen.max_completion_length == 65536
        assert trainer.vllm_importance_sampling_correction is False
    assert vars(gen) == before and trainer.vllm_importance_sampling_correction is True


def test_an_eval_that_raises_still_restores_the_training_sampling():
    cfg = rl_config.load(str(ROOT / "rl/configs/qwen3.5-4b.yaml"))
    gen, trainer = fake_gen(), types.SimpleNamespace(vllm_importance_sampling_correction=True)
    with pytest.raises(RuntimeError):
        with dev_sampling_swap(trainer, gen, rl_config.dev_sampling(cfg)):
            raise RuntimeError("vLLM went away")
    assert gen.generation_kwargs is None and gen.max_completion_length == 40960


def test_a_config_without_a_penalty_adds_none():
    cfg = rl_config.load(str(ROOT / "rl/configs/qwen3-4b.yaml"))
    gen, trainer = fake_gen(), types.SimpleNamespace(vllm_importance_sampling_correction=True)
    with dev_sampling_swap(trainer, gen, rl_config.dev_sampling(cfg)):
        assert gen.generation_kwargs is None


@pytest.mark.parametrize("grpo", [{"generation_kwargs": {"presence_penalty": 1.0}},
                                  {"repetition_penalty": 1.1},
                                  {"generation_kwargs": {"frequency_penalty": 0.2}}])
def test_a_penalty_other_than_the_evals_is_refused(grpo):
    with pytest.raises(SystemExit, match="penalty"):
        check_penalties(grpo, 1.5)


def test_the_evals_presence_penalty_is_allowed():
    check_penalties({"generation_kwargs": {"presence_penalty": 1.5}}, 1.5)


@pytest.mark.parametrize("path", ["rl/configs/qwen3.5-4b.yaml", "rl/configs/qwen3-4b.yaml",
                                  "rl/configs/smoke.yaml",
                                  "rl/configs/qwen3.5-4b-curriculum.yaml"])
def test_every_config_passes_the_penalty_check(path):
    cfg = rl_config.load(str(ROOT / path))
    check_penalties(cfg["grpo"], rl_config.dev_sampling(cfg)["presence_penalty"])


def _warn(guard, msg):
    guard.emit(logging.LogRecord("transformers", logging.WARNING, "", 0, msg, None, None))


def test_a_gated_deltanet_fallback_stops_the_run_and_the_conv_one_does_not():
    guard = FallbackGuard()
    _warn(guard, "`causal_conv1d_fn` is falling back to its reference PyTorch implementation "
                 "because `causal_conv1d` is not installed. This is correct but much slower")
    guard.check()
    _warn(guard, "`chunk_gated_delta_rule` is falling back to its reference PyTorch "
                 "implementation because `flash-linear-attention` is not installed.")
    with pytest.raises(RuntimeError, match="chunk_gated_delta_rule"):
        guard.check()
