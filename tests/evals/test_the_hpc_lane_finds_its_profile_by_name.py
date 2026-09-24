"""The HPC lane pairs an eval config with the profile that serves it by file stem.

Neither file names the other, and the checkpoint id lives only in the eval
config, so the pairing rule and the scripts that apply it are the whole contract.
These run without a GPU: the runner's --dry-run and submit_truba.sh's
SUBMIT_DRY_RUN stop before anything is served or submitted.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HPC = ROOT / "hpc" / "vllm"
sys.path.insert(0, str(HPC))

from pairing import level_of, profile_of  # noqa: E402

PROFILES = {"hf-qwen3-8b", "hf-qwen3.8-27b", "hf-gpt-oss-20b", "hf-qwen3.5-9b"}


@pytest.mark.parametrize("config, profile, level", [
    ("hf-qwen3-8b", "hf-qwen3-8b", None),
    ("hf-qwen3.8-27b-xhigh", "hf-qwen3.8-27b", "xhigh"),
    ("hf-gpt-oss-20b-low", "hf-gpt-oss-20b", "low"),
    # A thinking-off config is served by its thinking-on sibling's profile.
    ("hf-qwen3.5-9b-nothink", "hf-qwen3.5-9b", None),
    ("hf-unknown-nothink", None, None),
    # A level suffix only counts when what is left is a profile.
    ("hf-qwen3-8b-turbo", None, None),
    ("hf-unknown-medium", None, None),
])
def test_a_config_is_served_by_the_profile_its_stem_names(config, profile, level):
    assert profile_of(config, PROFILES) == profile
    assert level_of(config, PROFILES) == level


def test_the_bundle_verifies():
    out = subprocess.run([sys.executable, str(HPC / "verify_bundle.py")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def run_dry(config):
    return subprocess.run([sys.executable, str(HPC / "run_vllm_arggym.py"), config,
                           "--dry-run"], capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "GPU_TYPE": "H100"})


def test_the_runner_serves_the_checkpoint_the_config_names():
    out = run_dry("hf-qwen3.8-27b-medium")
    assert out.returncode == 0, out.stderr
    assert "Profile: hf-qwen3.8-27b" in out.stdout
    assert "vllm serve Qwen/Qwen3.8-27B " in out.stdout
    assert "--served-model-name Qwen/Qwen3.8-27B " in out.stdout
    assert "hf-qwen3.8-27b-medium__xml_tags__cot" in out.stdout


def test_the_runner_names_the_levels_when_asked_for_a_levelled_model_without_one():
    out = run_dry("hf-qwen3.8-27b")
    assert out.returncode != 0
    assert "hf-qwen3.8-27b-low, hf-qwen3.8-27b-medium, hf-qwen3.8-27b-xhigh" in out.stderr


def test_submit_passes_the_config_and_the_profiles_gpu_count(tmp_path):
    env = {**os.environ, "SUBMIT_DRY_RUN": "1", "GPU_TYPE": "H100",
           "ARGGYM_RUNTIME_DIR": str(tmp_path), "USER": os.environ.get("USER", "u")}
    out = subprocess.run(["bash", str(HPC / "submit_truba.sh"), "hf-gpt-oss-120b-high"],
                         capture_output=True, text=True, cwd=ROOT, env=env)
    assert out.returncode == 0, out.stderr
    assert "MODEL_CONFIG=hf-gpt-oss-120b-high" in out.stderr
    assert "--gres=gpu:2" in out.stderr  # tensor_parallel_size_h100: 2
    assert "Serving profile: hf-gpt-oss-120b" in out.stdout


def test_submit_refuses_a_config_that_does_not_exist(tmp_path):
    env = {**os.environ, "SUBMIT_DRY_RUN": "1", "ARGGYM_RUNTIME_DIR": str(tmp_path)}
    out = subprocess.run(["bash", str(HPC / "submit_truba.sh"), "hf-qwen3.8-27b"],
                         capture_output=True, text=True, cwd=ROOT, env=env)
    assert out.returncode == 2
    assert "Pick one of" in out.stderr
