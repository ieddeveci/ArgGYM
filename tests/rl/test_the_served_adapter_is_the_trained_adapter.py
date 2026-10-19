"""Under `vllm.sync: adapter`, vLLM serves the trainer's adapter, named on every request.

vLLM accepts an adapter whose tensor names it does not map and serves the base
model for those modules, so the saved adapter is put through vLLM's own loader
for both trained models and every module it yields must be one vLLM holds a
weight for. That the served module also applies LoRA (a supported layer, the
packed assembly) needs a built model; at run time a probe per module kind
(`rl.train.probe_served`) stops the run when vLLM's log-prob shift is not the
trainer's. The server and the trainer are stood in for on CPU: a fake vLLM
loads each published adapter with PEFT onto a fresh copy of the base model.

Needs the `rl` group; the name checks also need the pinned checkpoints in the
Hugging Face cache and are skipped without them.
"""
from __future__ import annotations

import importlib
import json
import types
from pathlib import Path

import pytest

from rl import config as rl_config
from rl.launch import vllm_command, vllm_env
from rl.train import (
    PROBE_SHIFT,
    PROBE_TOLERANCE,
    adapter_tensors,
    check_served,
    probe_adapter,
    probe_served,
    probe_size,
    sync_adapter,
    write_adapter,
)

ROOT = Path(__file__).resolve().parents[2]
HARD = "rl/configs/gsm8k-hard-qwen3-0.6b.yaml"
ADAPTER = "rl/configs/gsm8k-hard-qwen3-0.6b-adapter.yaml"


def test_the_server_takes_runtime_adapters_only_under_adapter_sync():
    merged = rl_config.load(str(ROOT / HARD))
    adapter = rl_config.load(str(ROOT / ADAPTER))
    cmd = vllm_command(adapter)
    assert cmd[cmd.index("--max-lora-rank") + 1] == "32" and "--enable-lora" in cmd
    assert int(cmd[cmd.index("--max-loras") + 1]) >= 2
    assert cmd[cmd.index("--dtype") + 1] == "bfloat16"
    assert vllm_env(adapter, {})["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "1"
    cmd = vllm_command(merged)
    assert not {"--enable-lora", "--max-lora-rank", "--max-loras"} & set(cmd)
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" not in vllm_env(merged, {})


def test_adapter_sync_refuses_an_fp32_server_and_an_unknown_mode():
    cfg = rl_config.load(str(ROOT / HARD), ["vllm.dtype=float32"])
    cfg["vllm"]["sync"] = "adapter"
    with pytest.raises(SystemExit, match="BF16"):
        vllm_command(cfg)
    cfg["vllm"]["sync"] = "lora"
    with pytest.raises(SystemExit, match="merged or adapter"):
        rl_config.vllm_sync(cfg)


def test_the_adapter_config_differs_from_the_hard_control_only_in_sync():
    hard = rl_config.load(str(ROOT / HARD))
    adapter = rl_config.load(str(ROOT / ADAPTER))
    assert (adapter["vllm"]["sync"], adapter["vllm"]["dtype"]) == ("adapter", "bfloat16")
    assert adapter["name"] == "gsm8k-hard-qwen3-0.6b-adapter"
    for c in (hard, adapter):
        del c["name"], c["vllm"]["dtype"]
    del adapter["vllm"]["sync"]
    assert hard == adapter


def test_the_lora_rank_is_one_vllm_accepts():
    assert [rl_config.max_lora_rank(r) for r in (1, 8, 20, 32, 33)] == [1, 8, 32, 32, 64]


# --- names: what the trainer saves is what vLLM maps -------------------------------------

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

MODELS = {"Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
          "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}


def _snapshot(name):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(snapshot_download(name, revision=MODELS[name], local_files_only=True))
    except LocalEntryNotFoundError:
        pytest.skip(f"{name} is not in the Hugging Face cache")


def _vllm_view(snapshot):
    """Checkpoint names, vLLM's module names for them, and the served class's renaming."""
    pytest.importorskip("vllm")
    from vllm.model_executor.models.registry import _VLLM_MODELS

    from rl.export import _tensors

    arch = json.loads((snapshot / "config.json").read_text())["architectures"][0]
    module, cls = _VLLM_MODELS[arch]
    served = getattr(importlib.import_module(f"vllm.model_executor.models.{module}"), cls)
    mapper = getattr(served, "hf_to_vllm_mapper", None)
    mapper = mapper.get_rename_mapper() if mapper is not None else None
    keys = set(_tensors(snapshot))
    modules = set()
    for k in keys:
        mapped = mapper._map_name(k) if mapper is not None else k
        if mapped is not None and mapped.endswith(".weight"):
            modules.add(mapped.removesuffix(".weight"))

    return keys, modules, mapper


def _trained_on_meta(name, r=32):
    """The adapter `rl.train` trains: the text-only class, all-linear LoRA, no weights loaded."""
    config = transformers.AutoConfig.from_pretrained(name, revision=MODELS[name])
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(config.get_text_config()
                                                              if hasattr(config, "text_config")
                                                              else config)
        return peft.get_peft_model(model, peft.LoraConfig(r=r, lora_alpha=r,
                                                          target_modules="all-linear"))


@pytest.mark.parametrize("name", list(MODELS))
def test_vllms_loader_takes_every_adapted_module_onto_a_module_it_serves(name, tmp_path,
                                                                         monkeypatch):
    import vllm.lora.lora_model as lora_model
    from vllm.config.lora import LoRAConfig
    from vllm.lora.peft_helper import PEFTHelper

    snapshot = _snapshot(name)
    keys, modules, mapper = _vllm_view(snapshot)
    # Rank 1 has the names of rank 32 at a fraction of the bytes.
    pm = _trained_on_meta(name, r=1)
    adapter = tmp_path / "policy-v1"
    write_adapter(pm, adapter, {k: torch.zeros(v.shape)
                                for k, v in adapter_tensors(pm, keys).items()})
    # What `/v1/load_lora_adapter` runs (vllm/lora/worker_manager.py, vLLM 0.30),
    # on CPU: the adapter config against the server's LoRA limits, then the
    # tensors, names mapped by the served class and checked against the module
    # kinds vLLM holds weights for.
    helper = PEFTHelper.from_local_dir(str(adapter), 4096)
    helper.validate_legal(LoRAConfig(max_lora_rank=32, max_loras=2))
    monkeypatch.setattr(lora_model, "PIN_MEMORY", False)
    loaded = lora_model.LoRAModel.from_local_checkpoint(
        str(adapter), {m.rsplit(".", 1)[-1] for m in modules}, helper, device="cpu",
        weights_mapper=mapper)
    landed = set(loaded.loras)
    assert landed <= modules, sorted(landed - modules)[:3]
    # Every adapted module, once.
    assert len(landed) == sum(".lora_A." in n for n, _ in pm.named_parameters())


def test_qwen35s_raw_peft_names_would_serve_the_base_model():
    """Why `adapter_tensors` renames: the text-only class's names map onto nothing in vLLM."""
    snapshot = _snapshot("Qwen/Qwen3.5-4B")
    from vllm.lora.utils import parse_fine_tuned_lora_name

    _, modules, mapper = _vllm_view(snapshot)
    raw = peft.get_peft_model_state_dict(_trained_on_meta("Qwen/Qwen3.5-4B"))
    assert not {parse_fine_tuned_lora_name(k, mapper)[0] for k in raw} & modules


# --- the guard ------------------------------------------------------------------------------


def test_the_guard_trips_when_vllm_serves_the_base_model():
    trained = [0.3, -0.8, 1.1, 0.2, -0.5]
    with pytest.raises(RuntimeError, match="does not serve"):
        check_served("in_proj_a", [0.0] * len(trained), trained)
    with pytest.raises(RuntimeError, match="does not serve"):  # half the shift
        check_served("q_proj", [t * 0.4 for t in trained], trained)
    noisy = [t + e for t, e in zip(trained, [0.05, -0.04, 0.06, 0.0, -0.03])]
    assert check_served("q_proj", noisy, trained) < PROBE_TOLERANCE


def test_the_probe_grows_until_a_saturating_kind_shifts_enough():
    """Qwen3.5-4B's `in_proj_b` on CPU: 0.08 RMS at size 0.3, 0.64 at 4.8, flat near 0.9."""
    import math

    def in_proj_b(size):
        return [0.9 * math.tanh(size / 4.5)] * 8

    size, shift = probe_size("in_proj_b", in_proj_b)
    assert shift[0] >= PROBE_SHIFT and size > 2.4


def test_a_probe_that_cannot_shift_enough_stops_before_vllm_is_asked(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="too little"):
        probe_size("in_proj_b", lambda size: [min(size, 0.3)] * 8)
    monkeypatch.setattr("rl.train.PROBE_SHIFT", 1e6)
    gen = _with_base(fake_gen(trained_adapter(), tmp_path))
    with pytest.raises(RuntimeError, match="too little"):
        probe_served(gen)
    assert gen.vllm_client.calls == []


def tiny_model():
    """A tiny Qwen3 whose log-probs a probe can move: at the default init they are near uniform."""
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
        initializer_range=0.2)
    return transformers.Qwen3ForCausalLM(config)


def trained_adapter():
    pm = peft.get_peft_model(tiny_model(), peft.LoraConfig(r=8, lora_alpha=8,
                                                           target_modules="all-linear"))
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    return pm


def test_a_probe_moves_one_kind_by_its_size_and_puts_the_trained_factors_back():
    pm = trained_adapter()
    before = {n: p.detach().clone() for n, p in pm.named_parameters()}
    with probe_adapter(pm, "v_proj", 0.3):
        for name, module in pm.named_modules():
            if not hasattr(module, "lora_B") or "default" not in module.lora_B:
                continue
            delta = module.get_delta_weight("default")
            if name.endswith("v_proj"):
                ratio = delta.norm() / module.base_layer.weight.norm()
                assert ratio.item() == pytest.approx(0.3, rel=1e-4)
            else:
                assert not delta.any()
    assert all(torch.equal(before[n], p) for n, p in pm.named_parameters())


class FakeVLLM:
    """vLLM's LoRA endpoints and prompt log-probs, over PEFT on a fresh base model.

    `drop` serves the base model whatever adapter a request names, as vLLM does
    for modules whose names it does not map.
    """

    base_url = "http://fake"

    def __init__(self, drop=False):
        self.model, self.drop, self.loaded, self.calls = "Qwen/Base", drop, {}, []
        self.session = types.SimpleNamespace(post=self._post)

    def _post(self, url, json, timeout):
        assert timeout == 240  # the trainer's vllm_server_timeout
        endpoint = url.rsplit("/", 1)[-1]
        self.calls.append((endpoint, json["lora_name"]))
        if endpoint == "load_lora_adapter":
            self.loaded[json["lora_name"]] = peft.PeftModel.from_pretrained(
                tiny_model(), json["lora_path"])
        else:
            del self.loaded[json["lora_name"]]
        return types.SimpleNamespace(status_code=200, text="")

    def get_sequence_logprobs(self, sequences, prompt_lengths, top_logprobs):
        from rl.train import _trainer_logprobs

        model = tiny_model() if self.drop or self.model == "Qwen/Base" else self.loaded[self.model]
        lp = _trainer_logprobs(model, sequences[0])[prompt_lengths[0] - 1:]
        return {"actual_logprobs": [[[x] for x in lp]]}


def fake_gen(pm, tmp_path, drop=False):
    return types.SimpleNamespace(
        model=pm, accelerator=types.SimpleNamespace(is_main_process=True),
        vllm_client=FakeVLLM(drop), adapter_root=tmp_path / "vllm-lora", adapter_version=0,
        server_timeout=240,
        checkpoint_keys={n.replace(".base_layer", "") for n, _ in
                         pm.get_base_model().named_parameters() if "lora_" not in n},
        processing_class=lambda text, add_special_tokens: {
            "input_ids": [ord(c) % 128 for c in text[:48]]})


def test_every_request_after_a_sync_names_the_newest_adapter(tmp_path):
    pm = trained_adapter()
    gen = fake_gen(pm, tmp_path)
    for stale in ("policy-v7", "probe-q_proj"):  # an earlier attempt in this output dir
        (gen.adapter_root / stale).mkdir(parents=True)
    sync_adapter(gen)
    client = gen.vllm_client
    assert client.model == "policy-v1" and set(client.loaded) == {"policy-v1"}
    assert sorted(p.name for p in gen.adapter_root.iterdir()) == ["policy-v1", "probe.json"]
    probe = json.loads((gen.adapter_root / "probe.json").read_text())
    assert set(probe) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                          "down_proj"}
    assert all(p["shift_rms"] >= PROBE_SHIFT for p in probe.values())
    assert max(p["error"] for p in probe.values()) < 1e-3  # same kernels on both sides
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.add_(0.01)
    sync_adapter(gen)
    assert client.model == "policy-v2" and set(client.loaded) == {"policy-v2"}
    assert sorted(p.name for p in gen.adapter_root.iterdir()) == ["policy-v2", "probe.json"]
    # The served adapter is the trained one, to the last factor.
    served = peft.get_peft_model_state_dict(client.loaded["policy-v2"])
    trained = peft.get_peft_model_state_dict(pm)
    assert served.keys() == trained.keys()
    assert all(torch.equal(served[k], trained[k]) for k in trained)


def test_a_server_that_drops_the_adapter_stops_the_run_before_any_rollout(tmp_path):
    gen = fake_gen(trained_adapter(), tmp_path, drop=True)
    with pytest.raises(RuntimeError, match="does not serve"):
        sync_adapter(gen)
    assert gen.vllm_client.model == "Qwen/Base" and gen.adapter_version == 0


def test_the_probe_leaves_the_trained_adapter_and_the_mode_as_they_were(tmp_path):
    pm = trained_adapter()
    pm.train()
    before = {n: p.detach().clone() for n, p in pm.named_parameters()}
    probe_served(_with_base(fake_gen(pm, tmp_path)))
    assert pm.training
    assert all(torch.equal(before[n], p) for n, p in pm.named_parameters())


def _with_base(gen):
    gen.base_name = gen.vllm_client.model
    gen.adapter_root.mkdir(parents=True)
    return gen
