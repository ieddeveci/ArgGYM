"""What vLLM serves is the base model plus the trained adapter, during training and after.

During training, the weights pushed to the rollout server are merged out of
place, so the trainer's BF16 base weights never move (`rl.train.merged_params`).
After training, `rl.export` passes a merged model whose trained tensors differ
from base and whose other tensors are identical, and fails one that matches base.

A tiny Qwen3 on CPU; needs the `rl` group, skipped without it.
"""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from rl.export import compare, trained_weights  # noqa: E402
from rl.train import merged_params  # noqa: E402


def tiny_model():
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        tie_word_embeddings=True)
    return transformers.Qwen3ForCausalLM(config).to(torch.bfloat16)


def with_adapter(model, train: bool):
    lora = peft.LoraConfig(r=8, lora_alpha=8, target_modules="all-linear")
    pm = peft.get_peft_model(model, lora)
    if train:
        with torch.no_grad():
            for name, p in pm.named_parameters():
                if "lora_B" in name:
                    p.normal_(0, 0.05)
    return pm


def gen_for(pm):
    """The attributes of TRL's VLLMGeneration that `merged_params` reads."""
    g = types.SimpleNamespace(model=pm)
    g._fix_param_name_to_vllm = lambda name, extra_prefixes=(): name.replace(
        "modules_to_save.default.", "")
    return g


def test_sync_never_writes_the_base_weights_and_sends_base_plus_delta():
    pm = with_adapter(tiny_model(), train=True)
    before = {n: p.detach().clone() for n, p in pm.named_parameters() if "base_layer" in n}
    lin = pm.base_model.model.model.layers[0].self_attn.q_proj
    for _ in range(3):
        sent = dict(merged_params(gen_for(pm)))
        with torch.no_grad():
            want = (lin.base_layer.weight.float()
                    + lin.get_delta_weight("default").float()).to(torch.bfloat16)
            assert torch.equal(sent["model.layers.0.self_attn.q_proj.weight"], want)
            assert not torch.equal(want, lin.base_layer.weight)
            for name, p in pm.named_parameters():
                if "lora_B" in name:
                    p.add_(0.01)
    after = {n: p.detach() for n, p in pm.named_parameters() if "base_layer" in n}
    assert all(torch.equal(before[n], after[n]) for n in before)
    # Names are the plain model's, as vLLM expects, and no LoRA tensor is sent.
    assert set(sent) == {n for n, _ in tiny_model().named_parameters()}


def test_peft_in_place_merge_would_have_moved_them():
    """Why `merged_params` exists: PEFT's round trip is not exact in BF16."""
    pm = with_adapter(tiny_model(), train=True)
    w = pm.base_model.model.model.layers[0].mlp.up_proj.base_layer.weight
    before = w.detach().clone()
    for _ in range(20):
        with torch.no_grad():
            for name, p in pm.named_parameters():
                if "lora_B" in name:
                    p.add_(1e-3)
        pm.merge_adapter()
        pm.unmerge_adapter()
    assert not torch.equal(before, w.detach())


def _export(tmp_path, train: bool):
    base = tiny_model()
    base_dir, adapter_dir, merged_dir = (tmp_path / d for d in ("base", "adapter", "merged"))
    base.save_pretrained(base_dir, safe_serialization=True)
    pm = with_adapter(base, train)
    pm.save_pretrained(adapter_dir)
    fresh = transformers.Qwen3ForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16)
    merged = peft.PeftModel.from_pretrained(fresh, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    return compare(merged_dir, base_dir, trained_weights(adapter_dir))


def test_export_passes_a_trained_adapter(tmp_path):
    report = _export(tmp_path, train=True)
    assert report["ok"], report
    assert report["n_trained"] == 2 * 7  # q k v o gate up down, two layers


def test_export_fails_an_adapter_that_changes_nothing(tmp_path):
    report = _export(tmp_path, train=False)
    assert not report["ok"]
    assert len(report["trained_identical_to_base"]) == report["n_trained"]
