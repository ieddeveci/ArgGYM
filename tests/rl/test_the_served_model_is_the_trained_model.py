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


def tiny_qwen35():
    """Qwen3.5's text model in miniature: one Gated DeltaNet layer, one attention layer."""
    torch.manual_seed(0)
    config = transformers.Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"], num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        tie_word_embeddings=True)
    return transformers.Qwen3_5ForCausalLM(config).to(torch.bfloat16)


def vl_name(name):
    """Where Qwen3.5's vision-language checkpoint keeps a text-model tensor."""
    return "model.language_model." + name[len("model."):] if name.startswith("model.") else name


def gen_for(pm, keys=None):
    """The attributes of TRL's VLLMGeneration that `merged_params` reads."""
    g = types.SimpleNamespace(model=pm)
    g._fix_param_name_to_vllm = lambda name, extra_prefixes=(): name.replace(
        "modules_to_save.default.", "")
    plain = {n for n, _ in pm.get_base_model().named_parameters()}
    g.checkpoint_keys = keys if keys is not None else {
        n.replace(".base_layer", "") for n in plain if "lora_" not in n}
    return g


def test_qwen35_syncs_every_trained_projection_under_the_checkpoints_names():
    pm = with_adapter(tiny_qwen35(), train=True)
    trained = sorted({n.split(".lora_A.")[0].rsplit(".", 1)[-1]
                      for n, _ in pm.named_parameters() if ".lora_A." in n})
    # The Gated DeltaNet projections are adapted; the conv and the decay are not.
    assert trained == ["down_proj", "gate_proj", "in_proj_a", "in_proj_b", "in_proj_qkv",
                       "in_proj_z", "k_proj", "o_proj", "out_proj", "q_proj", "up_proj",
                       "v_proj"]
    plain = {n for n, _ in tiny_qwen35().named_parameters()}
    keys = {vl_name(n) for n in plain}
    sent = dict(merged_params(gen_for(pm, keys)))
    assert set(sent) == keys
    lin = pm.base_model.model.model.layers[0].linear_attn.in_proj_qkv
    want = (lin.base_layer.weight.float() + lin.get_delta_weight("default").float())
    assert torch.equal(sent["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"],
                       want.to(torch.bfloat16))


def test_a_tensor_the_checkpoint_lacks_stops_the_sync():
    pm = with_adapter(tiny_qwen35(), train=True)
    keys = {vl_name(n) for n, _ in tiny_qwen35().named_parameters()}
    keys.discard("model.language_model.layers.0.linear_attn.A_log")
    with pytest.raises(RuntimeError, match="A_log"):
        dict(merged_params(gen_for(pm, keys)))


def test_export_writes_the_checkpoints_layout_for_a_vision_language_base(tmp_path):
    from safetensors.torch import save_file

    from rl.export import to_checkpoint, write_merged

    base = tiny_qwen35()
    base_vl = tmp_path / "base"
    base_vl.mkdir()
    state = {vl_name(n): t.contiguous() for n, t in base.state_dict().items()
             if n != "lm_head.weight"}
    state["mtp.layers.0.norm.weight"] = torch.ones(4, dtype=torch.bfloat16)
    # Qwen3.5 stores the Gated DeltaNet decay in FP32; a BF16 load casts it, and
    # the export must still write the stored bytes.
    a_log = "model.language_model.layers.0.linear_attn.A_log"
    state[a_log] = state[a_log].float() + 1e-3
    save_file(state, str(base_vl / "model.safetensors"), metadata={"format": "pt"})
    (base_vl / "config.json").write_text("{}")
    pm = with_adapter(base, train=True)
    adapter_dir = tmp_path / "adapter"
    pm.save_pretrained(adapter_dir)
    merged = pm.merge_and_unload()
    out = tmp_path / "merged"
    write_merged(merged.state_dict(), base_vl, out, trained_weights(adapter_dir))
    assert (out / "config.json").exists()
    trained = {to_checkpoint(n, state) for n in trained_weights(adapter_dir)}
    report = compare(out, base_vl, trained)
    assert report["ok"] and report["n_tensors"] == len(state)
    assert "mtp.layers.0.norm.weight" not in trained


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


@pytest.mark.parametrize("scale, served", [(0.05, (0.95, 1.05)), (1e-5, (0.0, 0.5))])
def test_the_sync_reports_how_much_of_the_update_survived_bf16(scale, served):
    pm = with_adapter(tiny_qwen35(), train=True)
    with torch.no_grad():
        for name, p in pm.named_parameters():
            if "lora_B" in name:
                p.normal_(0, scale)
    keys = {vl_name(n) for n, _ in tiny_qwen35().named_parameters()}
    gen = gen_for(pm, keys)
    dict(merged_params(gen))
    low, high = served
    assert low <= gen.sync_stats["sync/served_share"] <= high
    assert (gen.sync_stats["sync/moved_fraction"] > 0.9) == (scale == 0.05)
