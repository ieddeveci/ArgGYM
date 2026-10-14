"""Merge the selected adapter into the base model, and prove the merged files are not the base.

    uv run python -m rl.export outputs/rl/runs/<run>            # the checkpoint selection.json names
    uv run python -m rl.export outputs/rl/runs/<run> --checkpoint outputs/rl/runs/<run>/checkpoint-20

The merged model is written to `<run>/merged` as plain safetensors that
`vllm serve <run>/merged` loads with no LoRA flag, so nothing can serve the base
model while claiming to serve the adapter. Serve it with
`--served-model-name <the eval config's model>` (`Qwen/Qwen3-4B`): vLLM
otherwise registers the directory path as the name and the eval harness's first
request is refused. The check then reads the files back
from disk, the bytes vLLM would load, and compares every tensor against the
pinned base checkpoint: every tensor the adapter trained must differ from base
and every other tensor must be identical. Either failure exits non-zero.
`export.json` records the numbers, including the share of elements that moved:
the merge rounds the LoRA delta into BF16, and an update smaller than half a
BF16 step at that weight is lost from the served model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from rl.train import SELECTION


def _tensors(directory: Path) -> Dict[str, Path]:
    from safetensors import safe_open

    out = {}
    for f in sorted(directory.glob("*.safetensors")):
        with safe_open(str(f), "pt") as st:
            out.update({k: f for k in st.keys()})
    return out


def _load(path: Path, key: str):
    from safetensors import safe_open

    with safe_open(str(path), "pt") as st:
        return st.get_tensor(key)


def trained_weights(adapter_dir: Path) -> set:
    """Base-model tensor names the adapter changes, read from the adapter's own keys."""
    from safetensors import safe_open

    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as st:
        keys = list(st.keys())
    out = set()
    for k in keys:
        if ".lora_A." in k:
            module = k.split(".lora_A.")[0].removeprefix("base_model.model.")
            out.add(f"{module}.weight")
    return out


def compare(merged_dir: Path, base_dir: Path, trained: set) -> Dict[str, Any]:
    merged, base = _tensors(merged_dir), _tensors(base_dir)
    missing = sorted(set(base) - set(merged))
    if missing:
        raise SystemExit(f"merged model lacks {len(missing)} base tensors, e.g. {missing[:3]}")
    unknown = sorted(trained - set(base))
    if unknown:
        raise SystemExit(f"adapter names tensors the base model lacks, e.g. {unknown[:3]}")
    same_trained, changed_untrained, moved = [], [], []
    max_diff = 0.0
    for key in sorted(base):
        a, b = _load(merged[key], key), _load(base[key], key)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise SystemExit(f"{key}: merged {a.dtype}{tuple(a.shape)} vs base "
                             f"{b.dtype}{tuple(b.shape)}")
        diff = (a.float() - b.float()).abs()
        if key in trained:
            if diff.max().item() == 0:
                same_trained.append(key)
            moved.append((diff > 0).float().mean().item())
            max_diff = max(max_diff, diff.max().item())
        elif diff.max().item() != 0:
            changed_untrained.append(key)
    report = {"n_tensors": len(base), "n_trained": len(trained),
              "trained_identical_to_base": same_trained,
              "untrained_changed": changed_untrained,
              "max_abs_diff": max_diff,
              "moved_element_share_mean": sum(moved) / len(moved) if moved else 0.0,
              "moved_element_share_min": min(moved) if moved else 0.0}
    report["ok"] = bool(trained) and not same_trained and not changed_untrained
    return report


def export(run_dir: Path, checkpoint: Optional[Path] = None) -> Dict[str, Any]:
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run = json.loads((run_dir / "run.json").read_text())
    m = run["config"]["model"]
    if checkpoint is None:
        selection = json.loads((run_dir / SELECTION).read_text())
        checkpoint = Path(selection["best_model_checkpoint"])
    base_dir = Path(snapshot_download(m["name"], revision=m["revision"],
                                      allow_patterns=["*.json", "*.safetensors", "*.txt"]))
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(checkpoint)).merge_and_unload()
    out = run_dir / "merged"
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_dir).save_pretrained(out)
    del model
    report = compare(out, base_dir, trained_weights(checkpoint))
    report.update(checkpoint=str(checkpoint), merged_dir=str(out), base_dir=str(base_dir))
    (run_dir / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--checkpoint", type=Path, default=None)
    a = p.parse_args(argv)
    report = export(a.run_dir, a.checkpoint)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("trained_identical_to_base", "untrained_changed")}, indent=2))
    if not report["ok"]:
        print(f"FAIL: {len(report['trained_identical_to_base'])} trained tensors equal base, "
              f"{len(report['untrained_changed'])} untrained tensors changed")
        return 1
    print("ok: every trained tensor differs from base; every other tensor is identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
