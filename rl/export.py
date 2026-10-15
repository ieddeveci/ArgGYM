"""Merge the selected adapter into the base model, and prove the merged files are not the base.

    uv run python -m rl.export outputs/rl/runs/<run>            # the checkpoint selection.json names
    uv run python -m rl.export outputs/rl/runs/<run> --checkpoint outputs/rl/runs/<run>/checkpoint-20

The merged model is written to `<run>/merged` as a copy of the pinned base
checkpoint: the same files, tensor names and config, with every tensor the
adapter trained replaced by base plus adapter. `vllm serve <run>/merged` loads
it with no LoRA flag and with the flags the base is served with
(`--language-model-only` for Qwen3.5), so nothing can serve the base model while
claiming to serve the adapter. Serve it with
`--served-model-name <the eval config's model>`: vLLM otherwise registers the
directory path as the name and the eval harness's first request is refused.

Training loads the text-only model class, whose tensors are named
`model.layers...` where a vision-language checkpoint such as Qwen3.5's says
`model.language_model.layers...`. `to_checkpoint` maps one onto the other; the
weight sync to vLLM during training uses the same map.

The check then reads the files back from disk, the bytes vLLM would load, and
compares every tensor against the base checkpoint: every tensor the adapter
trained must differ from base and every other tensor must be identical. Either
failure exits non-zero. `export.json` records the numbers, including the share
of elements that moved: the merge rounds the LoRA delta into BF16, and an update
smaller than half a BF16 step at that weight is lost from the served model.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

SELECTION = "selection.json"

#: Text-model tensor prefixes, and where a vision-language checkpoint keeps them.
_PREFIXES = (("model.", "model.language_model."),)


def to_checkpoint(name: str, keys: Iterable[str]) -> Optional[str]:
    """The checkpoint's name for a text-model tensor, or None if it has none."""
    if name in keys:
        return name
    for old, new in _PREFIXES:
        if name.startswith(old) and new + name[len(old):] in keys:
            return new + name[len(old):]
    return None


def base_snapshot(name: str, revision: str) -> Path:
    """The pinned checkpoint's directory: the cached copy if there is one.

    The cache already holds what training loaded. Asking the Hub again would try
    to fetch files training never needed (a README, say), and fails where the
    cache is not writable, as it is not on this machine.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(snapshot_download(name, revision=revision, local_files_only=True))
    except LocalEntryNotFoundError:
        return Path(snapshot_download(name, revision=revision))


def checkpoint_keys(name: str, revision: str) -> set:
    """Every tensor name in a Hub checkpoint, from its safetensors files."""
    return set(_tensors(base_snapshot(name, revision)))


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
    """Text-model tensor names the adapter changes, read from the adapter's own keys."""
    from safetensors import safe_open

    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as st:
        keys = list(st.keys())
    out = set()
    for k in keys:
        if ".lora_A." in k:
            module = k.split(".lora_A.")[0].removeprefix("base_model.model.")
            out.add(f"{module}.weight")
    return out


def write_merged(state: Dict[str, Any], base_dir: Path, out: Path, trained: set) -> None:
    """The base checkpoint's files, with the `trained` tensors of `state` written over them.

    Only trained tensors are taken from the merged model; every other tensor is
    copied from the checkpoint as stored. Loading the model in BF16 casts a few
    checkpoint tensors kept in FP32 (Qwen3.5's `A_log`), and writing those back
    would change weights training never touched.
    """
    from safetensors.torch import save_file

    where = _tensors(base_dir)
    replace = {}
    for name in sorted(trained):
        key = to_checkpoint(name, where)
        if key is None:
            raise SystemExit(f"{name} has no tensor in the base checkpoint {base_dir}")
        stored = _load(where[key], key)
        if state[name].shape != stored.shape or state[name].dtype != stored.dtype:
            raise SystemExit(f"{name}: merged {state[name].dtype}{tuple(state[name].shape)} "
                             f"vs checkpoint {stored.dtype}{tuple(stored.shape)}")
        replace[key] = state[name]
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(base_dir.iterdir()):
        if f.suffix == ".safetensors":
            names = [k for k, p in where.items() if p == f]
            save_file({k: (replace[k] if k in replace else _load(f, k)).contiguous()
                       for k in names}, str(out / f.name), metadata={"format": "pt"})
        elif f.is_file():
            shutil.copyfile(f, out / f.name)


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
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    run = json.loads((run_dir / "run.json").read_text())
    m = run["config"]["model"]
    if checkpoint is None:
        selection = json.loads((run_dir / SELECTION).read_text())
        checkpoint = Path(selection["best_model_checkpoint"])
    base_dir = base_snapshot(m["name"], m["revision"])
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(checkpoint)).merge_and_unload()
    out = run_dir / "merged"
    trained = trained_weights(checkpoint)
    write_merged(model.state_dict(), base_dir, out, trained)
    del model
    keys = set(_tensors(base_dir))
    report = compare(out, base_dir, {to_checkpoint(n, keys) for n in trained})
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
