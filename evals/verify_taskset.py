"""Check that a shipped taskset still matches what the generator produces.

Regenerates the grid from the taskset's own config.yaml and compares the content
hash. Catches silent drift: a change to the generator, the difficulty schedule,
or kb.json will alter prompts, and a benchmark whose questions moved is not the
same benchmark.

    python -m evals.verify_taskset data/tasksets/pilot-650388ac
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import artifacts  # noqa: E402
from evals.taskset import build_rows, load_taskset, sha256_file, taskset_hash  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("taskset_dir")
    a = ap.parse_args()

    d = Path(a.taskset_dir)
    root = Path(__file__).resolve().parent.parent
    manifest = artifacts.read_json(d / "manifest.json") or {}
    cfg = OmegaConf.load(d / "config.yaml")

    kb = root / "kb.json"
    kb_sha = sha256_file(kb) if kb.exists() else None
    print(f"taskset : {manifest.get('taskset_id')}")
    print(f"expected: {manifest.get('taskset_hash')}")

    if kb_sha != manifest.get("kb_sha256"):
        print("\nkb.json differs from the one this taskset was built against:")
        print(f"  shipped: {manifest.get('kb_sha256')}")
        print(f"  local  : {kb_sha}")
        print("Content-mode prompts are drawn from kb.json, so regeneration "
              "cannot reproduce this taskset. The shipped file remains usable.")
        sys.exit(2)

    print("\nregenerating (a few minutes)...")
    rows = build_rows(list(cfg.tasks), list(cfg.modes), list(cfg.levels),
                      int(cfg.n), int(cfg.base_seed), bool(cfg.validate_gold))
    got = taskset_hash(rows, kb_sha)
    print(f"regenerated: {got}")

    if got == manifest.get("taskset_hash"):
        print(f"\nMATCH -- {len(rows)} samples reproduce exactly.")
        return

    shipped = {r["sample_id"]: r["prompt"] for r in load_taskset(d)}
    fresh = {r["sample_id"]: r["prompt"] for r in rows}
    only_shipped = sorted(set(shipped) - set(fresh))
    only_fresh = sorted(set(fresh) - set(shipped))
    changed = sorted(k for k in set(shipped) & set(fresh) if shipped[k] != fresh[k])
    print(f"\nMISMATCH: {len(changed)} prompts changed, "
          f"{len(only_shipped)} missing, {len(only_fresh)} new")
    for k in (changed or only_shipped or only_fresh)[:5]:
        print(f"  {k}")
    if manifest.get("git_dirty"):
        print(f"\nThis taskset was built from a dirty working tree, so its recorded "
              f"git_sha ({(manifest.get('git_sha') or '?')[:8]}) is not the code that "
              f"produced it and no commit regenerates it. The mismatch above is "
              f"expected and says nothing about the current generator. Rebuild from "
              f"a clean tree to get a taskset that verifies.")
    else:
        print(f"\nThis taskset was built clean at {(manifest.get('git_sha') or '?')[:8]}, "
              f"so the generator has changed since. Either the change was intended -- "
              f"rebuild and re-run -- or it is drift to investigate.")
    sys.exit(1)


if __name__ == "__main__":
    main()
