"""Check that a shipped v2 taskset still matches what the generator produces.

Regenerates the grid from the taskset's own manifest and compares the content
hash. A change to a v2 generator, curriculum or prompt alters the questions, and
a benchmark whose questions moved is not the same benchmark -- so the number a
report quotes must name a taskset that can still be rebuilt from source.

    python -m evals.verify_v2_taskset data/tasksets/v2-pilot-...-01deb2a1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals import artifacts  # noqa: E402
from evals.v2_taskset import build_rows, load_taskset, taskset_hash  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("taskset_dir")
    a = ap.parse_args()

    d = Path(a.taskset_dir)
    m = artifacts.read_json(d / "manifest.json") or {}
    if not m:
        sys.exit(f"no manifest.json in {d}")

    print(f"taskset : {m.get('taskset_id')}")
    print(f"expected: {m.get('taskset_hash')}")
    print(f"built at: {m.get('git_sha')} (dirty={m.get('git_dirty')})")

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    if m.get("git_sha") and head != m["git_sha"]:
        print(f"\nnote: HEAD is {head}, taskset was built at {m['git_sha']}.\n"
              "      A mismatch here explains a mismatch below.")
    if m.get("git_dirty"):
        print("\nnote: built from a dirty tree, so the recorded git_sha does "
              "not fully describe the code that produced it.")

    print("\nregenerating...", flush=True)
    rows, rejected = build_rows(tuple(m["tasks"]), tuple(m["levels"]),
                                int(m["seeds_per_cell"]))
    got = taskset_hash(rows)
    print(f"regenerated: {got}")

    if got == m.get("taskset_hash"):
        print(f"\nMATCH -- {len(rows)} samples reproduce exactly.")
        return

    shipped = {r["sample_id"]: r for r in load_taskset(d)}
    fresh = {r["sample_id"]: r for r in rows}
    only_shipped = sorted(set(shipped) - set(fresh))
    only_fresh = sorted(set(fresh) - set(shipped))
    changed = [k for k in sorted(set(shipped) & set(fresh))
               if shipped[k]["prompt"] != fresh[k]["prompt"]
               or shipped[k]["reference"] != fresh[k]["reference"]]

    print(f"\nMISMATCH -- shipped {len(shipped)}, regenerated {len(fresh)}")
    if only_shipped:
        print(f"  {len(only_shipped)} only in shipped, e.g. {only_shipped[:3]}")
    if only_fresh:
        print(f"  {len(only_fresh)} only in regenerated, e.g. {only_fresh[:3]}")
    if changed:
        print(f"  {len(changed)} items changed, e.g. {changed[:3]}")
        k = changed[0]
        for field in ("prompt", "reference"):
            if shipped[k][field] != fresh[k][field]:
                print(f"\n  first diff in {k} ({field}):")
                print(f"    shipped: {shipped[k][field][:200]!r}")
                print(f"    fresh  : {fresh[k][field][:200]!r}")
                break
    sys.exit(1)


if __name__ == "__main__":
    main()
