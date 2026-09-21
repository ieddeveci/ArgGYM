"""Audit Qwen baseline completion-token lengths from an existing eval run.

Reads ``generations.jsonl`` only. This is a PRE-RL systems audit, not model
selection. It helps freeze ``max_completion_length`` before the main run.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence


def percentile(xs: list[int], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    pos = (len(ys) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ys[lo])
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    a = p.parse_args(argv)
    path = Path(a.run_dir) / "generations.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    lengths: list[int] = []
    truncated = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n = (r.get("usage") or {}).get("completion_tokens")
            if n is not None:
                lengths.append(int(n))
            truncated += int(bool(r.get("truncated")))
    if not lengths:
        raise SystemExit("no completion_tokens found")

    print(f"n={len(lengths)} truncated_flags={truncated}")
    for q in (0.50, 0.90, 0.95, 0.975, 0.99, 0.995, 1.0):
        print(f"p{q*100:g}={percentile(lengths, q):.1f}")
    for cap in (12288, 16384, 20480, 24576, 32768):
        n = sum(x > cap for x in lengths)
        print(f">{cap}: {n}/{len(lengths)} = {100*n/len(lengths):.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
