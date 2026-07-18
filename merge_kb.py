#!/usr/bin/env python3

import sys
import json
import argparse

sys.path.insert(0, ".")
from audit_kb import audit_record                   


def _score(rec, stats):
    return (stats["verified_args"] - stats["failed_args"],
            stats["n_atoms"], stats["n_neg_rules"])


def _load(path):
    with open(path) as f:
        kb = json.load(f)
    return kb if isinstance(kb, list) else list(kb.values())


def merge(paths, min_verified=1, keep_all_variants=False):
    best = {}                      
    report = {"inputs": {}, "dropped_invalid": [], "dedup_collisions": [], "kept": 0}
    for path in paths:
        recs = _load(path)
        report["inputs"][path] = len(recs)
        for i, rec in enumerate(recs):
            ok, issues, stats = audit_record(rec, i)
            if not ok or stats["verified_args"] < min_verified:
                report["dropped_invalid"].append(
                    {"source": path, "index": i, "claim": rec.get("claim", "")[:60],
                     "reason": (issues[:1] or [f"verified_args<{min_verified}"])[0]})
                continue
            claim = rec.get("claim", "")
            key = f"{claim}  [{path}#{i}]" if keep_all_variants else claim
            sc = _score(rec, stats)
            if key in best:
                report["dedup_collisions"].append(
                    {"claim": claim[:50], "kept_source": best[key][2], "other_source": path})
                if sc > best[key][0]:
                    best[key] = (sc, rec, path)
            else:
                best[key] = (sc, rec, path)

    merged = [v[1] for _, v in sorted(best.items(), key=lambda kv: kv[0])]
    report["kept"] = len(merged)
    return merged, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--min-verified", type=int, default=1,
                    help="drop records with fewer than N engine-verified arguments")
    ap.add_argument("--keep-all-variants", action="store_true",
                    help="keep same-claim records from different sources (tagged), not just the best")
    args = ap.parse_args()

    merged, report = merge(args.inputs, args.min_verified, args.keep_all_variants)
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=1)

    print(f"merge -> {args.out}")
    for path, n in report["inputs"].items():
        print(f"  input {path}: {n} records")
    print(f"  kept:            {report['kept']}")
    print(f"  dropped invalid: {len(report['dropped_invalid'])}")
    print(f"  dedup collisions:{len(report['dedup_collisions'])}")
    if report["dropped_invalid"]:
        print("  first dropped:")
        for d in report["dropped_invalid"][:5]:
            print(f"    {d['source']}#{d['index']} \"{d['claim']}\": {d['reason']}")
    print(f"\nRun: python3 audit_kb.py {args.out} --strict   # to confirm the corpus")


if __name__ == "__main__":
    main()
