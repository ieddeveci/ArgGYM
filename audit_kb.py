#!/usr/bin/env python3
import sys
import json
import argparse
from collections import Counter

sys.path.insert(0, ".")
from aspic_engine import Operation                  
from aspic_api import ASPICVerifier                 

REQUIRED_RECORD_KEYS = {"claim", "atoms", "support", "disclaim"}
REQUIRED_ATOM_KEYS = {"pos", "neg"}


def _contra(x):
    return x[1:] if x.startswith("-") else "-" + x


def _arg_to_ops(atoms, arg):
    ops = []
    for lit in arg.get("premises", []):
        ops.append(Operation(kind="premise", content=lit))
    for lit in arg.get("axioms", []):
        base = lit.lstrip("-")
        kind = "axiom" if atoms.get(base, {}).get("axiomatic") else "premise"
        ops.append(Operation(kind=kind, content=lit))
    for r in arg.get("rules", []):
        ops.append(Operation(kind="strict" if r.get("strict") else "defeasible",
                             name=r.get("name"), antecedents=tuple(r["antecedents"]),
                             consequent=r["consequent"]))
    return ops


def audit_record(rec, idx):
    issues = []
    stats = {"n_atoms": 0, "n_support": 0, "n_disclaim": 0, "n_attacks": 0,
             "has_prem_prefs": False, "n_neg_rules": 0, "verified_args": 0, "failed_args": 0}

    missing = REQUIRED_RECORD_KEYS - set(rec)
    if missing:
        issues.append(f"missing keys: {sorted(missing)}")
        return False, issues, stats

    atoms = rec["atoms"]
    stats["n_atoms"] = len(atoms)
    if "c0" not in atoms:
        issues.append("no c0 atom (the claim)")
    for aid, a in atoms.items():
        if not REQUIRED_ATOM_KEYS <= set(a):
            issues.append(f"atom {aid} missing pos/neg gloss")
            break

    stats["n_support"] = len(rec.get("support", []))
    stats["n_disclaim"] = len(rec.get("disclaim", []))
    stats["n_attacks"] = len(rec.get("attacks", []))
    stats["has_prem_prefs"] = bool(rec.get("premise_preferences"))

    for arg in rec.get("support", []) + rec.get("disclaim", []):
        for r in arg.get("rules", []):
            if r["consequent"].startswith("-"):
                stats["n_neg_rules"] += 1

    for stance in ("support", "disclaim"):
        for j, arg in enumerate(rec.get(stance, [])):
            concl = arg.get("conclusion")
            if concl is None:
                issues.append(f"{stance}[{j}] has no conclusion")
                stats["failed_args"] += 1
                continue
            try:
                v = ASPICVerifier.from_operations(_arg_to_ops(atoms, arg),
                                                  ordering="last_link_elitist")
                st = str(v.status(concl))
            except Exception as e:
                issues.append(f"{stance}[{j}] engine error: {type(e).__name__}")
                stats["failed_args"] += 1
                continue
            if st == "JUSTIFIED":
                stats["verified_args"] += 1
            else:
                issues.append(f"{stance}[{j}] concludes {concl} but engine says {st}")
                stats["failed_args"] += 1

    return (stats["failed_args"] == 0 and not issues), issues, stats


def audit_kb(kb):
    per = []
    agg = Counter()
    claims = Counter()
    for i, rec in enumerate(kb):
        ok, issues, stats = audit_record(rec, i)
        claims[rec.get("claim", f"<record {i}>")] += 1
        per.append({"index": i, "claim": rec.get("claim", "")[:70], "ok": ok,
                    "issues": issues, "stats": stats})
        agg["records"] += 1
        agg["ok"] += int(ok)
        for k in ("n_atoms", "n_support", "n_disclaim", "n_attacks",
                  "n_neg_rules", "verified_args", "failed_args"):
            agg[k] += stats[k]
        agg["with_prem_prefs"] += int(stats["has_prem_prefs"])
    dupes = {c: n for c, n in claims.items() if n > 1}
    return {"per_record": per, "aggregate": dict(agg), "duplicate_claims": dupes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kb")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    with open(args.kb) as f:
        kb = json.load(f)
    if not isinstance(kb, list):
        kb = list(kb.values())

    report = audit_kb(kb)
    agg = report["aggregate"]

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        n = agg["records"]
        print(f"KB audit: {args.kb}")
        print(f"  records:            {n}")
        print(f"  well-formed+verified: {agg['ok']}/{n}")
        print(f"  duplicate claims:   {len(report['duplicate_claims'])}")
        print(f"  atoms/record:       {agg['n_atoms'] / n:.1f} avg")
        print(f"  support args:       {agg['n_support']} (verified {agg['verified_args']}, "
              f"failed {agg['failed_args']})")
        print(f"  disclaim args:      {agg['n_disclaim']}")
        print(f"  attacks:            {agg['n_attacks']}")
        print(f"  neg-concluding rules: {agg['n_neg_rules']}  (structural-negation supply)")
        print(f"  records w/ premise prefs: {agg['with_prem_prefs']}/{n}")
        bad = [r for r in report["per_record"] if not r["ok"]]
        if bad:
            print(f"\n  {len(bad)} record(s) with issues:")
            for r in bad[:10]:
                print(f"    #{r['index']} \"{r['claim']}\": {r['issues'][:2]}")
            if len(bad) > 10:
                print(f"    ... and {len(bad) - 10} more")

    if args.strict and agg["ok"] != agg["records"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
