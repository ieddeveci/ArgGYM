from __future__ import annotations

import json
import re
import sys
from pathlib import Path


AUDIT_ROOT = Path(__file__).parent
V2_ROOT = AUDIT_ROOT / "repo" / "ArgGYM_v2"
if not V2_ROOT.exists():
    raise SystemExit(f"fresh v2 clone is missing: {V2_ROOT}")
sys.path.insert(0, str(V2_ROOT.resolve()))

from aspic.api import ASPICVerifier  # noqa: E402
from aspic.dsl import parse_dsl  # noqa: E402
from aspic.engine import Operation  # noqa: E402
from core.scoring import parse_answer, score_item  # noqa: E402
from tasks import claim_chain, counter_argument, formalization, status_query  # noqa: E402


def probe_claim_chain_order() -> dict:
    item = claim_chain.make_item(9, 0, "last_link_elitist")
    assert item is not None
    match = re.search(r"\[answer\](.*?)\[/answer\]", item.reference, re.S)
    assert match is not None
    lines = [line for line in match.group(1).splitlines() if line.strip()]
    answer = "[answer]\n" + "\n".join(reversed(lines)) + "\n[/answer]"
    result = claim_chain.score(answer, item)
    return {
        "name": "claim_chain_order",
        "line_count": len(lines),
        "score": result["score"],
        "correct_order": result["correct_order"],
    }


def probe_rule_arrow() -> dict:
    item = counter_argument.make_item(3, 0, "last_link_elitist")
    assert item is not None and "=>" in item.reference
    answer = item.reference.replace("=>", "->", 1)
    canonical = parse_dsl("[defeasible w: aa0 -> bb0]")
    scorer_parse = parse_answer("[answer]\n[defeasible w: aa0 -> bb0]\n[/answer]")
    result = score_item(answer, counter_argument.as_score_input(item))
    return {
        "name": "rule_arrow",
        "canonical_operations": len(canonical.operations),
        "canonical_drop_reason": canonical.dropped[0]["reason"],
        "scorer_operations": len(scorer_parse.ops),
        "mutated_reference_score": result["score"],
    }


def probe_formalization_preference() -> dict:
    item = formalization.make_item(15, 0, "weakest_link_elitist")
    assert item is not None
    gold = "[prefer_rule: q_14 > q_15]"
    reversed_preference = "[prefer_rule: q_15 > q_14]"
    assert gold in item.reference
    answer = item.reference.replace(gold, reversed_preference, 1)
    result = formalization.score(answer, item)
    return {
        "name": "formalization_preference",
        "gold": gold,
        "submitted": reversed_preference,
        "score": result["score"],
        "behavioural": result["behavioural"],
        "shape_f1": result["shape_f1"],
        "type_score": result["type_score"],
    }


def probe_status_grid() -> dict:
    missing = []
    for level in (3, 6, 9, 12, 15):
        for ordering in ("last_link_elitist", "weakest_link_elitist"):
            for seed in (0, 1):
                if status_query.make_item(level, seed, ordering) is None:
                    missing.append([level, ordering, seed])
    return {"name": "status_grid", "expected": 20, "missing": missing}


def probe_counter_argument_names() -> dict:
    rows = []
    for level in (3, 6):
        for seed in (0, 1):
            for ordering in ("last_link_elitist", "weakest_link_elitist"):
                item = counter_argument.make_item(level, seed, ordering)
                assert item is not None
                names = [
                    op.name
                    for op in item.base_ops
                    if op.kind in ("defeasible", "strict") and op.name
                ]
                filler = sorted(name for name in names if name.startswith("lx_"))
                rows.append(
                    {
                        "level": level,
                        "seed": seed,
                        "ordering": ordering,
                        "filler_names": filler,
                        "filler_in_reference": any(
                            name in item.reference for name in filler
                        ),
                    }
                )
    return {"name": "counter_argument_names", "cells": rows}


def probe_overruled_definition() -> dict:
    ops = [
        Operation(kind="premise", content="aa0"),
        Operation(kind="premise", content="bb0"),
        Operation(
            kind="defeasible",
            name="r1",
            antecedents=("aa0",),
            consequent="qq0",
        ),
        Operation(
            kind="defeasible",
            name="r2",
            antecedents=("bb0",),
            consequent="-r1",
        ),
    ]
    verifier = ASPICVerifier.from_operations(ops)
    preferred = verifier.preferred_conclusions()
    return {
        "name": "overruled_definition",
        "engine_status": str(verifier.status("qq0")),
        "grounded_conclusions": verifier.justified(),
        "claim_in_preferred": any("qq0" in extension for extension in preferred),
        "contrary_in_preferred": any("-qq0" in extension for extension in preferred),
        "preferred_conclusions": [sorted(extension) for extension in preferred],
    }


def main() -> None:
    results = [
        probe_claim_chain_order(),
        probe_rule_arrow(),
        probe_formalization_preference(),
        probe_status_grid(),
        probe_counter_argument_names(),
        probe_overruled_definition(),
    ]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
