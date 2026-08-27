"""Focused probes for the v2 ladder-and-coverage audit.

Run with the Python environment created inside the fresh owner-main clone:

    repo/.venv/bin/python probes.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


AUDIT_DIR = Path(__file__).parent
PROJECT_ROOT = AUDIT_DIR.parent.parent
REPO = AUDIT_DIR / "repo"
V2 = REPO / "ArgGYM_v2"
LEVELS = (3, 6, 9, 12, 15)
PILOT = PROJECT_ROOT / "data/tasksets/v2-pilot-20260804T220828Z-e89f0b8a/taskset.jsonl"
FIXED = PROJECT_ROOT / "data/tasksets/v2-fixed-20260817T143923Z-ea3fd430/taskset.jsonl"


def read_rows(path: Path) -> list[dict]:
    assert path.exists(), f"missing taskset: {path}"
    return [json.loads(line) for line in path.read_text().splitlines()]


def show(revision: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def historical_taskset_probes() -> None:
    pilot = read_rows(PILOT)
    fixed = read_rows(FIXED)

    pilot_pc = [row for row in pilot if row["task"] == "preference_construction"]
    print("A1 pilot shared conflicts",
          sum(row["metadata"]["shared_conflict"] for row in pilot_pc),
          "/", len(pilot_pc))
    for taskset_name, rows in (("pilot", pilot), ("fixed", fixed)):
        pc = [row for row in rows if row["task"] == "preference_construction"]
        for ordering in ("last_link_elitist", "weakest_link_elitist"):
            gaps = defaultdict(list)
            for row in pc:
                if row["ordering"] == ordering:
                    md = row["metadata"]
                    gaps[row["level"]].append(
                        md["n_claims"] - md["min_within_reference"]
                    )
            print(f"A1 {taskset_name} mean goal-minus-min gap", ordering,
                  {level: round(mean(gaps[level]), 3) for level in LEVELS})

    recipes: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    keys = ("n_attackers", "n_strict_attackers", "support_depth", "attacker_depth",
            "n_decoys", "n_decoy_strict")
    for row in pilot:
        if row["task"] == "defence" and row["level"] in (6, 9, 12, 15):
            recipes[row["level"]].add(tuple(row["metadata"][key] for key in keys))
    print("A2/A3 pilot defence recipes", dict(recipes))

    costs: dict[int, list[int]] = defaultdict(list)
    for row in pilot:
        if row["task"] == "counter_argument_strict" and row["level"] in (3, 6):
            costs[row["level"]].append(row["metadata"]["min_within_reference"])
    print("A4 pilot mean directives", {level: mean(values) for level, values in costs.items()})

    rule_re = re.compile(r"\[(?:defeasible|strict) [^:\]]+: [^\]\n]* AND [^\]\n]*(?:=>|->)")
    for name, rows in (("pilot", pilot), ("fixed", fixed)):
        all_count = sum(
            len(rule_re.findall(row.get(field, "")))
            for row in rows
            for field in ("prompt", "reference")
        )
        counts = {
            task: sum(
                len(rule_re.findall(row["prompt"] if task == "status_query" else row["reference"]))
                for row in rows
                if row["task"] == task
            )
            for task in ("status_query", "formalization")
        }
        print(f"A5 {name} conjunctive DSL occurrences across all tasks", all_count)
        print(f"A5 {name} conjunctive rules", counts)


def current_main_probes() -> None:
    sys.path.insert(0, str(V2))
    from tasks import attack_defense, counter_argument, formalization, status_query

    flags = {level: level >= 5 and level % 3 == 2 for level in LEVELS}
    print("A1 current shared flag", flags)

    defence = {}
    for level in LEVELS:
        item = attack_defense.make_item(
            level,
            0,
            "last_link_elitist",
            mode=attack_defense.DEFENCE,
        )
        assert item is not None, f"defence L{level} did not generate"
        md = item.metadata
        defence[level] = tuple(
            md[key]
            for key in (
                "n_attackers",
                "n_strict_attackers",
                "support_depth",
                "attacker_depth",
                "n_decoys",
                "n_decoy_strict",
            )
        )
    print("A2/A3 current defence recipes", defence)

    means = {}
    for level in (3, 6):
        items = [
            counter_argument.make_item(
                level,
                seed,
                "last_link_elitist",
                allow_strict=True,
            )
            for seed in range(8)
        ]
        assert all(item is not None for item in items), f"counter_argument_strict L{level} failed"
        means[level] = mean(item.min_directives for item in items if item is not None)
    print("A4 current 8-seed last-link mean directives", means)

    for name, module, attr in (
        ("status_query", status_query, "base_ops"),
        ("formalization", formalization, "reference_ops"),
    ):
        items = [module.make_item(15, seed, "last_link_elitist") for seed in range(8)]
        assert all(item is not None for item in items), f"{name} L15 failed"
        count = sum(
            1
            for item in items
            for op in getattr(item, attr)
            if op.kind in ("strict", "defeasible") and len(op.antecedents) > 1
        )
        print("A5 current L15 conjunctive rules", name, count, "across", len(items), "items")


def harness_probes() -> None:
    before = show("b7df560^", "evals/client.py")
    after = show("b7df560", "evals/client.py")
    namespace_before: dict = {}
    namespace_after: dict = {}
    exec(compile(before, "b7df560^:evals/client.py", "exec"), namespace_before)
    exec(compile(after, "b7df560:evals/client.py", "exec"), namespace_after)
    sampling = {"temperature": 0.6, "repetition_penalty": 1.2}
    args = ("model", "prompt", sampling, 100)
    old_payload = namespace_before["build_payload"](*args)
    new_payload = namespace_after["build_payload"](*args)
    print("B1 old payload keys", sorted(old_payload))
    print("B1 fixed payload keys", sorted(new_payload))

    old_driver = show("9fc9556^", "evals/run_grid.sh")
    new_driver = show("9fc9556", "evals/run_grid.sh")
    print("B2 old driver health check", "ENDPOINT_HEALTH" in old_driver,
          "breaks after high-error gate", "api_error_rate=$err -- server died" in old_driver)
    print("B2 fixed driver health check", "ENDPOINT_HEALTH" in new_driver,
          "higherror marker", ".higherror" in new_driver)


if __name__ == "__main__":
    historical_taskset_probes()
    current_main_probes()
    harness_probes()
