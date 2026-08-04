from __future__ import annotations

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aspic.engine import Operation
from core.scoring import score_item

LAST_LINK, WEAKEST_LINK = "last_link_elitist", "weakest_link_elitist"


_EXPORTABLE = {
    "preference_construction": "prefcon",
    "counter_argument": "counterarg",
    "counter_argument_strict": "counterarg",
    "claim_chain": "claimchain",
    "defeat_diagnosis": "defeatdiag",
    "formalization": "formalize",
    "status_query": "statusquery",
}


def _export_row(task: str, lv: int, o: str, s: int):
    from tasks import preference_construction as prefcon
    from tasks import (counter_argument as counterarg, claim_chain as claimchain,
                       defeat_diagnosis as defeatdiag, formalization as formalize,
                       status_query as statusquery)
    if task == "preference_construction":
        it = prefcon.make_item(lv, s, o)
        if it is None:
            return None
        return it, score_item(it.reference, it.as_score_input())["score"], \
            [dict(g) for g in it.goals], it.min_directives
    if task.startswith("counter_argument"):
        it = counterarg.make_item(lv, s, o, allow_strict=task.endswith("_strict"))
        if it is None:
            return None
        return it, score_item(it.reference, counterarg.as_score_input(it))["score"], \
            [{"claim": "-" + it.target, "want": "JUSTIFIED"},
             {"claim": it.target, "want": "OVERRULED"}], it.min_directives
    if task == "claim_chain":
        it = claimchain.make_item(lv, s, o)
        if it is None:
            return None
        return it, claimchain.score(it.reference, it)["score"], \
            [{"claim": it.claim, "want": "trace the justifying line"}], it.metadata["line_length"]
    if task == "defeat_diagnosis":
        it = defeatdiag.make_item(lv, s, o)
        if it is None:
            return None
        return it, defeatdiag.score(it.reference, it)["score"], \
            [{"claim": it.claim, "want": it.claim_status}], len(it.diagnoses)
    if task == "formalization":
        it = formalize.make_item(lv, s, o)
        if it is None:
            return None
        return it, formalize.score(it.reference, it)["score"], \
            [{"claim": l, "want": it.gold_status[l]} for l in it.queried], \
            it.metadata["n_directives"]
    if task == "status_query":
        it = statusquery.make_item(lv, s, o)
        if it is None:
            return None
        return it, statusquery.score(it.reference, it)["score"], \
            [{"claim": c, "want": it.gold[c]} for c in it.queried], it.metadata["n_queried"]
    return None


def export_task(path: str, task: str, levels=(3, 6, 9, 12, 15), seeds=(0, 1)) -> None:
    if task not in _EXPORTABLE:
        raise SystemExit(f"unknown task {task}; known: {sorted(_EXPORTABLE)}")
    rows = []
    for lv in levels:
        for o in (LAST_LINK, WEAKEST_LINK):
            for s in seeds:
                got = _export_row(task, lv, o, s)
                if got is None:
                    continue
                it, refscore, goals, mind = got
                rows.append({"task": task, "level": lv, "ordering": o, "seed": s,
                             "prompt": it.prompt, "reference": it.reference,
                             "goals": goals, "min_directives": mind,
                             "metadata": it.metadata, "reference_score": refscore})
    h = hashlib.blake2b(digest_size=16)
    for r in rows:
        h.update((r["prompt"] + r["reference"]
                  + json.dumps(r["metadata"], sort_keys=True, default=str)).encode())
    manifest = {"task": task, "taskset_hash": h.hexdigest(), "n_items": len(rows),
                "n_valid": sum(1 for r in rows if r["reference_score"] >= 0.999),
                "levels": list(levels), "orderings": [LAST_LINK, WEAKEST_LINK],
                "python": sys.version.split()[0],
                "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
                "minimum_caveat": ("min_directives is minimal among the candidate directives the "
                                   "generator produced, not proven globally minimal")}
    with open(path, "w") as f:
        f.write(json.dumps({"__manifest__": manifest}) + "\n")
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"wrote {len(rows)} items ({manifest['n_valid']} valid) -> {path}")
    print(f"taskset_hash {manifest['taskset_hash']}")


