from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from aspic.engine import Operation
from aspic.api import ASPICVerifier

BLOAT_FACTOR = 2
PARTIAL_CAP = 0.25

_ANSWER = re.compile(r"\[answer\](.*?)\[/answer\]", re.S | re.I)
_PREMISE = re.compile(r"^\[(premise|axiom)\s*:\s*(-?[A-Za-z]\w*)\]$")
_RULE = re.compile(r"^\[(defeasible|strict)\s*([A-Za-z]\w*)?\s*:\s*(.+?)\s*(=>|->)\s*(-?[A-Za-z]\w*)\]$")
_PREF = re.compile(r"^\[prefer_(rule|premise)\s*:\s*(-?[A-Za-z]\w*)\s*>\s*(-?[A-Za-z]\w*)\]$")


@dataclass
class ParsedAnswer:
    ops: List[Operation] = field(default_factory=list)
    n_lines: int = 0
    n_unparseable: int = 0
    unparseable_examples: List[str] = field(default_factory=list)
    no_region: bool = False


def parse_answer(text: str) -> ParsedAnswer:
    out = ParsedAnswer()
    m = _ANSWER.search(text or "")
    if m is None:
        out.no_region = True
        return out
    auto = 0
    body = m.group(1)
    units = re.findall(r"\[[^\]]*\]", body)
    leftover = re.sub(r"\[[^\]]*\]", " ", body)
    stray = [t for t in leftover.split()
             if t.strip(",;.-*\u2022()[]") and not re.fullmatch(r"\d+[.)]?", t)]
    for raw in units + stray:
        line = raw.strip()
        if not line:
            continue
        out.n_lines += 1
        mm = _PREMISE.match(line)
        if mm:
            out.ops.append(Operation(kind=mm.group(1), content=mm.group(2)))
            continue
        mm = _RULE.match(line)
        if mm:
            auto += 1
            name = mm.group(2) or f"m{auto}"
            ants = tuple(a.strip() for a in mm.group(3).split("AND") if a.strip())
            if not ants:
                out.n_unparseable += 1
                out.unparseable_examples.append(line[:60])
                continue
            out.ops.append(Operation(kind=mm.group(1), name=name,
                                     antecedents=ants, consequent=mm.group(5)))
            continue
        mm = _PREF.match(line)
        if mm:
            out.ops.append(Operation(
                kind="prefer_rule" if mm.group(1) == "rule" else "prefer_premise",
                stronger=mm.group(2), weaker=mm.group(3)))
            continue
        out.n_unparseable += 1
        if len(out.unparseable_examples) < 3:
            out.unparseable_examples.append(line[:60])
    return out


def check_legality(ops: Sequence[Operation], base_ops: Sequence[Operation],
                   allow_strict: bool = False,
                   prefs_only: bool = False) -> Tuple[List[Operation], List[str]]:
    ordinary = {o.content for o in base_ops if o.kind == "premise"}
    axioms = {o.content for o in base_ops if o.kind == "axiom"}
    kept: List[Operation] = []
    reasons: List[str] = []
    for o in ops:
        if o.kind == "strict" and not allow_strict:
            reasons.append(f"illegal_strict_rule:{o.name or '?'}")
            continue
        if prefs_only and o.kind not in ("prefer_rule", "prefer_premise"):
            reasons.append(f"illegal_non_preference:{o.kind}")
            continue
        if o.kind in ("premise", "axiom"):
            c = o.content
            if o.kind == "axiom":
                reasons.append(f"illegal_new_axiom:{c}")
                continue
            if not c.startswith("-"):
                reasons.append(f"illegal_new_premise:{c}")
                continue
            base = c[1:]
            if base in axioms:
                reasons.append(f"illegal_undermine_axiom:{c}")
                continue
            if base not in ordinary:
                reasons.append(f"illegal_asserted_contrary:{c}")
                continue
        kept.append(o)
    return kept, reasons


def _apply(base_ops: Sequence[Operation], added: Sequence[Operation]):
    plain = [o for o in added if o.kind not in ("prefer_rule", "prefer_premise")]
    prefs = [o for o in added if o.kind in ("prefer_rule", "prefer_premise")]
    return list(base_ops) + plain + prefs


def score_item(answer_text: str, item: Dict, strict_parse: bool = True) -> Dict:
    base_ops = item["base_ops"]
    ordering = item["ordering"]
    goals: List[Dict] = item["goals"]
    minimum = item.get("min_directives")

    diag: Dict = {"n_lines": 0, "n_unparseable": 0, "illegal": [], "goals_met": [],
                  "n_used": 0, "minimum": minimum}

    p = parse_answer(answer_text)
    diag.update(n_lines=p.n_lines, n_unparseable=p.n_unparseable,
                unparseable_examples=p.unparseable_examples)
    if p.no_region:
        return {"score": 0.0, "reason": "no_answer_region", "diagnostics": diag}
    if strict_parse and p.n_unparseable:
        return {"score": 0.0, "reason": f"unparseable_lines:{p.n_unparseable}", "diagnostics": diag}
    if not p.ops:
        return {"score": 0.0, "reason": "no_directives", "diagnostics": diag}

    kept, illegal = check_legality(p.ops, base_ops, item.get("allow_strict", False),
                                   item.get("preferences_only", False))
    diag["illegal"] = illegal
    if not kept:
        return {"score": 0.0, "reason": "all_directives_illegal", "diagnostics": diag}

    try:
        v = ASPICVerifier.from_operations(_apply(base_ops, kept), ordering=ordering)
        consistent = v.is_consistent()
    except Exception as e:
        return {"score": 0.0, "reason": f"engine_rejected:{type(e).__name__}", "diagnostics": diag}

    met = []
    for g in goals:
        got = str(v.status(g["claim"]))
        met.append(got == g["want"])
        diag["goals_met"].append({"claim": g["claim"], "want": g["want"], "got": got})
    success = all(met) and consistent
    n_used = len(kept)
    diag["n_used"] = n_used

    diag["achieved_status"] = {g["claim"]: g["got"] for g in diag["goals_met"]}
    diag["deadlock_not_defeat"] = sum(
        1 for g in diag["goals_met"]
        if g["got"] == "UNDECIDED" and g["want"] in ("OVERRULED", "JUSTIFIED"))

    if not success:
        # PARTIAL CREDIT FOR MEASURABLE PROGRESS.
        #
        # Measured: with an all-or-nothing reward, a GRPO group drawn from a policy that cannot yet
        # solve the item scores 0.000 on every sample -- standard deviation 0.000 across all four
        # construction tasks, so the group contributes NO GRADIENT. The F1-scored tasks are dense by
        # construction (status_query 0.336 to 1.000, claim_chain 0.400 to 1.000); the construction
        # tasks were not, which made five of the suite's modes unusable for training until the policy
        # could already solve them.
        #
        # The progress signal was already being computed and discarded. Two things count:
        #   * a goal MET while others are not -- visible in attack_defense, which has two goals
        #   * a goal that reached UNDECIDED when OVERRULED or JUSTIFIED was wanted -- the policy
        #     created the conflict and failed to win it, which is strictly closer than doing nothing
        #
        # SIZING MATTERS MORE THAN THE IDEA. Partial credit is capped at PARTIAL_CAP = 0.25 against a
        # success floor of 0.5, so a fully-deadlocked answer is worth half the WORST successful one.
        # Any higher and deadlock-farming becomes a local optimum for a policy with a low success rate,
        # since deadlocking is cheaper than winning.
        # SUB-GOAL PROGRESS, for goals whose own status cannot move incrementally.
        #
        # `attack` has one goal and N supporting chains. The target is justified if ANY chain survives,
        # so its status stays JUSTIFIED until the LAST cut lands -- measured, apexes killed 0,1,2,3,4,5
        # while the target reads JUSTIFIED throughout and flips only at 5. Goal-status progress is
        # therefore invisible and the reward was 0.000 for every near-miss.
        #
        # The item may declare `subgoals`: literals whose defeat constitutes progress. For `attack`
        # those are the chain apexes, which DO move one at a time. This is the only place the scorer
        # accepts task-specific structure, and it does so through a declared field rather than by
        # inspecting the theory, so no task is privileged.
        subgoals = item.get("subgoals") or []
        sub_progress = None
        if subgoals:
            hit = 0
            for lit in subgoals:
                try:
                    if str(v.status(lit)) != "JUSTIFIED":
                        hit += 1
                except Exception:
                    pass
            sub_progress = hit / len(subgoals)
            diag["subgoals_defeated"] = f"{hit}/{len(subgoals)}"

        # PER-GOAL credit, with sub-goal progress applied only to the goal it belongs to.
        #
        # An earlier version took max(goal_progress, sub_progress) globally. That gave brute force on
        # `attack_defense` the full cap: it kills every chain apex -- full sub-goal credit -- while
        # destroying the claim it is supposed to DEFEND. Sub-goal progress must not paper over damage
        # done elsewhere.
        #
        # Sub-goal credit is also weighted BELOW a deadlock (0.3 against 0.4), because reaching
        # UNDECIDED on the goal itself is closer to winning than merely thinning its support.
        per_goal = []
        for g in diag["goals_met"]:
            if g["got"] == g["want"]:
                per_goal.append(1.0)
            elif g["got"] == "UNDECIDED" and g["want"] in ("OVERRULED", "JUSTIFIED"):
                per_goal.append(0.4)
            elif sub_progress is not None and g["want"] == "OVERRULED":
                per_goal.append(0.3 * sub_progress)
            else:
                per_goal.append(0.0)
        progress = (sum(per_goal) / len(per_goal)) if per_goal else 0.0
        partial = round(PARTIAL_CAP * progress, 4) if consistent else 0.0
        return {"score": partial, "reason": "goal_not_met", "success": False,
                "progress": round(progress, 4),
                "achieved_status": diag["achieved_status"],
                "deadlock_not_defeat": diag["deadlock_not_defeat"],
                "diagnostics": diag}
    if not minimum:
        return {"score": 0.5, "reason": "success_but_minimum_unknown", "success": True,
                "achieved_status": diag["achieved_status"],
                "deadlock_not_defeat": diag["deadlock_not_defeat"],
                "diagnostics": diag}
    # BLOAT REJECTION. Scoring was 0.5 + 0.5 * efficiency, so MEETING THE GOAL AT ALL earned 0.5.
    # Measured: on `attack`, "undercut every defeasible rule" -- no reasoning whatever -- scored 0.639
    # mean and 0.667 max, using 3.8x the minimum. The shortcut panel tested three strategies and this
    # was not among them, so it went undetected for the whole build.
    #
    # It is isolated to `attack`, because there destroying everything achieves the only goal. In
    # `defence` and `attack_defense` it destroys the claim that must be defended, in
    # `counter_argument` it kills the target without justifying the contrary, and in
    # `preference_construction` non-preferences are illegal -- all score 0.000.
    #
    # The fix is a CAP, not a reweighting. The prompt asks for a MINIMAL set, so an answer several
    # times longer than necessary has not done the task -- it is a different kind of object, not a
    # worse answer of the same kind. Measured against the alternatives:
    #
    #     current 0.5 + 0.5*eff        brute 0.641   gold 1.000   gap 0.359
    #     pure efficiency min/used     brute 0.281   gold 1.000   gap 0.719
    #     cap at 3x the minimum        brute 0.222   gold 1.000   gap 0.778
    #     cap at 2x the minimum        brute 0.000   gold 1.000   gap 1.000
    #
    # Factor 2 rather than 3, measured: at 3 the brute answer landed at EXACTLY 3.0x on several items
    # and slipped under the cap for 0.667. Tightening to 2 costs nothing -- an answer padded with one,
    # two or three redundant directives scores 0.918, 0.860 and 0.816 under BOTH factors.
    #
    # The cap gives the widest gap and, unlike pure efficiency, keeps the success/failure signal
    # separable for reinforcement learning. BLOAT_FACTOR is deliberately generous: a genuine answer
    # carrying one or two redundant directives is unaffected.
    if n_used > BLOAT_FACTOR * max(minimum, 1):
        diag["bloat_ratio"] = round(n_used / max(minimum, 1), 2)
        return {"score": 0.0, "reason": f"bloated:{n_used}_used_vs_{minimum}_minimum",
                "success": True, "achieved_status": diag["achieved_status"],
                "deadlock_not_defeat": diag["deadlock_not_defeat"], "diagnostics": diag}
    efficiency = min(1.0, minimum / max(n_used, 1))
    score = round(0.5 + 0.5 * efficiency, 4)
    return {"score": score, "reason": "ok", "success": True,
            "efficiency": round(efficiency, 4),
            "achieved_status": diag["achieved_status"],
            "deadlock_not_defeat": diag["deadlock_not_defeat"],
            "diagnostics": diag}
