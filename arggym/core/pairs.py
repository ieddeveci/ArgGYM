from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, NamedTuple, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)

#: What a `claim: status` pattern may have in front of it, and what it may have
#: behind it, so the pattern can be anchored to a whole line.
#:
#: Anchoring is the point. Written unanchored, the label-map patterns read a
#: status out of any sentence that carried one -- "My conclusion: overruled"
#: parsed as the claim `conclusion` and quietly added a wrong claim to the
#: prediction, lowering precision on an answer the model never gave. The prompts
#: already say "one line per claim", so the line is the unit.
#:
#: A bullet or list number is allowed in front and has to be followed by a space:
#: that is what keeps `- ao0: overruled` readable while leaving `-ao0: overruled`
#: with the minus that makes its claim a negated literal.
LINE_START = r"^[ \t]*(?:[-*\u2022][ \t]+|\d+[.)][ \t]+)?"
LINE_END = r"[ \t]*[.,;]?[ \t]*$"


def collect(matches: Iterable[Tuple[K, str]]) -> Dict[K, List[str]]:
    pred: Dict[K, List[str]] = {}
    for key, status in matches:
        stats = pred.setdefault(key, [])
        if status not in stats:
            stats.append(status)
    return pred


class PairF1(NamedTuple):
    tp: int
    precision: float
    recall: float
    f1: float
    exact_match: bool
    contradicted: list


def pair_f1(pred: Dict[K, List[str]], gold: Dict[K, str]) -> PairF1:
    tp = sum(1 for k, s in pred.items() if s == [gold.get(k)])
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(gold), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    contradicted = sorted(k for k, s in pred.items() if len(s) > 1)
    return PairF1(tp, precision, recall, f1, tp == len(gold) == len(pred), contradicted)
