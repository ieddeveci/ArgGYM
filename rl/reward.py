"""TRL reward adapter delegating correctness to ArgGYM's existing scorer."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import arggym
from evals.prompt import region


def completion_text(value: Any) -> str:
    """Normalize TRL standard/conversational completion representations."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content") or "")
    if isinstance(value, list):
        if not value:
            return ""
        if isinstance(value[-1], dict):
            return str(value[-1].get("content") or "")
        return "".join(str(x) for x in value)
    return str(value or "")


def arggym_reward(
    completions: Sequence[Any],
    row_json: Sequence[str],
    template_name: Sequence[Optional[str]],
    task: Optional[Sequence[str]] = None,
    difficulty_band: Optional[Sequence[Optional[str]]] = None,
    curriculum_stage: Optional[Sequence[Optional[str]]] = None,
    log_extra=None,
    log_metric=None,
    **_: Any,
) -> List[float]:
    """Return ArgGYM's native [0,1] task score for the final submitted answer.

    Only the last ``<answer>...</answer>`` submission region is rewarded. A
    missing answer region receives zero; the reasoning trace itself is never
    graded. Scorer/engine exceptions are deliberately *not* caught: an
    infrastructure failure must stop training rather than become a fake zero.
    """
    if not (len(completions) == len(row_json) == len(template_name)):
        raise ValueError(
            "TRL reward columns are misaligned: "
            f"{len(completions)=}, {len(row_json)=}, {len(template_name)=}"
        )

    rewards: List[float] = []
    successes: List[bool] = []
    reasons: List[str] = []
    answer_regions: List[bool] = []
    for completion, raw_row, template in zip(completions, row_json, template_name):
        row: Dict[str, Any] = json.loads(raw_row)
        body, found = region(completion_text(completion), template)
        answer_regions.append(bool(found))
        if not found:
            reward = 0.0
            success = False
            reason = "missing_answer_region"
        else:
            result = arggym.score_row(body, row)
            reward = float(result.score)
            success = bool(result.success)
            reason = str(result.reason)
            if not math.isfinite(reward) or not (0.0 <= reward <= 1.0):
                raise RuntimeError(
                    f"ArgGYM returned invalid RL reward {reward!r} for row {row.get('id')}"
                )
        rewards.append(reward)
        successes.append(success)
        reasons.append(reason)

    if log_extra is not None:
        log_extra("arggym_success", successes)
        log_extra("arggym_reason", reasons)
        log_extra("arggym_answer_region", answer_regions)
    if log_metric is not None and rewards:
        log_metric("arggym_success_rate", sum(successes) / len(successes))
        log_metric("arggym_answer_region_rate", sum(answer_regions) / len(answer_regions))
        # These slices are diagnostic only; they never alter reward weighting.
        for label, values in (("task", task), ("band", difficulty_band), ("stage", curriculum_stage)):
            if values is None:
                continue
            grouped: Dict[str, List[float]] = defaultdict(list)
            for key, reward in zip(values, rewards):
                grouped[str(key)].append(reward)
            for key, vals in grouped.items():
                safe = key.replace("/", "_").replace(" ", "_")
                log_metric(f"arggym_reward_{label}_{safe}", sum(vals) / len(vals))
    return rewards
