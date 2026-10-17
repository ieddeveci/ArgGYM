"""GSM8K as a positive control: the same trainer on a task GRPO is known to move.

`task: gsm8k` in a config swaps the data, the prompt, the reward and the
metrics for these; the trainer, vLLM, the weight sync, the loss and dev
evaluation are the ArgGYM run's own. Training rows are HF `openai/gsm8k`
`train`; dev is a seeded subset of `test`, with any question `train` also holds
dropped.

The reward is 1 when the last `\\boxed{...}` in the completion is the gold
number after `####`, compared as numbers once commas and dollar signs are
stripped, and 0 otherwise. A truncated completion returns `None`, as the
ArgGYM reward does (`rl.reward`).
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence

from rl.reward import _gather, text_of

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)")


def last_boxed(text: str) -> Optional[str]:
    """The content of the last `\\boxed{...}`, nested braces included; None if there is none."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None


def to_number(text: Optional[str]) -> Optional[float]:
    """`1,234`, `$18`, `\\$-3.50` as a float; None for anything that is not one number."""
    if text is None:
        return None
    s = re.sub(r"\\\$|\$|,|\s", "", text)
    return float(s) if _NUMBER.fullmatch(s) else None


def gold(answer: str) -> float:
    """The number after `####` in a GSM8K answer."""
    value = to_number(answer.split("####")[-1])
    if value is None:
        raise ValueError(f"no number after ####: {answer[-80:]!r}")
    return value


def correct(text: str, gold_value: float) -> bool:
    value = to_number(last_boxed(text))
    return value is not None and abs(value - gold_value) < 1e-6


def load_splits(data: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Training rows from `train`, `dev_size` rows of `test` drawn with `dev_seed`."""
    from datasets import load_dataset

    ds = load_dataset(data["dataset"], data["config"], revision=data["revision"])
    train = [{"id": f"train-{i}", **r} for i, r in enumerate(ds["train"])]
    seen = {r["question"].strip() for r in train}
    test = [{"id": f"test-{i}", **r} for i, r in enumerate(ds["test"])
            if r["question"].strip() not in seen]
    dev = random.Random(data["dev_seed"]).sample(test, data["dev_size"])
    return train, dev


def trl_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"prompt": [{"role": "user", "content": f"{r['question']}\n{INSTRUCTION}"}],
             "gold": gold(r["answer"]), "id": r["id"]} for r in rows]


def make_reward(stop_ids: Sequence[int],
                gather: Optional[Callable[[List[Any]], List[Any]]] = None) -> Callable:
    """A TRL reward function; `stop_ids` and `gather` as in `rl.reward.make_reward`."""
    stop = set(stop_ids)
    gather = gather or _gather

    def gsm8k_correct(completions, completion_ids, gold, log_metric=None, log_extra=None,
                      **columns: Any) -> List[Optional[float]]:
        records = []
        for completion, ids, g, i in zip(completions, completion_ids, gold, columns["id"],
                                         strict=True):
            text = text_of(completion)
            records.append({"id": i, "success": correct(text, g),
                            "boxed": last_boxed(text) is not None,
                            "truncated": bool(ids) and ids[-1] not in stop,
                            "length": len(ids)})
        if log_extra is not None:
            log_extra("success", [r["success"] for r in records])
            log_extra("truncated", [r["truncated"] for r in records])
        if log_metric is not None:
            for name, value in batch_metrics(gather(records)).items():
                log_metric(name, value)
        return [None if r["truncated"] else float(r["success"]) for r in records]

    return gsm8k_correct


def batch_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Metrics over one batch. A truncated completion counts as a failure."""
    n = len(records)
    if not n:
        return {}
    finished = [r for r in records if not r["truncated"]]
    out = {"gsm8k_success_rate": sum(r["success"] and not r["truncated"] for r in records) / n,
           "gsm8k_truncated_rate": (n - len(finished)) / n,
           "gsm8k_no_box_rate": sum(not r["boxed"] for r in finished) / max(len(finished), 1)}
    for label, subset in (("correct", [r for r in finished if r["success"]]),
                          ("wrong", [r for r in finished if not r["success"]])):
        if subset:
            out[f"gsm8k_length/{label}"] = sum(r["length"] for r in subset) / len(subset)
    groups: Dict[str, List[bool]] = defaultdict(list)
    for r in finished:
        groups[r["id"]].append(r["success"])
    n_groups = len({r["id"] for r in records})
    # As `arggym_groups/no_signal`: groups whose scored members all agree, or
    # that have fewer than two, carry no gradient.
    alive = sum(len(g) > 1 and len(set(g)) > 1 for g in groups.values())
    out["gsm8k_groups/no_signal"] = 1 - alive / n_groups
    return out
