"""The reward: ArgGYM's strict success, scored exactly as `evals/score.py` scores a run.

A completion is split into reasoning and answer where vLLM's `qwen3` reasoning
parser splits it, at `</think>`, and handed to `evals.score.score_one` -- the
function that scores every eval run -- so the answer region, the fallback into
the reasoning trace and the scorer are the harness's own. The reward is
`success` as 1.0 or 0.0. The partial-credit `score` is logged and never paid:
a policy that reads nothing already earns a large share of it.

Two cases return `None`, which TRL treats as unscorable: the completion leaves
the group baseline and gets advantage 0.

- Truncated, meaning the last token is neither EOS nor padding. Scoring it 0
  would hand every finished sibling a higher advantage whatever it answered,
  and push the policy shorter for the wrong reason. `mask_truncated_completions`
  takes it out of the loss; `None` takes it out of the baseline too.
- The scorer refused the row (`score_one` reports `scorer_refused`). That is a
  fault in the row or the install, and eval keeps it out of the score as well.

Everything the reward computes is also logged, once per batch, over the whole
batch: every rank gathers the batch before computing a metric, so a group split
across ranks is still one group.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence

from evals.score import score_one

THINK_END = "</think>"


def split_reasoning(text: str) -> Dict[str, str]:
    """Reasoning and completion, as the eval server's reasoning parser returns them."""
    head, sep, tail = text.partition(THINK_END)
    if not sep:
        return {"reasoning": "", "completion": text}
    return {"reasoning": head.replace("<think>", "", 1).strip(), "completion": tail.strip()}


def text_of(completion: Any) -> str:
    """TRL passes a conversational completion as `[{"role": "assistant", "content": ...}]`."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return completion[-1].get("content") or ""
    raise TypeError(f"unexpected completion shape: {type(completion).__name__}")


def score(text: str, row: Dict[str, Any], truncated: bool, template: str) -> Dict[str, Any]:
    """The eval harness's record for this completion."""
    gen = {"id": row["id"], "task": row["task"], "level": row["metadata"]["level"],
           "ordering": row["metadata"]["ordering"], "truncated": truncated,
           **split_reasoning(text)}
    return score_one(gen, row, template)


def reward_of(record: Dict[str, Any]) -> Optional[float]:
    if record["truncated"] or record["success"] is None:
        return None
    return 1.0 if record["success"] else 0.0


def make_reward(stop_ids: Sequence[int], template: str = "xml_tags",
                gather: Optional[Callable[[List[Any]], List[Any]]] = None) -> Callable:
    """A TRL reward function. `stop_ids` are the token ids that end a completion (EOS, pad).

    `gather` collects a list from every rank; the default does so when
    `torch.distributed` is running and is the identity otherwise.
    """
    stop = set(stop_ids)
    gather = gather or _gather

    def arggym_success(completions, completion_ids, row_json, log_metric=None,
                       log_extra=None, **_: Any) -> List[Optional[float]]:
        records = []
        for completion, ids, raw in zip(completions, completion_ids, row_json, strict=True):
            row = json.loads(raw)
            truncated = bool(ids) and ids[-1] not in stop
            rec = score(text_of(completion), row, truncated, template)
            rec["length"] = len(ids)
            records.append(rec)
        rewards = [reward_of(r) for r in records]
        if log_extra is not None:
            log_extra("task", [r["task"] for r in records])
            log_extra("level", [r["level"] for r in records])
            log_extra("success", [r["success"] for r in records])
            log_extra("partial_score", [r["score"] for r in records])
            log_extra("reason", [str(r["reason"])[:200] for r in records])
            log_extra("truncated", [r["truncated"] for r in records])
        if log_metric is not None:
            slim = [{k: r[k] for k in ("id", "task", "success", "score", "truncated",
                                       "length", "no_answer_region")} for r in records]
            for name, value in batch_metrics(gather(slim)).items():
                log_metric(name, value)
        return rewards

    return arggym_success


def batch_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Metrics over one batch. A truncated completion counts as a failure, as in eval.

    Groups are keyed by row id: a training batch holds each prompt once, with
    its `num_generations` completions.
    """
    n = len(records)
    if not n:
        return {}
    ok = [r for r in records if r["success"] is not None]
    out: Dict[str, float] = {
        "arggym_success_rate": sum(bool(r["success"]) for r in ok) / max(len(ok), 1),
        "arggym_partial_score": sum(r["score"] for r in ok) / max(len(ok), 1),
        "arggym_truncated_rate": sum(r["truncated"] for r in records) / n,
        "arggym_no_answer_region_rate": sum(bool(r["no_answer_region"]) for r in ok)
        / max(len(ok), 1),
        "arggym_scorer_refused": float(n - len(ok)),
    }
    finished = [r for r in ok if not r["truncated"]]
    for label, subset in (("correct", [r for r in finished if r["success"]]),
                          ("wrong", [r for r in finished if not r["success"]])):
        if subset:
            out[f"arggym_length/{label}"] = sum(r["length"] for r in subset) / len(subset)
    by_task: Dict[str, List[bool]] = defaultdict(list)
    for r in ok:
        by_task[r["task"]].append(bool(r["success"]))
    for task, s in sorted(by_task.items()):
        out[f"arggym_success_rate/{task}"] = sum(s) / len(s)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[r["id"]].append(r)
    # The completions the reward scores, i.e. not truncated, decide the group's
    # variance, so an all-wrong group is one whose scored members all failed.
    scored = [[r for r in g if not r["truncated"] and r["success"] is not None]
              for g in groups.values()]
    scored = [g for g in scored if g]
    if scored:
        out["arggym_groups/all_correct"] = sum(all(r["success"] for r in g)
                                               for g in scored) / len(groups)
        out["arggym_groups/all_wrong"] = sum(not any(r["success"] for r in g)
                                             for g in scored) / len(groups)
    out["arggym_groups/no_scored_member"] = (len(groups) - len(scored)) / len(groups)
    return out


def _gather(items: List[Any]) -> List[Any]:
    try:
        import torch.distributed as dist
    except ImportError:
        return items
    if not (dist.is_available() and dist.is_initialized()):
        return items
    from accelerate.utils import gather_object

    return gather_object(items)
