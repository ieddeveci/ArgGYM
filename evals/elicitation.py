"""How an answer is elicited -- the evaluator's choice, not the benchmark's.

A taskset item carries the task and the submission contract ("give your final
answer between [answer] and [/answer]"). Everything about *how* the model is made
to produce that answer -- a system prompt, step-by-step instructions, few-shot
examples, a scaffold -- is method, and method is what is being evaluated. Keeping
it here means two runs of the same taskset under different elicitation are
distinguishable in `run.json` rather than indistinguishable.

Config lives in evals/conf/elicitation/*.yaml; `none` is the default.
"""
from __future__ import annotations

from typing import Optional, Tuple


def apply(prompt: str, elicitation: Optional[dict]) -> Tuple[str, Optional[str]]:
    """Return (prompt, system) after applying an elicitation config.

    `prefix` and `suffix` wrap the item prompt; `system` becomes a system
    message. All three are optional, so `none` is a genuine no-op -- the model
    sees the taskset bytes unchanged.
    """
    if not elicitation:
        return prompt, None

    prefix = (elicitation.get("prefix") or "").strip("\n")
    suffix = (elicitation.get("suffix") or "").strip("\n")
    system = elicitation.get("system") or None

    parts = [p for p in (prefix, prompt, suffix) if p]
    return "\n\n".join(parts), system


def describe(elicitation: Optional[dict]) -> dict:
    """The record written to run.json, so a result can be traced to its method."""
    if not elicitation:
        return {"name": "none", "system": False, "prefix": False, "suffix": False}
    return {
        "name": elicitation.get("name", "custom"),
        "system": bool(elicitation.get("system")),
        "prefix": bool(elicitation.get("prefix")),
        "suffix": bool(elicitation.get("suffix")),
    }
