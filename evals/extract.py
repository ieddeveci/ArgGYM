"""Decide which text gets scored, and why.

Keeping chain-of-thought out of the scored text is the whole job of this module.
A reasoning model may draft a candidate answer mid-chain-of-thought and then
revise it, so the draft must never be what gets scored: the post-thinking
`content` field wins whenever it carries an answer region, and where the two
fields have to be concatenated, aspic_gym._answer_region takes the last complete
region rather than the first.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aspic_gym import _answer_region  # noqa: E402


def has_region(text: Optional[str]) -> bool:
    return bool(text) and _answer_region(text) is not None


def extract(content: Optional[str], reasoning_content: Optional[str]) -> dict:
    """Return the text to score plus audit flags.

    Policy:
      1. Score `content` when it carries an answer region. With a reasoning
         parser enabled, chain-of-thought lands in `reasoning_content` and only
         the post-thinking answer is in `content`.
      2. Fall back to reasoning + content only when `content` has no region --
         this covers models served without a reasoning parser, where everything
         arrives in one field.
      3. `answer_in_cot` is recorded for auditing and never affects a score.
    """
    content = content or ""
    reasoning_content = reasoning_content or ""

    in_content = has_region(content)
    in_cot = has_region(reasoning_content)

    if in_content:
        text, used_fallback = content, False
    elif in_cot or reasoning_content:
        text, used_fallback = (reasoning_content + "\n" + content), True
    else:
        text, used_fallback = content, False

    return {
        "text": text,
        "used_fallback": used_fallback,
        "answer_in_cot": in_cot,
        "has_region": has_region(text),
    }
