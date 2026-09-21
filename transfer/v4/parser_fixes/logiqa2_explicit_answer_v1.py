from __future__ import annotations

import re


PARSER_VERSION = "logiqa2-explicit-answer-v1"


# After Markdown bold normalization, accept only an A/B/C/D label
# immediately following an explicit Answer: field.
#
# Accepted examples:
#
#   Answer: A
#   Answer: \boxed{A}
#   **Answer:** A
#   **Answer:** **A**
#   **Answer:**
#   A
#   Answer: C. Explanation follows...
#
# This parser never infers an answer from reasoning prose.
_EXPLICIT_ANSWER_RE = re.compile(
    r"""(?imx)
    ^[ \t]*
    Answer[ \t]*:[ \t]*
    (?:\r?\n[ \t]*)?
    (?:\\boxed\{[ \t]*)?
    ([ABCD])
    (?:[ \t]*\})?
    """,
)

# Preserve the original adapter's direct-leading-label fallback.
_LEADING_LABEL_RE = re.compile(
    r"^[ \t]*([ABCD])(?:\b|[\.\):,-])",
    flags=re.I,
)


def parse_logiqa2_answer(completion: str) -> str | None:
    """Extract the model's explicit LogiQA2 answer label.

    The transformation is formatting-only. Markdown bold markers are
    removed before extraction; no semantic inference is performed.
    """

    text = completion.strip().replace("**", "")

    explicit = [
        match.group(1).upper()
        for match in _EXPLICIT_ANSWER_RE.finditer(text)
    ]

    if explicit:
        # Multiple identical explicit answers are harmless.
        # Conflicting explicit answers are deliberately rejected.
        if len(set(explicit)) == 1:
            return explicit[-1]

        return None

    match = _LEADING_LABEL_RE.match(text)

    if match:
        return match.group(1).upper()

    return None
