from __future__ import annotations

import re


PARSER_VERSION = "multilogieval-explicit-answer-v1"


_EXPLICIT_ANSWER_RE = re.compile(
    r"""(?imx)
    ^[ \t]*
    Answer[ \t]*:[ \t]*
    (?:\r?\n[ \t]*)?
    (yes|no)\b
    """
)

_BARE_ANSWER_RE = re.compile(
    r"^\s*(yes|no)\s*[.!]?\s*$",
    flags=re.I,
)


def parse_multilogieval_answer(
    completion: str,
) -> str | None:
    """Extract the model's final explicit yes/no answer.

    Formatting-only normalization is applied.

    If multiple explicit Answer: fields occur, the final explicit
    answer is used because it is the model's last declared answer.

    No inference is made from reasoning prose.
    """

    text = completion.strip().replace("**", "")

    explicit = [
        match.group(1).lower()
        for match in _EXPLICIT_ANSWER_RE.finditer(text)
    ]

    if explicit:
        return explicit[-1]

    match = _BARE_ANSWER_RE.fullmatch(text)

    if match:
        return match.group(1).lower()

    return None
