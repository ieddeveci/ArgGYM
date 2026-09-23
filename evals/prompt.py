"""Composing the prompt, which is the harness's job and not the dataset's.

A frozen question says nothing about where to put the answer
(`arggym/core/prompting.py:answer_format`, which renders no fence unless a
caller asks for one). So the submission convention is named here, exactly once,
and both the instruction the model reads and the extraction that reads it back
come from the same `AnswerTemplate`. They cannot disagree.

The previous harness named its convention in three files and left a comment in
each telling the reader to keep them in lockstep by hand.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import arggym

#: Named templates a config may select, one file each under `conf/template/`.
#: Adding a convention is adding a line: `AnswerTemplate(name, open, close)`
#: carries its own instruction sentence and its own delimiters, so the extractor
#: follows automatically.
#:
#: No `\boxed{...}` here. Its closing delimiter is a brace, and every answer
#: shape this benchmark uses contains braces or spans lines, so the region would
#: end at the first `}` rather than the matching one. A convention that can
#: truncate an answer is worse than no convention, because the score it produces
#: looks like a reasoning failure.
TEMPLATES = {
    "xml_tags": arggym.XML_TAGS,
    "square_tags": arggym.AnswerTemplate("square_tags", "[answer]", "[/answer]"),
}


@dataclass(frozen=True)
class Elicitation:
    """How the run asks for reasoning, kept apart from what it asks for.

    The question is the task. Everything here is the evaluator's choice about
    how to put it, and the whole of it -- not just `name` -- is recorded in the
    run manifest, so two runs that differ only in this are comparable as an
    experiment rather than confusable as a result. `run.py` compares it before
    resuming: a name is what a config file happens to call a system prompt, and
    `elicitation.system=...` changes the prompt while leaving the name alone.
    """

    name: str = "none"
    system: str = ""
    prefix: str = ""
    suffix: str = ""


def compose(row: Dict, template_name: Optional[str],
            elicit: Elicitation) -> Tuple[Optional[str], str]:
    """The system message and the user message, from a row.

    `template_name` of None submits with no fence at all, which is the right
    setting for a solver that returns a value rather than text.
    """
    user = row["question"]
    if template_name is not None:
        user = f"{user}\n{template(template_name).instruction}"
    if elicit.prefix:
        user = f"{elicit.prefix}\n\n{user}"
    if elicit.suffix:
        user = f"{user}\n\n{elicit.suffix}"
    return (elicit.system or None), user


def template(name: str) -> arggym.AnswerTemplate:
    try:
        return TEMPLATES[name]
    except KeyError:
        raise KeyError(f"unknown template {name!r}; known: {sorted(TEMPLATES)}") from None


def region(text: str, template_name: Optional[str]) -> Tuple[str, bool]:
    """The answer region and whether there was one.

    Two facts, because they lead to different findings. A completion with no
    region is a formatting failure or a truncation; a completion with a region
    that scores zero is a reasoning failure, and the second is the only one that
    is a result about the model's argumentation.

    The region is the body of the last complete pair of delimiters
    (`arggym.AnswerTemplate.region`), which is also what `arggym.extract_answer`
    reads for `<answer>`. Every template goes through that one method, so the
    harness and the package cannot disagree on where an answer starts.

    With no template the whole completion is the answer.
    """
    if template_name is None:
        return text, True
    body = template(template_name).region(text)
    return (body, True) if body is not None else (text or "", False)
