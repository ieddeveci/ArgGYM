"""The solver the harness ships: compose a prompt, call a chat endpoint, return.

One implementation of the `Solver` protocol, and the only one that knows an
inference provider exists. A different solver -- an agent with tools, a
constrained decoder, a symbolic procedure with no model in it -- replaces this
file and nothing else, because `run.py` holds it only as a `Solver`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from evals.client import ChatClient, Endpoint
from evals.prompt import Elicitation, compose
from evals.types import Attempt


class ChatSolver:
    """Question in, completion out, over an OpenAI-compatible endpoint."""

    def __init__(self, endpoint: Endpoint, template: Optional[str] = "xml_tags",
                 elicitation: Optional[Elicitation] = None) -> None:
        self.client = ChatClient(endpoint)
        self.template = template
        self.elicitation = elicitation or Elicitation()

    def __call__(self, row: Dict[str, Any]) -> Attempt:
        system, user = compose(row, self.template, self.elicitation)
        return self.client.complete(system, user)

    def prompt_of(self, row: Dict[str, Any]):
        """The exact bytes this solver would send, without sending them.

        `run.py` writes these to disk before any call, so a run that dies
        mid-flight still says what it asked.
        """
        return compose(row, self.template, self.elicitation)
