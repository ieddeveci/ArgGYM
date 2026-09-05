"""Carrying a non-text answer through a JSON file.

A solver may produce the answer as a value rather than as text -- constrained
decoding, a JSON schema, a tool call -- and `arggym.score_row_value` takes it
directly (`docs/dataset-contract.md` section 4). But generating and scoring are
separate programs here, so the value spends the time in between as a line of
JSON, and two of the four answer shapes do not survive that trip unhelped:

  operation_list   a list of `Operation` objects, which is why `ops_to_json`
                   and `ops_from_json` are public
  label_map        `semantics_query` keys its map by a `(claim, semantics)`
                   pair, and JSON has no tuple key

Choosing an encoding is the harness's job. The dataset refuses to dictate a
serialization, which is the whole reason `score_row_value` exists.
"""
from __future__ import annotations

from typing import Any, Dict

import arggym


def encode(value: Any, row: Dict[str, Any]) -> Any:
    """A value as JSON, for a solver writing one into `Attempt.value`."""
    shape = row["metadata"].get("answer_shape")
    if shape == "operation_list":
        return arggym.ops_to_json(value)
    if shape == "label_map" and any(isinstance(k, tuple) for k in value):
        return [[list(k), v] for k, v in value.items()]
    return value


def check_json(value: Any, row_id: str) -> None:
    """Refuse a value that cannot survive the trip to `score.py`.

    Generations are written with `default=str`, which would quietly turn a list
    of live `Operation` objects into a list of strings. `decode` then hands
    those strings to `score_row_value`, which at best refuses the row and at
    worst grades it zero -- and by then the inference has been paid for.

    `run.py` catches this per row and records the item as an error, keeping the
    completion. A solver that does it on one row does it on all of them, so the
    run ends on `max_error_rate` rather than on an exception out of the thread
    pool that discarded the generation it was complaining about.
    """
    import json

    try:
        json.dumps(value)
    except TypeError as e:
        raise TypeError(
            f"{row_id}: the solver's value is not JSON ({e}). It travels to "
            f"score.py as a line in a file, so pass it through "
            f"evals.values.encode(value, row) first -- an operation list and "
            f"semantics_query's tuple-keyed map both need it.") from None


def decode(payload: Any, row: Dict[str, Any]) -> Any:
    """JSON back to the value `score_row_value` expects.

    `ordered_sequence` and `record_list` are already JSON, and a `label_map`
    keyed by strings is too, so those pass through untouched.
    """
    shape = row["metadata"].get("answer_shape")
    if shape == "operation_list":
        return arggym.ops_from_json(payload)
    if shape == "label_map" and isinstance(payload, list):
        return {tuple(k): v for k, v in payload}
    return payload
