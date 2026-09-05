"""The run directory: what is written, when, and what a rerun may skip.

Three rules, each one paid for by the previous harness.

Write the prompt before the call, so a run that dies mid-flight still says what
it asked. Append every result the moment it lands, so a reboot costs the item in
flight and not the run -- one did, and 439 generations were nearly lost.
And keep the raw completion forever, because scoring reads it again every time
the scorer changes.

    runs/<stamp>__<model>/
        run.json          manifest; status running -> completed | failed
        prompts.jsonl     written before inference
        generations.jsonl appended as results land
        samples.jsonl     written by score.py
        metrics.json      written by score.py
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Iterator, Set

RUN = "run.json"
PROMPTS = "prompts.jsonl"
GENERATIONS = "generations.jsonl"
SAMPLES = "samples.jsonl"
METRICS = "metrics.json"


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    # A manifest half-written by an interrupted process is worse than no
    # manifest: it reads as valid JSON right up to the point it does not.
    os.replace(tmp, path)


class Appender:
    """Line-buffered append, safe across the worker threads."""

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a")
        self._lock = threading.Lock()

    def write(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, default=str)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "Appender":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def completed_ids(run_dir: str) -> Set[str]:
    """Ids already generated without an error, which a rerun may skip.

    An errored record is not complete: it is retried, and the retry appends a
    second line for that id. `score.py` keeps the last line per id, so the retry
    is the one that counts.
    """
    return {r["id"] for r in read_jsonl(os.path.join(run_dir, GENERATIONS))
            if not r.get("error")}


def last_per_id(run_dir: str, name: str = GENERATIONS) -> Dict[str, Dict[str, Any]]:
    """One record per id, the last written winning."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in read_jsonl(os.path.join(run_dir, name)):
        out[r["id"]] = r
    return out
