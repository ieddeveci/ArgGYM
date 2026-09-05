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
        f.flush()
        # Flushed to the device before the rename, not just out of Python's
        # buffer. Without this the rename survives a killed process but not a
        # machine reboot, which is the failure this module exists for.
        os.fsync(f.fileno())
    # A manifest half-written by an interrupted process is worse than no
    # manifest: it reads as valid JSON right up to the point it does not.
    os.replace(tmp, path)


def write_jsonl(path: str, rows: Any) -> None:
    """A whole JSONL, written atomically.

    Not `Appender`: this replaces a file rather than growing one, and a scoring
    pass killed halfway would otherwise leave a truncated `samples.jsonl` beside
    a `metrics.json` from the previous scorer version, with nothing comparing
    the two.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
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


def best_per_id(run_dir: str, name: str = GENERATIONS) -> Dict[str, Dict[str, Any]]:
    """One record per id: the last that succeeded, or the last that failed.

    An id can hold several records -- a failure and the retry that replaced it,
    in either order, because a resumed run appends. Preferring the last
    *successful* one rather than simply the last means a generation that was
    paid for is never thrown away by a later transient failure, and it is what
    makes `completed_ids` and this function agree: both are answering "is there
    a good generation for this id", so resume cannot skip an id whose good
    record scoring would then ignore.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for r in read_jsonl(os.path.join(run_dir, name)):
        current = best.get(r["id"])
        if current is None or not r.get("error") or current.get("error"):
            best[r["id"]] = r
    return best


def completed_ids(run_dir: str) -> Set[str]:
    """Ids with a good generation on disk, which a rerun may skip."""
    return {i for i, r in best_per_id(run_dir).items() if not r.get("error")}


def restart(run_dir: str) -> None:
    """Throw away a directory's generations, for a run that asked not to resume.

    `Appender` opens for append, so leaving the file in place would mix the old
    records in with the new ones and `best_per_id` would keep whichever
    succeeded -- which is resume, under a flag that says otherwise.
    """
    for name in (GENERATIONS, PROMPTS, SAMPLES, METRICS):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            os.remove(path)
