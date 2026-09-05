"""The run directory: what is written, when, and what a rerun may skip.

Three rules, each one paid for by the previous harness.

Write the prompt before the call, so a run that dies mid-flight still says what
it asked. Append every result the moment it lands, so a reboot costs the item in
flight and not the run -- one did, and 439 generations were nearly lost.
And keep the raw completion forever, because scoring reads it again every time
the scorer changes.

    runs/<model>__<template>__<elicitation>/
        run.json          manifest; status running -> completed | failed | crashed
        run.lock          held for the length of a run
        prompts.jsonl     written before inference
        generations.jsonl appended as results land
        samples.jsonl     written by score.py
        metrics.json      written by score.py

The directory is named by the configuration rather than by a timestamp, so
running the same command twice resumes rather than starting a second copy. That
makes a double launch the natural accident instead of an impossible one, hence
`exclusive`.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
from typing import Any, Dict, Iterator, Set

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no flock
    fcntl = None  # type: ignore[assignment]

RUN = "run.json"
LOCK = "run.lock"
PROMPTS = "prompts.jsonl"
GENERATIONS = "generations.jsonl"
SAMPLES = "samples.jsonl"
METRICS = "metrics.json"


class Torn(SystemExit):
    """A JSONL file is damaged somewhere other than its last line."""


class Busy(SystemExit):
    """A run directory is already held by another process."""


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Every record, tolerating a half-written final line and nothing else.

    A machine that dies mid-append leaves the last line truncated, and that is
    the failure this module exists for: raising there would let one torn line
    cost `best_per_id`, resume, scoring and the manifest's error rate -- the
    whole run, to save the item in flight. A damaged line anywhere *earlier*
    is not a torn write, so it is refused rather than skipped.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = [ln for ln in f if ln.strip()]
    for i, line in enumerate(lines):
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            if i != len(lines) - 1:
                raise Torn(
                    f"{path} line {i + 1} of {len(lines)} is not JSON ({e}). "
                    f"Only the last line of an appended file can be a torn "
                    f"write; damage before it means the file was rewritten by "
                    f"something other than this harness, and skipping the line "
                    f"would silently drop a generation that was paid for."
                ) from None
            print(f"{path}: dropping a truncated last line ({e}); the item in "
                  f"flight when the writer stopped will be generated again.",
                  file=sys.stderr)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Per process: `run.json.tmp` is one fixed name, so two writers would
    # interleave into one file and `os.replace` would publish the mixture.
    tmp = f"{path}.{os.getpid()}.tmp"
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
    tmp = f"{path}.{os.getpid()}.tmp"
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
            # To the device, not just out of Python's buffer, for the same
            # reason `write_json` does it: a flushed line survives a killed
            # process but not a reboot, and a reboot is what cost the previous
            # harness 439 generations. One fsync per item is nothing beside the
            # seconds of inference that produced it.
            os.fsync(self._f.fileno())

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


@contextlib.contextmanager
def exclusive(run_dir: str) -> Iterator[None]:
    """Hold a run directory for the length of a run.

    The directory is named by the configuration, so launching the same command
    twice is an ordinary accident rather than a contrived one -- and both
    invocations would read `completed_ids` before either wrote, generate the
    same items, and append two records per id.

    An advisory `flock`, released when the process exits however it exits.
    Where there is no `fcntl` there is no lock; the run proceeds, because
    refusing to run at all on such a platform is a worse trade than the race.
    """
    os.makedirs(run_dir, exist_ok=True)
    if fcntl is None:  # pragma: no cover - Windows
        yield
        return
    f = open(os.path.join(run_dir, LOCK), "w")
    try:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Busy(
                f"{run_dir} is already held by another run (see "
                f"{LOCK}). Two runs sharing a directory both skip the ids the "
                f"other is generating and both append to the same file. Wait "
                f"for it, or pass a different run_id=.") from None
        f.write(f"pid {os.getpid()}\n")
        f.flush()
        yield
    finally:
        f.close()
