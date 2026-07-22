"""Theory generation must be a pure function of its seed.

Iterating a Python `set` of strings makes output depend on PYTHONHASHSEED, which
is randomised per process. That cannot be caught inside a single interpreter, so
these tests re-run the generator in subprocesses under different hash seeds and
compare the results.

Run with pytest, or directly:  python tests/test_generation_determinism.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HASH_SEEDS = ("0", "1", "12345")

# Levels chosen to cover the preference-emitting paths; seeds kept small so the
# test stays under a couple of seconds per subprocess.
_SNIPPET = """
import hashlib, random, sys
sys.path.insert(0, {repo!r})
from aspic_gym import _raw_sample, _config, render_dsl

rendered = []
for level in (4, 5, 8, 12, 15):
    for seed in range(20):
        rendered.append(render_dsl(_raw_sample(random.Random(seed), _config(level))))

full = hashlib.sha256("\\n||\\n".join(rendered).encode()).hexdigest()

# Order-insensitive digest of the preference block: catches direction flips even
# if the emission order were somehow stabilised by accident.
pref_sets = sorted(
    "|".join(sorted(l for l in r.splitlines() if "prefer_" in l)) for r in rendered
)
pref = hashlib.sha256("\\n".join(pref_sets).encode()).hexdigest()
print(full + " " + pref)
"""


def _digests(hash_seed: str) -> tuple[str, str]:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    out = subprocess.run(
        [sys.executable, "-c", _SNIPPET.format(repo=str(REPO))],
        capture_output=True, text=True, env=env, check=True,
    ).stdout.split()
    return out[0], out[1]


def test_generation_is_invariant_under_hash_seed():
    """Same seeds must give the same theories regardless of PYTHONHASHSEED."""
    results = {h: _digests(h) for h in HASH_SEEDS}
    full = {h: d[0] for h, d in results.items()}
    assert len(set(full.values())) == 1, (
        "theory generation depends on PYTHONHASHSEED; digests: " + repr(full)
    )


def test_preference_sets_are_invariant_under_hash_seed():
    """The stronger property: the *set* of preferences, ignoring emission order.

    If this fails while the test above passes, generation is emitting different
    preferences -- including reversed directions -- not merely reordering them.
    """
    results = {h: _digests(h) for h in HASH_SEEDS}
    pref = {h: d[1] for h, d in results.items()}
    assert len(set(pref.values())) == 1, (
        "emitted preference sets depend on PYTHONHASHSEED; digests: " + repr(pref)
    )


# --- Full-pipeline determinism -------------------------------------------------
# The two tests above exercise only _raw_sample. Several framers emit preferences
# from their own set iteration (claim_identification, preference_construction),
# which _raw_sample coverage misses -- and which produced two byte-different
# tasksets from one seed. This builds real cells end to end and hashes the
# prompt + gold that actually ship.

# Preference-emitting tasks at a level where undercuts/preferences are active.
_PIPELINE_SNIPPET = """
import hashlib, json, sys
sys.path.insert(0, {repo!r})
from evals.taskset import build_rows

TASKS = ["claim_identification", "preference_construction", "status_query",
         "ordering_sensitivity", "attack"]
rows = []
for t in TASKS:
    for mode in ("symbolic", "content"):
        rows += build_rows([t], [mode], [5], 3, 20260721, True)
blob = "\\n".join(r["prompt"] + "\\x00" + r["entry"]["answer"]
                  for r in sorted(rows, key=lambda r: r["sample_id"]))
print(hashlib.sha256(blob.encode()).hexdigest())
"""


def _pipeline_digest(hash_seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    return subprocess.run(
        [sys.executable, "-c", _PIPELINE_SNIPPET.format(repo=str(REPO))],
        capture_output=True, text=True, env=env, check=True,
    ).stdout.strip()


def test_full_pipeline_is_invariant_under_hash_seed():
    """Prompt + gold of real generated cells must not depend on PYTHONHASHSEED.

    Covers the framer-level preference emission that the _raw_sample tests do
    not reach.
    """
    digests = {h: _pipeline_digest(h) for h in ("0", "1")}
    assert len(set(digests.values())) == 1, (
        "generated cells depend on PYTHONHASHSEED; digests: " + repr(digests)
    )


if __name__ == "__main__":
    test_generation_is_invariant_under_hash_seed()
    test_preference_sets_are_invariant_under_hash_seed()
    test_full_pipeline_is_invariant_under_hash_seed()
    print("ok: generation is invariant under PYTHONHASHSEED")
