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


if __name__ == "__main__":
    test_generation_is_invariant_under_hash_seed()
    test_preference_sets_are_invariant_under_hash_seed()
    print("ok: generation is invariant under PYTHONHASHSEED")
