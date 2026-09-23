"""Every model runs under the `cot` elicitation, and `cot` adds only the ask.

The laptop defaulted to `none` while the HPC lane defaulted to `cot`, so one
config run both ways landed in two directories and only one was the condition
the paper compares (#179). And the old `cot` text restated what every question
already says: its `Answer format:` block, and the template's "Give your final
answer between <answer> and </answer>."
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_every_place_that_writes_a_default_elicitation_says_cot():
    conf = yaml.safe_load((ROOT / "evals/conf/config.yaml").read_text())
    defaults = {k: v for d in conf["defaults"] if isinstance(d, dict) for k, v in d.items()}
    found = {"evals/conf/config.yaml": defaults.get("elicitation")}
    for path, pattern in (
        ("Makefile", r"^ELICITATION \?= (\S+)$"),
        ("hpc/vllm/submit_truba.sh", r"ELICITATION=\$\{ELICITATION:-([^}]+)\}"),
        ("hpc/vllm/truba_vllm.sbatch", r"\$\{ELICITATION:-([^}]+)\}"),
        ("hpc/vllm/run_vllm_arggym.py",
         r"add_argument\(\"--elicitation\", default=\"([^\"]+)\"\)"),
    ):
        hits = re.findall(pattern, (ROOT / path).read_text(), flags=re.M)
        assert hits, f"{path}: no default elicitation found"
        found[path] = hits[0] if len(set(hits)) == 1 else hits
    assert all(v == "cot" for v in found.values()), found


def test_the_cot_text_says_nothing_the_question_or_the_template_says():
    cot = yaml.safe_load((ROOT / "evals/conf/elicitation/cot.yaml").read_text())
    assert cot["name"] == "cot"
    assert cot["prefix"] == "" and cot["suffix"] == ""
    system = cot["system"].lower()
    assert system, "cot asks for nothing"
    # The question's `Answer format:` block and zero-score sentence, and the
    # template's answer tags, already cover all of these.
    for said in ("format", "<answer>", "answer tag", "final answer", "score"):
        assert said not in system, f"cot repeats {said!r}"


def test_the_standalone_example_asks_what_the_harness_asks():
    """`examples/evaluate.py` carries its own copy, so it can drift unseen."""
    cot = yaml.safe_load((ROOT / "evals/conf/elicitation/cot.yaml").read_text())
    example = (ROOT / "examples/evaluate.py").read_text()
    m = re.search(r'^SYSTEM = "([^"]*)"$', example, flags=re.M)
    assert m, "examples/evaluate.py no longer defines SYSTEM on one line"
    assert m.group(1) == cot["system"]
