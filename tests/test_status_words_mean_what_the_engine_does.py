"""`overruled` has to mean what the engine reports, and the docs have to say that.

NOTATION.md defined overruled as "its contrary is in the extension". `status_map`
(`arggym/aspic/engine.py:216-230`) reads the arguments FOR the claim: justified if one is
IN, undecided if none is IN but one is UNDEC, overruled otherwise. A justified undercut
defeats every argument for a claim without putting the contrary anywhere, so the engine
says overruled while neither the claim nor its contrary is in the extension -- which the
written definition called undecided (#29).

Gold is derived from the engine on every task, so the engine is the definition and the
document was wrong.

The items below are exhibits of prompt text rather than coverage: what is read off them
is the definition sentence the task states, which the level does not change.
"""
from __future__ import annotations

import pathlib

import pytest

from arggym.aspic.api import ASPICVerifier
from arggym.aspic.engine import Operation
from arggym.tasks import perturbation as pt
from arggym.tasks import status_query as sq

GLOSS = ("A claim is justified when some argument for it is accepted, overruled when "
         "every argument for it is defeated, and undecided otherwise.")


def _undercut_theory():
    """`c` rests on one defeasible rule, and an undisputed argument switches that rule off."""
    return [
        Operation(kind="axiom", content="a"),
        Operation(kind="defeasible", name="r1", antecedents=("a",), consequent="c"),
        Operation(kind="axiom", content="u"),
        Operation(kind="strict", name="r2", antecedents=("u",), consequent="-r1"),
    ]


def test_an_undercut_overrules_a_claim_without_establishing_its_contrary():
    v = ASPICVerifier.from_operations(_undercut_theory(), ordering="last_link_elitist")
    assert str(v.status("c")) == "OVERRULED"
    assert str(v.status("-c")) != "JUSTIFIED", (
        "if the contrary were justified, the old definition would have agreed here "
        "and this would not be the case #29 is about")


def test_the_notation_no_longer_defines_overruled_by_the_contrary():
    doc = (pathlib.Path(__file__).resolve().parent.parent / "NOTATION.md").read_text()
    section = doc.split("## 5. Statuses", 1)[1].split("---", 1)[0]
    assert "every argument for it is defeated" in section
    assert "its contrary is in the extension; the conflict was RESOLVED" not in section


@pytest.mark.parametrize("mod", [sq, pt])
def test_the_status_tasks_state_the_definition(mod):
    it = mod.make_item(6, 0, "last_link_elitist")
    assert it is not None
    assert GLOSS in it.prompt
