"""Three tasks share one builder, so the mode is an argument, not a default.

`make_item` used to fill a missing mode from `spec_for(level, ordering, seed % 5).mode`,
which picked attack, defence or mixed off the seed's parity. No caller reached it --
`core/registry.py` names the mode on each of the three tasks it registers -- and the
curriculum it read was consulted by nothing else in the module (#31).

With that gone the argument is required, so omitting it fails at the call, and a mode
that is not one of the three raises rather than reporting a rejection. Both matter: the
export records a rejection and scans on, so a typo would have been counted as a cell
where every candidate was refused and would have survived the export.
"""
from __future__ import annotations

import pytest

from arggym.core import registry
from arggym.core.curriculum import ATTACK, DEFENCE, MIXED
from arggym.tasks import attack_defense as ad

LAST_LINK = "last_link_elitist"


def test_every_registered_mode_names_itself():
    registered = {name: spec.variant.get("mode") for name, spec in registry.REGISTRY.items()
                  if spec.module == "attack_defense"}
    assert registered == {ATTACK: ATTACK, DEFENCE: DEFENCE, MIXED: MIXED}


def test_a_missing_mode_does_not_reach_the_builder():
    with pytest.raises(TypeError, match="mode"):
        ad.make_item(9, 0, LAST_LINK)


def test_an_unknown_mode_is_refused_rather_than_read_as_an_empty_cell():
    """A near miss of a real mode. A rejection here would export as a refused cell."""
    with pytest.raises(ValueError) as e:
        ad.make_item(9, 0, LAST_LINK, mode="defense")
    assert all(m in str(e.value) for m in (ATTACK, DEFENCE, MIXED))
