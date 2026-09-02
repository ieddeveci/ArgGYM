"""The export registry and the inspector's mode list must name the same modes (issue #23)."""
from __future__ import annotations


def test_exportable_matches_inspector_modes():
    from arggym import inspector
    from arggym.core import export

    assert set(export._EXPORTABLE) == {m[0] for m in inspector.MODES}
