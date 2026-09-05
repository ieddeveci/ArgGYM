"""The registry and the inspector's mode list must name the same tasks (issue #23)."""
from __future__ import annotations


def test_registry_matches_inspector_modes():
    from arggym import inspector
    from arggym.core import registry

    assert set(registry.task_names()) == {m[0] for m in inspector.MODES}
