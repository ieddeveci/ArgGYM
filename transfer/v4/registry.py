from __future__ import annotations

from transfer.v4.adapters.logiqa2 import LogiQA2Adapter
from transfer.v4.adapters.multilogieval import MultiLogiEvalAdapter
from transfer.v4.adapters.rulearena import RuleArenaAdapter


ADAPTERS = {
    "multilogieval": MultiLogiEvalAdapter,
    "rulearena": RuleArenaAdapter,
    "logiqa2": LogiQA2Adapter,
}


def get_adapter(name: str):
    try:
        cls = ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown v4 transfer benchmark: {name}"
        ) from exc

    return cls()
