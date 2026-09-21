from transfer.adapters.multilogieval import MultiLogiEvalAdapter
from transfer.adapters.rulearena import RuleArenaAdapter
from transfer.adapters.logiqa2 import LogiQA2Adapter
from transfer.adapters.finereason import FineReasonAdapter


ADAPTERS = {
    "multilogieval":
        MultiLogiEvalAdapter,
    "rulearena":
        RuleArenaAdapter,
    "logiqa2":
        LogiQA2Adapter,
    "finereason":
        FineReasonAdapter,
}


def get_adapter(name: str):
    try:
        return ADAPTERS[name]()
    except KeyError:
        raise KeyError(
            f"Unknown transfer adapter: {name}"
        )
