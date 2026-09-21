from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Protocol


@dataclass(frozen=True)
class TransferRow:
    row_id: str
    payload: Dict[str, Any]
    gold: Any
    metadata: Dict[str, Any]


class TransferBenchmark(Protocol):
    name: str
    version: str

    def rows(self) -> Iterable[TransferRow]: ...
    def messages(self, row: TransferRow) -> list[dict[str, str]]: ...
    def parse(self, completion: str, row: TransferRow) -> Any: ...
    def score(self, prediction: Any, row: TransferRow) -> float: ...
