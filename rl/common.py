"""Shared, dependency-light helpers for the ArgGYM RL harness."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def question_hash(row: Dict[str, Any]) -> str:
    return sha256_text(row["question"])


def theory_hash(row: Dict[str, Any]) -> Optional[str]:
    """Hash the theory visible in a row, independent of its seed/id.

    Formalization deliberately has no serialized theory.  For other tasks the
    theory fields live directly under metadata and are the only ``*_ops``
    fields outside ``gold``.  Include the ordering because the same operations
    under another ordering are a different formal state.
    """
    meta = row.get("metadata") or {}
    payload = {k: meta[k] for k in sorted(meta) if k.endswith("_ops")}
    if not payload:
        return None
    payload["ordering"] = meta.get("ordering")
    return sha256_text(canonical_json(payload))


def public_row_hash(row: Dict[str, Any]) -> str:
    """Hash everything model-visible/scorer-stateful except coordinate identity.

    ``reference_answer`` and ``metadata.gold`` are removed by construction.
    Seed/source_index/id are also removed so the same generated problem under a
    different coordinate still collides and is rejected across splits.
    """
    r = deepcopy(row)
    r.pop("reference_answer", None)
    r.pop("id", None)
    meta = r.get("metadata") or {}
    meta.pop("gold", None)
    meta.pop("seed", None)
    meta.pop("source_index", None)
    r["metadata"] = meta
    return sha256_text(canonical_json(r))


def row_record_hash(row: Dict[str, Any]) -> str:
    return sha256_text(canonical_json(row))


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical_json(row) + "\n")
