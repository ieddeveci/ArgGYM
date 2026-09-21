from __future__ import annotations

import json
import re

from pathlib import Path

from transfer.base import TransferRow
from pathlib import Path

FROZEN = (
    Path("/arf/scratch/futan/ArgGYM")
    / "data"
    / "transfer"
    / "v4"
    / "frozen"
)


# Multi-LogiEval EMNLP 2024 zero-shot-CoT evaluation prompt.
PROMPT_PREFIX = (
    "Given the context that contains rules of logical reasoning in "
    "natural language and question, perform step-by-step reasoning "
    "to answer the question. Based on context and reasoning steps, "
    "answer the question ONLY in 'yes' or 'no.' Please use the below format:"
)


class MultiLogiEvalAdapter:
    name = "multilogieval"
    version = "repo-6d55ade5-d1-d5-emnlp-zscot-v4"

    def __init__(self, root: Path | None = None):
        self.root = root or (FROZEN / "multilogieval")

    @staticmethod
    def _load(path: Path):
        raw = path.read_bytes()

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp1252")

        return json.loads(text)

    def rows(self):
        family_map = {
            "pl": "propositional",
            "fol": "first_order",
            "nm": "non_monotonic",
        }

        for depth in range(1, 6):
            depth_root = self.root / f"d{depth}_Data"

            for path in sorted(depth_root.rglob("*.json")):
                obj = self._load(path)

                family_dir = path.relative_to(depth_root).parts[0]

                for idx, sample in enumerate(obj["samples"]):
                    sample_id = sample.get("id")
                    rel = path.relative_to(self.root).as_posix()

                    yield TransferRow(
                        row_id=(
                            f"multilogieval:{rel}:{idx}:{sample_id}"
                        ),
                        payload={
                            "context": sample["context"],
                            "question": sample["question"],
                        },
                        gold=str(
                            sample["answer"]
                        ).strip().lower(),
                        metadata={
                            "depth": obj.get(
                                "depth",
                                f"d{depth}",
                            ),
                            "rule": obj.get("rule"),
                            "raw_logic": obj.get("logic"),
                            "logic_family":
                                family_map[family_dir],
                            "source_file": rel,
                            "sample_id": sample_id,
                        },
                    )

    def messages(self, row):
        user = (
            f"{PROMPT_PREFIX}\n"
            f"Context: {row.payload['context']}\n"
            f"Question: {row.payload['question']}\n"
            "Reasoning steps:"
        )

        return [
            {
                "role": "user",
                "content": user,
            }
        ]

    def parse(self, completion, row):
        matches = re.findall(
            r"(?i)\bAnswer\s*:\s*(yes|no)\b",
            completion,
        )

        return matches[-1].lower() if matches else None

    def score(self, prediction, row):
        return float(prediction == row.gold)
