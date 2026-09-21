from __future__ import annotations

import json
import re

from transfer.base import TransferRow
from transfer.integrity import FROZEN
from transfer.vendor_utils import VENDOR, ast_string


LABEL = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}


class LogiQA2Adapter:
    name = "logiqa2"
    version = "955e1d3-english-mrc-test"

    def __init__(self):
        self.path = (
            FROZEN
            / "logiqa2"
            / "logiqa"
            / "DATA"
            / "LOGIQA"
            / "test.txt"
        )

        script = (
            VENDOR
            / "logiqa2"
            / "multi-choice-prompt.py"
        )

        # Exact official in-context prefix, extracted without
        # executing the original script.
        self.incontext = ast_string(
            script,
            "incontext",
        )

    def rows(self):
        text = self.path.read_text(
            encoding="utf-8"
        )

        for line_idx, line in enumerate(
            text.splitlines()
        ):
            if not line.strip():
                continue

            obj = json.loads(line)

            yield TransferRow(
                row_id=(
                    f"logiqa2:{line_idx}:{obj.get('id')}"
                ),
                payload=obj,
                gold=LABEL[int(obj["answer"])],
                metadata={
                    "source_line_index": line_idx,
                    "source_id": obj.get("id"),
                },
            )

    def messages(self, row):
        obj = row.payload

        options = "".join(
            f"{LABEL[i]} {option}\n"
            for i, option in enumerate(
                obj["options"]
            )
        )

        prompt_input = (
            "Write a multi-choice question for the following article:\n"
            f"Article: {obj['text']}\n"
            f"Question: {obj['question']}\n"
            f"Options: {options}"
            "Answer: \n"
        )

        return [
            {
                "role": "user",
                "content":
                    self.incontext + prompt_input,
            }
        ]

    def parse(self, completion, row):
        text = completion.strip()

        match = re.search(
            r"(?im)^\s*Answer\s*:\s*([ABCD])\b",
            text,
        )

        if match:
            return match.group(1).upper()

        match = re.match(
            r"^\s*([ABCD])\b",
            text,
            flags=re.I,
        )

        if match:
            return match.group(1).upper()

        return None

    def score(self, prediction, row):
        return float(prediction == row.gold)
