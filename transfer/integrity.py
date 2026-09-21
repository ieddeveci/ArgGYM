from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "transfer" / "frozen"
CONFIG = ROOT / "configs" / "transfer" / "benchmarks.yaml"

EXPECTED_CONFIG_SHA256 = (
    "7b0e375507e5098355477eff6e32549ac639d73b32659655a1ffdf400e5f1b25"
)

EXPECTED_MANIFEST_SHA256 = (
    "1ccbab4c9807951464034b8519c3c373c1872f6356a25e76d6ee71afd8e73df2"
)

EXPECTED_TREE_SHA256 = {
    "finereason":
        "4e043e7560233f3f49da776303ecebec8eea2299286a16869018a936cc1c2be7",
    "logiqa2":
        "971219629df8885be141c08087273c4e5443a3301ed13102c9f4185eb9bdd106",
    "multilogieval":
        "8787a48ec67dccd76b9f1b39f599194a04ff8152c56363839907c1585f5690bc",
    "rulearena":
        "4ee1178b5d60edc3c5113a42b7e555a0ba7fb966e71ac9ddf2b485936a343c2c",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def verify_frozen(full_file_check: bool = True) -> dict:
    got_config = sha256_file(CONFIG)

    if got_config != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(
            "configs/transfer/benchmarks.yaml hash mismatch: "
            f"{got_config} != {EXPECTED_CONFIG_SHA256}"
        )

    top = FROZEN / "manifest.json"
    got_manifest = sha256_file(top)

    if got_manifest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "data/transfer/frozen/manifest.json hash mismatch: "
            f"{got_manifest} != {EXPECTED_MANIFEST_SHA256}"
        )

    manifest = json.loads(
        top.read_text(encoding="utf-8")
    )

    for name, expected in EXPECTED_TREE_SHA256.items():
        got = manifest["benchmarks"][name]["tree_sha256"]

        if got != expected:
            raise RuntimeError(
                f"{name}: tree hash mismatch: "
                f"{got} != {expected}"
            )

        if full_file_check:
            benchmark_manifest_path = (
                FROZEN / name / "manifest.json"
            )

            benchmark_manifest = json.loads(
                benchmark_manifest_path.read_text(
                    encoding="utf-8"
                )
            )

            for rec in benchmark_manifest["files"]:
                p = FROZEN / name / rec["path"]

                if not p.is_file():
                    raise RuntimeError(
                        f"{name}: missing frozen file: "
                        f"{rec['path']}"
                    )

                got_file = sha256_file(p)

                if got_file != rec["sha256"]:
                    raise RuntimeError(
                        f"{name}: frozen file hash mismatch: "
                        f"{rec['path']}: "
                        f"{got_file} != {rec['sha256']}"
                    )

    return manifest
