from __future__ import annotations

from pathlib import Path
import hashlib
import json

from transfer.v4.registry import get_adapter


ROOT = Path("/arf/scratch/futan/ArgGYM")

FREEZE_DIR = (
    ROOT
    / "outputs"
    / "transfer"
    / "protocol_freeze"
    / "v4"
    / "custom"
)

MANIFEST = FREEZE_DIR / "custom_v4_freeze_manifest.json"
SHA_FILE = FREEZE_DIR / "custom_v4_freeze_manifest.sha256"

EXPECTED_MANIFEST_SHA256 = (
    "975c1d31e43cff2ed3354695aaf13844129e5231dca027e44d3d3cf09b9d714b"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def normalize(obj):
    if obj is None:
        return None

    if isinstance(obj, (str, bool, int, float)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {
            str(k): normalize(v)
            for k, v in sorted(
                obj.items(),
                key=lambda kv: str(kv[0]),
            )
        }

    if isinstance(obj, (list, tuple)):
        return [normalize(x) for x in obj]

    if isinstance(obj, set):
        return sorted(
            (normalize(x) for x in obj),
            key=lambda x: json.dumps(
                x,
                sort_keys=True,
                ensure_ascii=False,
            ),
        )

    if hasattr(obj, "item"):
        try:
            return normalize(obj.item())
        except Exception:
            pass

    raise TypeError(
        f"Unsupported value type: "
        f"{type(obj)!r}: {obj!r}"
    )


def canonical_json(obj) -> bytes:
    return (
        json.dumps(
            normalize(obj),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def current_tree_inventory(root: Path):
    files = []

    for path in sorted(
        p for p in root.rglob("*")
        if p.is_file()
    ):
        files.append({
            "path":
                path.relative_to(root).as_posix(),
            "size":
                path.stat().st_size,
            "sha256":
                sha256_file(path),
        })

    return {
        "root":
            root.relative_to(ROOT).as_posix(),
        "file_count":
            len(files),
        "tree_sha256":
            sha256_bytes(canonical_json(files)),
        "files":
            files,
    }


def fail(message: str):
    raise RuntimeError(
        "CUSTOM V4 FREEZE VERIFICATION FAILED:\n"
        + message
    )


def verify_custom_freeze():
    # -----------------------------------------------------
    # Manifest identity.
    # -----------------------------------------------------

    if not MANIFEST.is_file():
        fail(f"manifest missing: {MANIFEST}")

    actual_manifest_sha = sha256_file(MANIFEST)

    if actual_manifest_sha != EXPECTED_MANIFEST_SHA256:
        fail(
            "manifest hash drift\n"
            f"expected={EXPECTED_MANIFEST_SHA256}\n"
            f"actual  ={actual_manifest_sha}"
        )

    if not SHA_FILE.is_file():
        fail(f"manifest SHA sidecar missing: {SHA_FILE}")

    sidecar_sha = (
        SHA_FILE.read_text(encoding="utf-8")
        .strip()
        .split()[0]
    )

    if sidecar_sha != EXPECTED_MANIFEST_SHA256:
        fail(
            "manifest SHA sidecar drift\n"
            f"expected={EXPECTED_MANIFEST_SHA256}\n"
            f"sidecar ={sidecar_sha}"
        )

    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    if (
        manifest.get("status")
        != "FROZEN_BEFORE_OFFICIAL_PRE"
    ):
        fail(
            "unexpected manifest status: "
            f"{manifest.get('status')!r}"
        )

    if manifest.get("total_rows") != 3944:
        fail(
            "unexpected total_rows: "
            f"{manifest.get('total_rows')!r}"
        )

    # -----------------------------------------------------
    # Locked code/config/audit files.
    # -----------------------------------------------------

    for rel, expected in manifest[
        "locked_files"
    ].items():
        path = ROOT / rel

        if not path.is_file():
            fail(f"locked file missing: {rel}")

        actual_sha = sha256_file(path)
        actual_size = path.stat().st_size

        if actual_sha != expected["sha256"]:
            fail(
                f"locked file hash drift: {rel}\n"
                f"expected={expected['sha256']}\n"
                f"actual  ={actual_sha}"
            )

        if actual_size != expected["size"]:
            fail(
                f"locked file size drift: {rel}\n"
                f"expected={expected['size']}\n"
                f"actual  ={actual_size}"
            )

    # -----------------------------------------------------
    # Provenance notes.
    # -----------------------------------------------------

    for name, expected in manifest.get(
        "provenance_notes",
        {},
    ).items():
        path = ROOT / expected["path"]

        if not path.is_file():
            fail(
                f"provenance file missing: "
                f"{name}: {path}"
            )

        actual_sha = sha256_file(path)

        if actual_sha != expected["sha256"]:
            fail(
                f"provenance hash drift: {name}\n"
                f"expected={expected['sha256']}\n"
                f"actual  ={actual_sha}"
            )

        if path.stat().st_size != expected["size"]:
            fail(
                f"provenance size drift: {name}"
            )

    # -----------------------------------------------------
    # Frozen data trees.
    # -----------------------------------------------------

    for benchmark, expected in manifest[
        "data_trees"
    ].items():
        root = ROOT / expected["root"]

        if not root.is_dir():
            fail(
                f"frozen data tree missing: "
                f"{benchmark}: {root}"
            )

        current = current_tree_inventory(root)

        if current["file_count"] != expected["file_count"]:
            fail(
                f"{benchmark}: file count drift\n"
                f"expected={expected['file_count']}\n"
                f"actual  ={current['file_count']}"
            )

        if current["tree_sha256"] != expected["tree_sha256"]:
            fail(
                f"{benchmark}: data tree hash drift\n"
                f"expected={expected['tree_sha256']}\n"
                f"actual  ={current['tree_sha256']}"
            )

        if current["files"] != expected["files"]:
            fail(
                f"{benchmark}: individual frozen "
                "file inventory drift"
            )

    # -----------------------------------------------------
    # Prompt-budget contract.
    # -----------------------------------------------------

    budget_path = (
        FREEZE_DIR
        / "prompt_budget_audit.json"
    )

    budget = json.loads(
        budget_path.read_text(encoding="utf-8")
    )

    checks = {
        "status": "PASS",
        "total": 3944,
        "context_window": 40960,
        "completion_budget": 16384,
        "prompt_budget": 24576,
        "over_budget_count": 0,
        "enable_thinking": True,
    }

    for key, expected in checks.items():
        actual = budget.get(key)

        if actual != expected:
            fail(
                f"prompt budget field drift: {key}\n"
                f"expected={expected!r}\n"
                f"actual  ={actual!r}"
            )

    if (
        budget["global_max"]["prompt_tokens"]
        != 22729
    ):
        fail(
            "unexpected global max prompt tokens: "
            f"{budget['global_max']['prompt_tokens']}"
        )

    if (
        budget["global_max"]["benchmark"]
        != "rulearena"
    ):
        fail(
            "unexpected max-prompt benchmark: "
            f"{budget['global_max']['benchmark']}"
        )

    # -----------------------------------------------------
    # Recompute exact adapter-emitted scientific inputs.
    # -----------------------------------------------------

    global_ids = set()

    for benchmark in manifest[
        "benchmark_names"
    ]:
        expected = manifest[
            "benchmarks"
        ][benchmark]

        adapter = get_adapter(benchmark)
        rows = list(adapter.rows())

        if len(rows) != expected["row_count"]:
            fail(
                f"{benchmark}: adapter row count drift\n"
                f"expected={expected['row_count']}\n"
                f"actual  ={len(rows)}"
            )

        if (
            getattr(adapter, "name", benchmark)
            != expected["adapter_name"]
        ):
            fail(
                f"{benchmark}: adapter name drift"
            )

        if (
            getattr(adapter, "version", None)
            != expected["adapter_version"]
        ):
            fail(
                f"{benchmark}: adapter version drift\n"
                f"expected={expected['adapter_version']!r}\n"
                f"actual="
                f"{getattr(adapter, 'version', None)!r}"
            )

        local_ids = set()
        h = hashlib.sha256()

        for row in rows:
            if row.row_id in local_ids:
                fail(
                    f"{benchmark}: duplicate row_id: "
                    f"{row.row_id}"
                )

            local_ids.add(row.row_id)

            global_id = (
                f"{benchmark}::{row.row_id}"
            )

            if global_id in global_ids:
                fail(
                    f"duplicate global ID: "
                    f"{global_id}"
                )

            global_ids.add(global_id)

            messages1 = adapter.messages(row)
            messages2 = adapter.messages(row)

            if messages1 != messages2:
                fail(
                    f"{benchmark}: nondeterministic "
                    f"messages for {row.row_id}"
                )

            scientific_record = {
                "benchmark": benchmark,
                "row_id": row.row_id,
                "payload": row.payload,
                "gold": row.gold,
                "metadata": row.metadata,
                "messages": messages1,
            }

            h.update(
                canonical_json(
                    scientific_record
                )
            )

        if (
            len(local_ids)
            != expected["unique_row_ids"]
        ):
            fail(
                f"{benchmark}: unique row count drift"
            )

        actual_input_sha = h.hexdigest()

        if (
            actual_input_sha
            != expected[
                "scientific_inputs_sha256"
            ]
        ):
            fail(
                f"{benchmark}: scientific input drift\n"
                f"expected="
                f"{expected['scientific_inputs_sha256']}\n"
                f"actual  ={actual_input_sha}"
            )

    if len(global_ids) != 3944:
        fail(
            f"global row total drift: "
            f"{len(global_ids)}"
        )

    print("=" * 72)
    print("CUSTOM V4 FREEZE VERIFICATION")
    print("=" * 72)
    print(
        "manifest sha256:",
        actual_manifest_sha,
    )
    print("rows:", len(global_ids))
    print(
        "max prompt tokens:",
        budget["global_max"]["prompt_tokens"],
    )
    print(
        "max prompt row:",
        budget["global_max"]["benchmark"],
        budget["global_max"]["row_id"],
    )
    print()
    print("CUSTOM V4 FREEZE VERIFICATION: PASS")

    return manifest


if __name__ == "__main__":
    verify_custom_freeze()
