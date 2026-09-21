from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import re
import sys

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
VENDOR = Path(__file__).resolve().parent / "vendor"


def ast_string(path: Path, name: str) -> str:
    """Read a top-level string assignment without executing upstream code.

    Prefer parsing the full source. If the pinned upstream script contains
    an unrelated syntax/indentation defect later in the file, fall back to
    parsing only the requested assignment expression.
    """
    text = path.read_text(encoding="utf-8")

    def from_tree(tree: ast.Module) -> str:
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == name
                    ):
                        value = ast.literal_eval(node.value)

                        if not isinstance(value, str):
                            raise TypeError(
                                f"{path}:{name} is not a string"
                            )

                        return value

            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == name
            ):
                value = ast.literal_eval(node.value)

                if not isinstance(value, str):
                    raise TypeError(
                        f"{path}:{name} is not a string"
                    )

                return value

        raise KeyError(f"{name} not found in {path}")

    try:
        return from_tree(ast.parse(text))
    except (SyntaxError, IndentationError):
        pass

    # Upstream may contain malformed code after the assignment we need.
    # Locate only `name = ...` and progressively parse that expression.
    pattern = re.compile(
        rf"(?m)^[ \t]*{re.escape(name)}[ \t]*=[ \t]*"
    )
    match = pattern.search(text)

    if match is None:
        raise KeyError(f"{name} not found in {path}")

    tail = text[match.end():]
    parts: list[str] = []

    for line in tail.splitlines(keepends=True):
        parts.append(line)
        candidate = "".join(parts).strip()

        if not candidate:
            continue

        try:
            fragment = ast.parse(
                "__value__ = " + candidate,
                mode="exec",
            )
        except (SyntaxError, IndentationError):
            continue

        if len(fragment.body) != 1:
            continue

        node = fragment.body[0]

        if not isinstance(node, ast.Assign):
            continue

        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            continue

        if not isinstance(value, str):
            raise TypeError(
                f"{path}:{name} is not a string"
            )

        return value

    raise SyntaxError(
        f"Could not parse string assignment "
        f"{name!r} from {path}"
    )


@contextmanager
def working_directory(path: Path):
    """Temporarily execute with path as CWD.

    Some pinned upstream evaluation code uses repository-relative file
    paths at import time. This preserves that behavior without modifying
    the vendored source itself.
    """
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@contextmanager
def prepend_sys_path(path: Path):
    value = str(path)
    sys.path.insert(0, value)

    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass


def fresh_import(
    module_name: str,
    directory: Path,
    clear: Iterable[str] = (),
):
    for name in clear:
        sys.modules.pop(name, None)

    sys.modules.pop(module_name, None)

    with prepend_sys_path(directory), working_directory(directory):
        return importlib.import_module(module_name)


def verify_vendor() -> dict:
    manifest_path = VENDOR / "manifest.json"

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    for rec in manifest["files"]:
        p = VENDOR / rec["path"]

        got = hashlib.sha256(p.read_bytes()).hexdigest()

        if got != rec["sha256"]:
            raise RuntimeError(
                f"vendor file hash mismatch: {rec['path']}"
            )

    return manifest
