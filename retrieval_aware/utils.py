"""Filesystem and serialization safeguards shared by E7 components."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from .config import E7Config


class E7SafetyError(RuntimeError):
    """Raised when an E7 operation could escape its isolated output tree."""


def _is_within(path: Path, parent: Path) -> bool:
    path = path.resolve()
    parent = parent.resolve()
    return path == parent or parent in path.parents


def assert_frozen_input_is_read_only(config: E7Config) -> None:
    """Reject configurations that place outputs in or over frozen inputs.

    This is a path-level guard.  E7 writers must additionally call
    ``assert_e7_output_path`` for every artifact they create.
    """
    data_root = config.data_root.resolve()
    frozen_root = config.frozen_root.resolve()
    frozen_query_dir = config.frozen_query_dir.resolve()
    output_root = config.output_root.resolve()

    if _is_within(output_root, data_root) or _is_within(data_root, output_root):
        raise E7SafetyError("E7 output_root must be disjoint from data/")
    if _is_within(output_root, frozen_root) or _is_within(frozen_root, output_root):
        raise E7SafetyError("E7 output_root must be disjoint from experiments/frozen/")
    if _is_within(output_root, frozen_query_dir) or _is_within(
        frozen_query_dir, output_root
    ):
        raise E7SafetyError("E7 output_root must be disjoint from frozen inputs")


def assert_e7_output_path(path: Path, config: E7Config) -> Path:
    """Return a resolved output path only if it is inside E7 output_root."""
    assert_frozen_input_is_read_only(config)
    resolved = path.resolve()
    if not _is_within(resolved, config.output_root):
        raise E7SafetyError(f"refusing to write outside E7 output_root: {resolved}")
    return resolved


def ensure_output_layout(config: E7Config) -> dict[str, Path]:
    """Create the E7-only artifact directories after all safety checks."""
    config.validate()
    assert_frozen_input_is_read_only(config)
    root = assert_e7_output_path(config.output_root, config)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, path in config.output_paths.items():
        safe_path = assert_e7_output_path(path, config)
        safe_path.mkdir(parents=True, exist_ok=True)
        paths[name] = safe_path
    return paths


def read_json(path: Path) -> Any:
    """Read JSON without modifying its source."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_e7_json(path: Path, value: Any, config: E7Config) -> None:
    """Atomically write JSON, but only under the configured E7 output root."""
    destination = assert_e7_output_path(path, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()


def write_e7_json_once(path: Path, value: Any, config: E7Config) -> None:
    """Write an immutable artifact and refuse to replace an existing file."""
    destination = assert_e7_output_path(path, config)
    if destination.exists():
        raise FileExistsError(f"immutable E7 artifact already exists: {destination}")
    write_e7_json(destination, value, config)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_document_text(text: str) -> str:
    """Canonicalize Unicode and whitespace without changing lexical case."""
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split())


def normalized_text_sha256(text: str) -> str:
    return hashlib.sha256(normalize_document_text(text).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
