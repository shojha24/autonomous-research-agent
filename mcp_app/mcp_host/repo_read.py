"""Safe, allowlisted reads of small repo files for the MCP tool harness.

Checks: no .. in path, path must be on allowlist, resolved path must stay
inside repo root (blocks symlink escape).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

_DEFAULT_MAX_BYTES: Final[int] = 65_536  # 64 KiB cap per read


def resolve_repo_root() -> Path:
    """Find the bioagent repository root directory.

    Uses BIOAGENT_ROOT if set; otherwise the parent of the mcp_host package.

    Returns:
        Absolute resolved path to the repo root.

    Raises:
        FileNotFoundError: BIOAGENT_ROOT is set but not an existing directory.
    """
    env_root = os.environ.get("BIOAGENT_ROOT", "").strip()
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if not candidate.is_dir():
            msg = f"BIOAGENT_ROOT is not a directory: {candidate}"
            raise FileNotFoundError(msg)
        return candidate
    return Path(__file__).resolve().parent.parent


def _has_parent_traversal(relative: Path) -> bool:
    """Check whether a path contains parent-directory (..) components.

    Args:
        relative: Path built from user-supplied segments.

    Returns:
        True if any path part is '..'.
    """
    return ".." in relative.parts


def _normalize_path_key(relative_posix: str) -> str:
    """Normalize a repo-relative path for allowlist comparison.

    Args:
        relative_posix: Forward-slash repo-relative path string.

    Returns:
        Normalized key with forward slashes and no leading './'.
    """
    return str(Path(relative_posix)).replace("\\", "/")


def read_repo_file(
    *,
    arguments: dict[str, Any],
    repo_root: Path,
    allowed_relative_paths: frozenset[str],
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """Read one allowlisted UTF-8 text file under repo_root.

    Args:
        arguments: Tool args; must include non-empty string "path".
        repo_root: Absolute repository root directory.
        allowed_relative_paths: Allowlisted paths from FILE_INDEX.md.
        max_bytes: Maximum bytes to read; larger files are truncated.

    Returns:
        On success: ok=True, path, content, truncated, bytes_read.
        On failure: ok=False and error message string.
    """
    raw_path = arguments.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return {"ok": False, "error": "Missing or invalid 'path' (non-empty string required)."}

    rel = Path(raw_path.strip())
    if rel.is_absolute() or _has_parent_traversal(rel):
        return {"ok": False, "error": "Path must be repo-relative with no '..' components."}

    key = _normalize_path_key(rel.as_posix())
    if key not in allowed_relative_paths:
        return {"ok": False, "error": f"Path not allowlisted: {key}"}

    full = (repo_root / rel).resolve()
    try:
        full.relative_to(repo_root.resolve())
    except ValueError:
        return {"ok": False, "error": "Resolved path escapes repository root."}

    if not full.is_file():
        return {"ok": False, "error": f"Not a file or does not exist: {key}"}

    try:
        data = full.read_bytes()
    except OSError as exc:
        return {"ok": False, "error": f"Read failed: {exc}"}

    truncated = len(data) > max_bytes
    chunk = data[:max_bytes]
    text = chunk.decode("utf-8", errors="replace")
    return {
        "ok": True,
        "path": key,
        "content": text,
        "truncated": truncated,
        "bytes_read": len(chunk),
    }
