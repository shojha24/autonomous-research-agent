"""Harness-only tools and allowlist loading (not on the MCP server).

The model can call read_repo_file to inspect allowlisted repo files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

# Markers in prompts/FILE_INDEX.md — only paths between these are readable.
_ALLOWLIST_BEGIN: Final[str] = "<!-- allowlist-begin -->"
_ALLOWLIST_END: Final[str] = "<!-- allowlist-end -->"

# Tool descriptor injected into the model's tool list.
READ_REPO_FILE_TOOL: dict[str, Any] = {
    "name": "read_repo_file",
    "description": (
        "Read a small, allowlisted repository file as UTF-8 text. "
        "Use only paths from the repository index in the system prompt; "
        "repo-relative forward slashes, no '..'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repo-relative path (must appear in FILE_INDEX allowlist).",
            },
        },
        "required": ["path"],
    },
}


def default_file_index_path(repo_root: Path) -> Path:
    """Return the default path to prompts/FILE_INDEX.md.

    Args:
        repo_root: Absolute path to the bioagent repository root.

    Returns:
        Path to prompts/FILE_INDEX.md under repo_root.
    """
    return repo_root / "prompts" / "FILE_INDEX.md"


def load_allowlisted_paths_from_index(index_path: Path) -> frozenset[str]:
    """Read allowed file paths from the allowlist block in FILE_INDEX.md.

    Scans lines between <!-- allowlist-begin --> and <!-- allowlist-end -->,
    ignoring blank lines and lines starting with #.

    Args:
        index_path: Path to FILE_INDEX.md.

    Returns:
        Frozen set of repo-relative POSIX path strings.

    Raises:
        FileNotFoundError: Index file missing.
        ValueError: Allowlist markers missing or empty block.
    """
    if not index_path.is_file():
        msg = f"File index not found: {index_path}"
        raise FileNotFoundError(msg)

    lines = index_path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start = lines.index(_ALLOWLIST_BEGIN) + 1
        end = lines.index(_ALLOWLIST_END)
    except ValueError as exc:
        msg = f"Allowlist markers missing in {index_path}"
        raise ValueError(msg) from exc

    if start >= end:
        msg = f"Empty or invalid allowlist block in {index_path}"
        raise ValueError(msg)

    paths: list[str] = []
    for line in lines[start:end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        paths.append(str(Path(stripped)).replace("\\", "/"))
    return frozenset(paths)


def load_file_index_excerpt(index_path: Path, *, max_chars: int = 100_000) -> str:
    """Load FILE_INDEX.md text for injection into the system prompt.

    Args:
        index_path: Path to FILE_INDEX.md.
        max_chars: Maximum characters to return before truncating.

    Returns:
        Full or truncated markdown text; placeholder string if file missing.
    """
    if not index_path.is_file():
        return f"(FILE_INDEX missing at {index_path})"

    text = index_path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[FILE_INDEX truncated for prompt size]\n"


def merge_harness_tools(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append harness-only read_repo_file to an MCP tool list.

    Args:
        mcp_tools: Tool descriptors from MCP list_tools or MOCK_TOOLS.

    Returns:
        New list with read_repo_file appended if not already present.
    """
    names = {t.get("name") for t in mcp_tools}
    merged = list(mcp_tools)
    if READ_REPO_FILE_TOOL["name"] not in names:
        merged.append(dict(READ_REPO_FILE_TOOL))
    return merged
