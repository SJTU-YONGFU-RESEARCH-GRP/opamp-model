"""Scan the package tree for forbidden external references."""

from __future__ import annotations

import re
from pathlib import Path

# Patterns that must not appear in shipped code.
_FORBIDDEN = re.compile(
    r"adc-model|OPAMP_RAK|(?:from|import)\s+adc_model",
    re.IGNORECASE,
)

_SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs"}
_TEXT_SUFFIXES = {".py", ".sh", ".va", ".scs", ".cir", ".md", ".yaml", ".yml", ".toml"}
# Files that document or implement the guard itself (not external dependencies).
_SKIP_FILES = {
    "independence.py",
    "test_independence.py",
    "check_independence.sh",
    "README.md",
}


def scan_tree(root: Path) -> list[tuple[Path, int, str]]:
    """Return ``(path, line_no, line)`` for each forbidden match under ``root``."""
    hits: list[tuple[Path, int, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.name in _SKIP_FILES:
            continue
        if path.suffix not in _TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if _FORBIDDEN.search(line):
                hits.append((path, line_no, line.strip()))
    return hits
