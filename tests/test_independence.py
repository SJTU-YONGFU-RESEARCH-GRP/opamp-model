"""Ensure the package has no forbidden legacy imports or paths."""

from __future__ import annotations

from opamp_model.independence import scan_tree
from opamp_model.io import package_root


def test_no_forbidden_references_in_package() -> None:
    """Fail when forbidden legacy import/path strings appear in the tree."""
    root = package_root()
    hits = scan_tree(root)
    if not hits:
        return
    lines = [f"{path}:{line_no}: {text}" for path, line_no, text in hits]
    msg = "Forbidden references:\n" + "\n".join(lines)
    raise AssertionError(msg)


def test_package_root_points_at_opamp_model() -> None:
    """package_root() resolves to the directory containing pyproject.toml."""
    root = package_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "opamp_model").is_dir()
