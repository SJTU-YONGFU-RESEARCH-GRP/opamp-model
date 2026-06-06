#!/usr/bin/env python3
"""Compare opamp_metrics.json across python, ngspice, and spectre engines."""

from __future__ import annotations

import argparse
from pathlib import Path

from opamp_model.compare import (
    DEFAULT_ENGINES,
    compare_engines,
    format_compare_table,
    write_compare_report,
)
from opamp_model.io import package_root


def main() -> None:
    """Load per-engine metrics, print spread table, exit non-zero on tolerance breach."""
    parser = argparse.ArgumentParser(
        description="Compare opamp_metrics.json across simulation engines.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing <engine>/opamp_metrics.json (default: outputs).",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="Optional golden_metrics.yaml for a reference column (not pass/fail).",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=list(DEFAULT_ENGINES),
        help=f"Engines to compare (default: {' '.join(DEFAULT_ENGINES)}).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write outputs/COMPARE_REPORT.md.",
    )
    args = parser.parse_args()

    root = args.output_root
    if not root.is_absolute():
        root = package_root() / root

    golden = args.golden
    if golden is not None and not golden.is_absolute():
        golden = package_root() / golden

    result = compare_engines(
        root,
        engines=tuple(args.engines),
        golden_path=golden,
    )
    print(format_compare_table(result))
    if not args.no_report:
        report_path = write_compare_report(root, result)
        print(f"Wrote {report_path}")
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
