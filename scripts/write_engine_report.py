#!/usr/bin/env python3
"""Regenerate top-level REPORT.md for one engine output directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from opamp_model.cli_helpers import add_output_args, add_simulator_args
from opamp_model.report import write_engine_report


def main() -> None:
    """Write or refresh ``REPORT.md`` from existing bench artifacts."""
    parser = argparse.ArgumentParser(description="Build engine-level REPORT.md.")
    add_simulator_args(parser)
    add_output_args(parser)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    path = write_engine_report(out_dir, engine=args.simulator)
    if path is None:
        print(f"No AC/STB reports found under {out_dir}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
