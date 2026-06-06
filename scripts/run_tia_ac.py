#!/usr/bin/env python3
"""TIA closed-loop AC / transimpedance testbench."""

from __future__ import annotations

import argparse
from pathlib import Path

from opamp_model.cli_helpers import (
    add_noise_args,
    add_opamp_args,
    add_output_args,
    add_simulator_args,
    add_tia_args,
    build_noise_config,
    build_opamp_config,
    build_tia_config,
    resolve_engine_label,
)
from opamp_model.io import package_root, write_tia_csv
from opamp_model.metrics import build_metrics_report, format_metrics_table, write_metrics_json
from opamp_model.ngspice_engine import NgspiceNotFoundError, simulate_tia_ac_ngspice
from opamp_model.report import (
    preserve_metrics_sections,
    read_metrics_json,
    write_engine_report,
    write_tia_report,
)
from opamp_model.simulation_log import SimulationLog, archive_veriloga_artifacts, log_run_context
from opamp_model.spectre_engine import SpectreLicenseError, SpectreNotFoundError, simulate_tia_ac_spectre
from opamp_model.tia import plot_zt, simulate_tia_ac


def main() -> None:
    """Run TIA AC simulation and write Zt CSV/SVG plus metrics JSON."""
    parser = argparse.ArgumentParser(description="TIA transimpedance AC testbench.")
    add_opamp_args(parser)
    add_tia_args(parser)
    add_noise_args(parser)
    add_simulator_args(parser)
    add_output_args(parser)
    args = parser.parse_args()

    cfg = build_opamp_config(args)
    tia = build_tia_config(args)
    noise = build_noise_config(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_name = "python_tia_ac.log" if args.simulator == "python" else f"{args.simulator}_tia_ac.log"
    log = SimulationLog(out_dir / "logs" / log_name)
    log.write_header("tia_ac", resolve_engine_label(args.simulator), cfg, noise)
    log_run_context(log)

    try:
        if args.simulator == "python":
            result = simulate_tia_ac(cfg, tia, noise)
        elif args.simulator == "ngspice":
            result = simulate_tia_ac_ngspice(cfg, tia, out_dir, noise)
        elif args.simulator == "spectre":
            result = simulate_tia_ac_spectre(cfg, tia, out_dir, noise)
        else:
            raise ValueError(f"Unknown simulator: {args.simulator}")
    except (NgspiceNotFoundError, SpectreNotFoundError, SpectreLicenseError) as exc:
        log.write(str(exc))
        log.close()
        raise SystemExit(str(exc)) from exc

    csv_path = out_dir / "tia_zt.csv"
    write_tia_csv(
        csv_path,
        result["frequency_hz"],
        result["zt_ohm"],
        result["zt_db"],
        result["zt_phase_deg"],
    )
    unit = "Ω" if tia.current_input else "V/V"
    zt_svg = out_dir / "tia_zt.svg"
    plot_zt(result, zt_svg, title="TIA closed-loop", unit=unit)

    report = build_metrics_report(
        cfg,
        noise,
        engine=args.simulator,
        tia_result=result,
    )
    report = preserve_metrics_sections(report, read_metrics_json(out_dir / "opamp_metrics.json"))
    write_metrics_json(out_dir / "opamp_metrics.json", report)
    tia_md = write_tia_report(
        out_dir,
        engine=args.simulator,
        cfg=cfg,
        tia=tia,
        report=report,
        zt_svg=zt_svg,
    )
    engine_md = write_engine_report(out_dir, engine=args.simulator)
    archive_veriloga_artifacts(package_root(), out_dir)
    log.write(f"wrote {csv_path}")
    log.write(f"wrote {zt_svg}")
    log.write(f"wrote {tia_md}")
    log.write(f"wrote {out_dir / 'opamp_metrics.json'}")
    if engine_md is not None:
        log.write(f"wrote {engine_md}")
    log.close()
    print(f"Wrote {csv_path}")
    print(f"Wrote {zt_svg}")
    print(f"Wrote {tia_md}")
    print(f"Wrote {out_dir / 'opamp_metrics.json'}")
    print(format_metrics_table(report))


if __name__ == "__main__":
    main()
