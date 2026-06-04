#!/usr/bin/env python3
"""AC / open-loop Bode testbench."""

from __future__ import annotations

import argparse
from pathlib import Path

from opamp_model.ac import cmrr_from_open_loop, plot_bode, plot_cmrr
from opamp_model.cli_helpers import (
    add_noise_args,
    add_opamp_args,
    add_output_args,
    add_simulator_args,
    build_noise_config,
    build_opamp_config,
    resolve_engine_label,
)
from opamp_model.io import package_root, write_bode_csv, write_cmrr_csv
from opamp_model.impedance import plot_impedance
from opamp_model.metrics import build_metrics_report, format_metrics_table, write_metrics_json
from opamp_model.model import simulate_ac
from opamp_model.ngspice_engine import NgspiceNotFoundError, simulate_ac_ngspice
from opamp_model.report import (
    preserve_metrics_sections,
    read_metrics_json,
    write_ac_report,
    write_engine_report,
)
from opamp_model.simulation_log import SimulationLog, archive_veriloga_artifacts, log_run_context
from opamp_model.spectre_engine import SpectreNotFoundError, simulate_ac_spectre


def main() -> None:
    """Run AC simulation and write Bode CSV/SVG plus metrics JSON."""
    parser = argparse.ArgumentParser(description="Op-amp AC / Bode testbench.")
    add_opamp_args(parser)
    add_noise_args(parser)
    add_simulator_args(parser)
    add_output_args(parser)
    args = parser.parse_args()

    cfg = build_opamp_config(args)
    noise = build_noise_config(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = SimulationLog(out_dir / "logs" / "python_ac.log")
    log.write_header("ac", resolve_engine_label(args.simulator), cfg, noise)
    log_run_context(log)

    try:
        if args.simulator == "python":
            result = simulate_ac(cfg, noise)
        elif args.simulator == "ngspice":
            result = simulate_ac_ngspice(cfg, out_dir, noise)
        elif args.simulator == "spectre":
            result = simulate_ac_spectre(cfg, out_dir, noise)
        else:
            raise ValueError(f"Unknown simulator: {args.simulator}")
    except (NgspiceNotFoundError, SpectreNotFoundError) as exc:
        log.write(str(exc))
        log.close()
        raise SystemExit(str(exc)) from exc

    csv_path = out_dir / "ac_bode.csv"
    write_bode_csv(
        csv_path,
        result["frequency_hz"],
        result["gain_db"],
        result["phase_deg"],
    )
    bode_svg = out_dir / "ac_bode.svg"
    plot_bode(
        result["frequency_hz"],
        result["gain_db"],
        result["phase_deg"],
        bode_svg,
        title="Open-loop AC",
        metrics=result["metrics"],
    )
    impedance_svg = out_dir / "impedance.svg"
    plot_impedance(cfg, impedance_svg, title="Input / output impedance")
    cmrr_result = cmrr_from_open_loop(
        result["frequency_hz"],
        result["gain_db"],
        cmrr_linear=cfg.cmrr_linear,
    )
    cmrr_csv = out_dir / "cmrr.csv"
    write_cmrr_csv(
        cmrr_csv,
        cmrr_result["frequency_hz"],
        cmrr_result["acm_db"],
        cmrr_result["cmrr_db"],
    )
    cmrr_svg = out_dir / "cmrr.svg"
    plot_cmrr(cmrr_result, cmrr_svg)
    report = build_metrics_report(
        cfg,
        noise,
        engine=args.simulator,
        ac_result=result,
        cmrr_result=cmrr_result,
    )
    report = preserve_metrics_sections(report, read_metrics_json(out_dir / "opamp_metrics.json"))
    write_metrics_json(out_dir / "opamp_metrics.json", report)
    ac_md = write_ac_report(
        out_dir,
        engine=args.simulator,
        cfg=cfg,
        noise=noise,
        report=report,
        bode_svg=bode_svg,
        impedance_svg=impedance_svg,
        cmrr_svg=cmrr_svg,
    )
    engine_md = write_engine_report(out_dir, engine=args.simulator)
    archive_veriloga_artifacts(package_root(), out_dir)
    log.write(f"wrote {csv_path}")
    log.write(f"wrote {bode_svg}")
    log.write(f"wrote {impedance_svg}")
    log.write(f"wrote {cmrr_csv}")
    log.write(f"wrote {cmrr_svg}")
    log.write(f"wrote {ac_md}")
    log.write(f"wrote {out_dir / 'opamp_metrics.json'}")
    if engine_md is not None:
        log.write(f"wrote {engine_md}")
    log.close()
    print(f"Wrote {csv_path}")
    print(f"Wrote {bode_svg}")
    print(f"Wrote {impedance_svg}")
    print(f"Wrote {cmrr_csv}")
    print(f"Wrote {cmrr_svg}")
    print(f"Wrote {ac_md}")
    print(f"Wrote {out_dir / 'opamp_metrics.json'}")
    print(format_metrics_table(report))


if __name__ == "__main__":
    main()
