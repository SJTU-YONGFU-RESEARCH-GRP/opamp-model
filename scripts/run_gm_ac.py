#!/usr/bin/env python3
"""Gm / OTA AC testbench (loaded Vout/Vdiff Bode and gm vs f)."""

from __future__ import annotations

import argparse
from pathlib import Path

from opamp_model.cli_helpers import (
    add_noise_args,
    add_output_args,
    add_simulator_args,
    build_gm_config,
    build_noise_config,
    resolve_engine_label,
)
from opamp_model.config import OpampConfig
from opamp_model.gm import (
    plot_gm_bode,
    plot_gm_vs_f,
    simulate_gm_ac,
    write_gm_ac_csv,
    write_gm_ac_report,
)
from opamp_model.io import package_root
from opamp_model.metrics import build_metrics_report, write_metrics_json
from opamp_model.ngspice_engine import NgspiceNotFoundError, simulate_gm_ac_ngspice
from opamp_model.report import preserve_metrics_sections, read_metrics_json
from opamp_model.simulation_log import SimulationLog, archive_veriloga_artifacts, log_run_context
from opamp_model.spectre_engine import SpectreLicenseError, SpectreNotFoundError, simulate_gm_ac_spectre


def add_gm_args(parser: argparse.ArgumentParser) -> None:
    """Register Gm macromodel arguments."""
    parser.add_argument("--gm-s", type=float, default=1.0e-3, help="Transconductance (S).")
    parser.add_argument(
        "--gm-rout-ohm",
        type=float,
        default=1.0e6,
        help="Output shunt resistance (ohm).",
    )
    parser.add_argument(
        "--gm-cout-f",
        type=float,
        default=500.0e-15,
        help="Output shunt capacitance (F).",
    )


def main() -> None:
    """Run Gm AC simulation and write CSV/SVG plus report."""
    parser = argparse.ArgumentParser(description="Gm / OTA AC testbench.")
    add_gm_args(parser)
    add_noise_args(parser)
    add_simulator_args(parser)
    add_output_args(parser)
    args = parser.parse_args()

    gm_cfg = build_gm_config(args)
    opamp_cfg = OpampConfig()
    noise = build_noise_config(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_name = "python_gm_ac.log" if args.simulator == "python" else f"{args.simulator}_gm_ac.log"
    log = SimulationLog(out_dir / "logs" / log_name)
    log.write(
        f"# bench=gm_ac engine={resolve_engine_label(args.simulator)} "
        f"gm_s={gm_cfg.gm_s} rout_ohm={gm_cfg.rout_ohm} cout_f={gm_cfg.cout_f}"
    )
    log.write(f"# noise_enabled={noise.enabled} seed={noise.noise_seed}")
    log_run_context(log)

    try:
        if args.simulator == "python":
            result = simulate_gm_ac(gm_cfg, noise)
        elif args.simulator == "ngspice":
            result = simulate_gm_ac_ngspice(gm_cfg, out_dir, noise, sweep=opamp_cfg.sweep)
        elif args.simulator == "spectre":
            result = simulate_gm_ac_spectre(gm_cfg, out_dir, noise, sweep=opamp_cfg.sweep)
        else:
            raise ValueError(f"Unknown simulator: {args.simulator}")
    except (NgspiceNotFoundError, SpectreNotFoundError, SpectreLicenseError) as exc:
        log.write(str(exc))
        log.close()
        raise SystemExit(str(exc)) from exc

    csv_path = out_dir / "gm_ac_bode.csv"
    write_gm_ac_csv(
        csv_path,
        result["frequency_hz"],
        result["gain_db"],
        result["phase_deg"],
        result["gm_s"],
    )
    bode_svg = out_dir / "gm_ac_bode.svg"
    plot_gm_bode(result, bode_svg)
    gm_svg = out_dir / "gm_vs_f.svg"
    plot_gm_vs_f(result, gm_svg)
    report_md = write_gm_ac_report(
        out_dir,
        engine=args.simulator,
        gm_cfg=gm_cfg,
        result=result,
        bode_svg=bode_svg,
        gm_svg=gm_svg,
    )

    metrics_report = build_metrics_report(
        opamp_cfg,
        noise,
        engine=args.simulator,
        gm_result=result,
    )
    metrics_report = preserve_metrics_sections(
        metrics_report,
        read_metrics_json(out_dir / "opamp_metrics.json"),
    )
    write_metrics_json(out_dir / "opamp_metrics.json", metrics_report)

    archive_veriloga_artifacts(package_root(), out_dir)
    log.write(f"wrote {csv_path}")
    log.write(f"wrote {bode_svg}")
    log.write(f"wrote {gm_svg}")
    log.write(f"wrote {report_md}")
    log.write(f"wrote {out_dir / 'opamp_metrics.json'}")
    log.close()

    metrics = result["metrics"]
    print(f"Wrote {csv_path}")
    print(f"Wrote {bode_svg}")
    print(f"Wrote {gm_svg}")
    print(f"Wrote {report_md}")
    print(f"Wrote {out_dir / 'opamp_metrics.json'}")
    print(f"gm = {metrics['gm_s']:.6g} S, Gain(DC) = {metrics['gain_db']:.2f} dB")


if __name__ == "__main__":
    main()
