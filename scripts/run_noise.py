#!/usr/bin/env python3
"""Noise spectrum and integrated-RMS testbench."""

from __future__ import annotations

import argparse
from pathlib import Path

from opamp_model.cli_helpers import (
    add_noise_args,
    add_opamp_args,
    add_output_args,
    add_simulator_args,
    build_noise_config,
    build_opamp_config,
    resolve_engine_label,
)
from opamp_model.io import package_root, write_noise_breakdown_csv, write_noise_csv
from opamp_model.metrics import build_metrics_report, format_metrics_table, write_metrics_json
from opamp_model.model import simulate_noise
from opamp_model.ngspice_engine import NgspiceNotFoundError, simulate_noise_ngspice
from opamp_model.noise_analysis import (
    compute_noise_breakdown,
    plot_noise_breakdown,
    plot_noise_spectrum,
)
from opamp_model.report import (
    preserve_metrics_sections,
    read_metrics_json,
    write_engine_report,
    write_noise_report,
)
from opamp_model.simulation_log import SimulationLog, archive_veriloga_artifacts, log_run_context
from opamp_model.spectre_engine import SpectreLicenseError, SpectreNotFoundError, simulate_noise_spectre


def main() -> None:
    """Run noise simulation and write spectrum CSV/SVG plus metrics JSON."""
    parser = argparse.ArgumentParser(description="Op-amp noise testbench.")
    add_opamp_args(parser)
    add_noise_args(parser)
    add_simulator_args(parser)
    add_output_args(parser)
    args = parser.parse_args()

    cfg = build_opamp_config(args)
    noise = build_noise_config(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = SimulationLog(out_dir / "logs" / "python_noise.log")
    log.write_header("noise", resolve_engine_label(args.simulator), cfg, noise)
    log_run_context(log)

    try:
        if args.simulator == "python":
            result = simulate_noise(cfg, noise)
        elif args.simulator == "ngspice":
            result = simulate_noise_ngspice(cfg, out_dir, noise)
        elif args.simulator == "spectre":
            result = simulate_noise_spectre(cfg, out_dir, noise)
        else:
            raise ValueError(f"Unknown simulator: {args.simulator}")
    except (NgspiceNotFoundError, SpectreNotFoundError, SpectreLicenseError) as exc:
        log.write(str(exc))
        log.close()
        raise SystemExit(str(exc)) from exc

    csv_path = out_dir / "noise_spectrum.csv"
    write_noise_csv(
        csv_path,
        result["frequency_hz"],
        result["noise_v_per_sqrt_hz"],
    )
    spectrum_svg = out_dir / "noise_spectrum.svg"
    plot_noise_spectrum(
        result["frequency_hz"],
        result["noise_v_per_sqrt_hz"],
        spectrum_svg,
        title="Output-referred noise",
        metrics=result["metrics"],
    )
    breakdown = compute_noise_breakdown(cfg, noise, result["frequency_hz"])
    breakdown_csv = out_dir / "noise_breakdown.csv"
    write_noise_breakdown_csv(
        breakdown_csv,
        breakdown["frequency_hz"],
        breakdown["en_in_white_v_per_sqrt_hz"],
        breakdown["en_in_flicker_v_per_sqrt_hz"],
        breakdown["en_in_total_v_per_sqrt_hz"],
        breakdown["en_out_white_v_per_sqrt_hz"],
        breakdown["en_out_flicker_v_per_sqrt_hz"],
        breakdown["en_out_total_v_per_sqrt_hz"],
    )
    breakdown_svg = out_dir / "noise_breakdown.svg"
    plot_noise_breakdown(breakdown, breakdown_svg)
    report = build_metrics_report(
        cfg,
        noise,
        engine=args.simulator,
        noise_result=result,
    )
    report = preserve_metrics_sections(report, read_metrics_json(out_dir / "opamp_metrics.json"))
    write_metrics_json(out_dir / "opamp_metrics.json", report)
    noise_md = write_noise_report(
        out_dir,
        engine=args.simulator,
        cfg=cfg,
        noise=noise,
        report=report,
        spectrum_svg=spectrum_svg,
        breakdown_svg=breakdown_svg,
        breakdown=breakdown,
    )
    engine_md = write_engine_report(out_dir, engine=args.simulator)
    archive_veriloga_artifacts(package_root(), out_dir)
    log.write(f"wrote {csv_path}")
    log.write(f"wrote {breakdown_csv}")
    log.write(f"wrote {spectrum_svg}")
    log.write(f"wrote {breakdown_svg}")
    log.write(f"wrote {noise_md}")
    log.write(f"wrote {out_dir / 'opamp_metrics.json'}")
    if engine_md is not None:
        log.write(f"wrote {engine_md}")
    log.close()
    print(f"Wrote {csv_path}")
    print(f"Wrote {breakdown_csv}")
    print(f"Wrote {spectrum_svg}")
    print(f"Wrote {breakdown_svg}")
    print(f"Wrote {noise_md}")
    print(f"Wrote {out_dir / 'opamp_metrics.json'}")
    print(format_metrics_table(report))


if __name__ == "__main__":
    main()
