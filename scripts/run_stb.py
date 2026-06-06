#!/usr/bin/env python3
"""STB / loop-gain Bode testbench."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from opamp_model.ac import extract_gbw_phase_margin, plot_bode
from opamp_model.cli_helpers import (
    add_noise_args,
    add_opamp_args,
    add_output_args,
    add_simulator_args,
    build_noise_config,
    build_opamp_config,
    resolve_engine_label,
)
from opamp_model.io import package_root, write_bode_csv
from opamp_model.metrics import build_metrics_report, format_metrics_table, write_metrics_json
from opamp_model.report import (
    preserve_metrics_sections,
    read_metrics_json,
    write_engine_report,
    write_stb_report,
)
from opamp_model.model import AcSimulationResult, simulate_stb
from opamp_model.ngspice_engine import NgspiceNotFoundError, simulate_ac_ngspice
from opamp_model.simulation_log import SimulationLog, archive_veriloga_artifacts, log_run_context
from opamp_model.spectre_engine import SpectreLicenseError, SpectreNotFoundError, simulate_ac_spectre


def _scale_loop_gain(result: AcSimulationResult, loop_beta: float) -> AcSimulationResult:
    """Apply ``loop_beta`` to open-loop gain (dB) and recompute metrics."""
    gain_db = result["gain_db"] + 20.0 * math.log10(max(loop_beta, 1.0e-30))
    metrics = extract_gbw_phase_margin(
        result["frequency_hz"],
        gain_db,
        result["phase_deg"],
    )
    return AcSimulationResult(
        frequency_hz=result["frequency_hz"],
        gain_db=gain_db,
        phase_deg=result["phase_deg"],
        metrics=metrics,
    )


def main() -> None:
    """Run loop-gain simulation and write Bode CSV/SVG plus metrics JSON."""
    parser = argparse.ArgumentParser(description="Op-amp STB / loop-gain testbench.")
    add_opamp_args(parser)
    add_noise_args(parser)
    add_simulator_args(parser)
    add_output_args(parser)
    args = parser.parse_args()

    cfg = build_opamp_config(args)
    noise = build_noise_config(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = SimulationLog(out_dir / "logs" / "stb.log")
    log.write_header("stb", resolve_engine_label(args.simulator), cfg, noise)
    log_run_context(log)

    try:
        if args.simulator == "python":
            result = simulate_stb(cfg, noise)
        elif args.simulator == "ngspice":
            result = _scale_loop_gain(simulate_ac_ngspice(cfg, out_dir, noise), cfg.loop_beta)
        elif args.simulator == "spectre":
            result = _scale_loop_gain(simulate_ac_spectre(cfg, out_dir, noise), cfg.loop_beta)
        else:
            raise ValueError(f"Unknown simulator: {args.simulator}")
    except (NgspiceNotFoundError, SpectreNotFoundError, SpectreLicenseError) as exc:
        log.write(str(exc))
        log.close()
        raise SystemExit(str(exc)) from exc

    csv_path = out_dir / "stb_bode.csv"
    write_bode_csv(
        csv_path,
        result["frequency_hz"],
        result["gain_db"],
        result["phase_deg"],
    )
    bode_svg = out_dir / "stb_bode.svg"
    plot_bode(
        result["frequency_hz"],
        result["gain_db"],
        result["phase_deg"],
        bode_svg,
        title="Loop gain (STB)",
        metrics=result["metrics"],
    )
    report = build_metrics_report(
        cfg,
        noise,
        engine=args.simulator,
        stb_result=result,
    )
    report = preserve_metrics_sections(report, read_metrics_json(out_dir / "opamp_metrics.json"))
    write_metrics_json(out_dir / "opamp_metrics.json", report)
    stb_md = write_stb_report(
        out_dir,
        engine=args.simulator,
        cfg=cfg,
        report=report,
        bode_svg=bode_svg,
    )
    engine_md = write_engine_report(out_dir, engine=args.simulator)
    archive_veriloga_artifacts(package_root(), out_dir)
    log.write(f"wrote {csv_path}")
    log.write(f"wrote {bode_svg}")
    log.write(f"wrote {stb_md}")
    log.write(f"wrote {out_dir / 'opamp_metrics.json'}")
    if engine_md is not None:
        log.write(f"wrote {engine_md}")
    log.close()
    print(f"Wrote {csv_path}")
    print(f"Wrote {bode_svg}")
    print(f"Wrote {stb_md}")
    print(f"Wrote {out_dir / 'opamp_metrics.json'}")
    print(format_metrics_table(report))


if __name__ == "__main__":
    main()
