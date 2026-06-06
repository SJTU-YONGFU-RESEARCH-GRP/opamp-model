#!/usr/bin/env python3
"""AC compensation bench: gain peaking near GBW from open-loop Bode."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from opamp_model.ac import extract_gain_peaking_near_gbw, plot_ac_comp
from opamp_model.cli_helpers import (
    add_noise_args,
    add_opamp_args,
    add_output_args,
    add_simulator_args,
    build_noise_config,
    build_opamp_config,
    resolve_engine_label,
)
from opamp_model.io import package_root
from opamp_model.metrics import (
    OpampMetricsReport,
    build_ac_metric_entries,
    build_metrics_report,
    format_metrics_table,
    merge_ac_comp_entries,
    write_metrics_json,
)
from opamp_model.model import simulate_ac
from opamp_model.ngspice_engine import NgspiceNotFoundError, simulate_ac_ngspice
from opamp_model.report import (
    preserve_metrics_sections,
    read_metrics_json,
    write_ac_comp_report,
    write_engine_report,
)
from opamp_model.simulation_log import SimulationLog, archive_veriloga_artifacts, log_run_context
from opamp_model.spectre_engine import SpectreLicenseError, SpectreNotFoundError, simulate_ac_spectre


def main() -> None:
    """Run open-loop AC, extract peaking near GBW, and write plot plus report."""
    parser = argparse.ArgumentParser(
        description="Op-amp AC compensation / gain-peaking testbench.",
    )
    add_opamp_args(parser)
    add_noise_args(parser)
    add_simulator_args(parser)
    add_output_args(parser)
    args = parser.parse_args()

    cfg = replace(build_opamp_config(args), fp2_hz=200.0e6)
    noise = build_noise_config(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = SimulationLog(out_dir / "logs" / "ac_comp.log")
    log.write_header("ac_comp", resolve_engine_label(args.simulator), cfg, noise)
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
    except (NgspiceNotFoundError, SpectreNotFoundError, SpectreLicenseError) as exc:
        log.write(str(exc))
        log.close()
        raise SystemExit(str(exc)) from exc

    comp = extract_gain_peaking_near_gbw(
        result["frequency_hz"],
        result["gain_db"],
        result["phase_deg"],
    )
    comp_svg = out_dir / "ac_comp.svg"
    plot_ac_comp(
        result["frequency_hz"],
        result["gain_db"],
        result["phase_deg"],
        comp_svg,
        title="Gain peaking near GBW",
        comp_metrics=comp,
    )

    previous = read_metrics_json(out_dir / "opamp_metrics.json")
    base_cfg = build_opamp_config(args)
    report = build_metrics_report(
        base_cfg,
        noise,
        engine=args.simulator,
    )
    report = preserve_metrics_sections(report, previous)
    if previous and previous.get("ac"):
        ac_entries = merge_ac_comp_entries(previous["ac"], comp, engine=args.simulator)
    else:
        ac_entries = merge_ac_comp_entries(
            build_ac_metric_entries(result["metrics"], engine=args.simulator),
            comp,
            engine=args.simulator,
        )
    merged = dict(report)
    merged["ac"] = ac_entries
    if previous and previous.get("stb"):
        merged["stb"] = previous["stb"]
    typed_report = OpampMetricsReport(**merged)
    write_metrics_json(out_dir / "opamp_metrics.json", typed_report)

    ac_comp_md = write_ac_comp_report(
        out_dir,
        engine=args.simulator,
        cfg=cfg,
        report=typed_report,
        comp_svg=comp_svg,
        peak_db=comp["peak_db"],
        peak_freq_hz=comp["peak_freq_hz"],
        gbw_hz=comp["gbw_hz"],
    )
    engine_md = write_engine_report(out_dir, engine=args.simulator)
    archive_veriloga_artifacts(package_root(), out_dir)

    log.write(f"wrote {comp_svg}")
    log.write(f"wrote {ac_comp_md}")
    log.write(f"wrote {out_dir / 'opamp_metrics.json'}")
    if engine_md is not None:
        log.write(f"wrote {engine_md}")
    log.close()

    print(f"Wrote {comp_svg}")
    print(f"peak_db={comp['peak_db']:.4g} dB @ {comp['peak_freq_hz']:.6g} Hz")
    print(f"Wrote {ac_comp_md}")
    print(f"Wrote {out_dir / 'opamp_metrics.json'}")
    print(format_metrics_table(typed_report))


if __name__ == "__main__":
    main()
