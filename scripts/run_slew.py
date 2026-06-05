#!/usr/bin/env python3
"""TRAN slew-rate testbench (unity-gain step response)."""

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
from opamp_model.config import OpampConfig
from opamp_model.io import write_slew_step_csv
from opamp_model.metrics import build_metrics_report, format_metrics_table, write_metrics_json
from opamp_model.report import (
    preserve_metrics_sections,
    read_metrics_json,
    write_engine_report,
    write_slew_report,
)
from opamp_model.ngspice_engine import NgspiceNotFoundError, run_ngspice_tran_stub
from opamp_model.simulation_log import SimulationLog, log_run_context
from opamp_model.spectre_engine import SpectreNotFoundError, run_spectre_tran_stub
from opamp_model.tran import (
    measure_slew_rates,
    plot_slew_step,
    plot_transient_noise_trace,
    transient_noise_rms,
)


def _cfg_for_slew(args: argparse.Namespace, *, ideal: bool) -> OpampConfig:
    """Build config with optional unlimited slew for ``--ideal``."""
    cfg = build_opamp_config(args)
    if ideal:
        return OpampConfig(
            a0_db=cfg.a0_db,
            gbw_hz=cfg.gbw_hz,
            cmrr_db=cfg.cmrr_db,
            psrr_db=cfg.psrr_db,
            rin_ohm=cfg.rin_ohm,
            rout_ohm=cfg.rout_ohm,
            loop_beta=cfg.loop_beta,
            nl_a2=cfg.nl_a2,
            nl_a3=cfg.nl_a3,
            vswing_high_v=cfg.vswing_high_v,
            vswing_low_v=cfg.vswing_low_v,
            vcm_v=cfg.vcm_v,
            slew_pos_vps=1.0e18,
            slew_neg_vps=-1.0e18,
        )
    return OpampConfig(
        a0_db=cfg.a0_db,
        gbw_hz=cfg.gbw_hz,
        cmrr_db=cfg.cmrr_db,
        psrr_db=cfg.psrr_db,
        rin_ohm=cfg.rin_ohm,
        rout_ohm=cfg.rout_ohm,
        loop_beta=cfg.loop_beta,
        nl_a2=cfg.nl_a2,
        nl_a3=cfg.nl_a3,
        vswing_high_v=cfg.vswing_high_v,
        vswing_low_v=cfg.vswing_low_v,
        vcm_v=cfg.vcm_v,
        slew_pos_vps=args.slew_pos_vps,
        slew_neg_vps=args.slew_neg_vps,
    )


def main() -> None:
    """Run positive/negative step tests and write CSV, SVG, and metrics."""
    parser = argparse.ArgumentParser(description="Op-amp TRAN slew-rate testbench.")
    add_opamp_args(parser)
    add_noise_args(parser)
    add_simulator_args(parser)
    add_output_args(parser)
    parser.add_argument(
        "--step-v",
        type=float,
        default=0.8,
        help="Step amplitude (V) for each polarity test.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=2.0e-6,
        help="Transient duration (s).",
    )
    parser.add_argument(
        "--dt-s",
        type=float,
        default=1.0e-9,
        help="Time step (s).",
    )
    parser.add_argument(
        "--slew-pos-vps",
        type=float,
        default=10.0e6,
        help="Positive slew limit (V/s).",
    )
    parser.add_argument(
        "--slew-neg-vps",
        type=float,
        default=-10.0e6,
        help="Negative slew limit (V/s).",
    )
    args = parser.parse_args()

    cfg = _cfg_for_slew(args, ideal=args.ideal)
    noise = build_noise_config(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = SimulationLog(out_dir / "logs" / "slew.log")
    log.write_header("slew", resolve_engine_label(args.simulator), cfg, noise)
    log_run_context(log)

    try:
        if args.simulator == "ngspice":
            run_ngspice_tran_stub(
                cfg,
                out_dir,
                template_name="slew_stub.cir",
                log_name="ngspice_slew_stub.log",
            )
        elif args.simulator == "spectre":
            run_spectre_tran_stub(
                cfg,
                out_dir,
                template_name="slew_stub.scs",
                log_name="spectre_slew_stub.log",
            )
    except (NgspiceNotFoundError, SpectreNotFoundError) as exc:
        log.write(str(exc))
        log.close()
        raise SystemExit(str(exc)) from exc

    metrics, pos, neg = measure_slew_rates(
        cfg,
        noise,
        step_v=args.step_v,
        duration_s=args.duration_s,
        dt_s=args.dt_s,
    )

    csv_path = out_dir / "slew_step.csv"
    write_slew_step_csv(csv_path, pos["time_s"], pos["vout_v"], neg["vout_v"])
    noise_rms_v = transient_noise_rms(pos["noise_v"]) if noise.enabled else 0.0
    slew_svg = out_dir / "slew.svg"
    plot_slew_step(pos, neg, slew_svg, metrics=metrics, noise_rms_v=noise_rms_v)
    noise_trace_svg: Path | None = None
    if noise.enabled and noise_rms_v > 0.0:
        noise_trace_svg = out_dir / "slew_noise.svg"
        plot_transient_noise_trace(
            pos["time_s"],
            pos["noise_v"],
            noise_trace_svg,
            noise_rms_v=noise_rms_v,
        )

    report = build_metrics_report(
        cfg,
        noise,
        engine=args.simulator,
        ideal_flag=args.ideal,
        slew_pos_measured=metrics["slew_pos_vps"],
        slew_neg_measured=metrics["slew_neg_vps"],
    )
    report = preserve_metrics_sections(report, read_metrics_json(out_dir / "opamp_metrics.json"))
    write_metrics_json(out_dir / "opamp_metrics.json", report)
    slew_md = write_slew_report(
        out_dir,
        engine=args.simulator,
        cfg=cfg,
        report=report,
        slew_svg=slew_svg,
        noise=noise,
        noise_rms_v=noise_rms_v,
        noise_trace_svg=noise_trace_svg,
    )
    engine_md = write_engine_report(out_dir, engine=args.simulator)

    log.write(f"wrote {csv_path}")
    log.write(f"wrote {slew_svg}")
    log.write(f"wrote {slew_md}")
    log.write(f"wrote {out_dir / 'opamp_metrics.json'}")
    if engine_md is not None:
        log.write(f"wrote {engine_md}")
    log.close()

    print(f"Wrote {csv_path}")
    print(f"Wrote {slew_svg}")
    print(f"Wrote {slew_md}")
    print(f"Wrote {out_dir / 'opamp_metrics.json'}")
    print(format_metrics_table(report))


if __name__ == "__main__":
    main()
