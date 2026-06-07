#!/usr/bin/env python3
"""THD testbench: large-signal sine steady state with FFT distortion analysis."""

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
from opamp_model.io import package_root, write_thd_waveform_csv
from opamp_model.metrics import build_metrics_report, format_metrics_table, write_metrics_json
from opamp_model.ngspice_engine import NgspiceNotFoundError, simulate_thd_ngspice
from opamp_model.report import (
    preserve_metrics_sections,
    read_metrics_json,
    write_engine_report,
    write_thd_report,
)
from opamp_model.simulation_log import SimulationLog, archive_veriloga_artifacts, log_run_context
from opamp_model.spectre_engine import (
    SpectreLicenseError,
    SpectreNotFoundError,
    simulate_thd_spectre,
)
from opamp_model.noise_tran import transient_noise_rms
from opamp_model.tran import (
    compute_thd,
    is_thd_ideal,
    plot_thd_spectrum,
    plot_thd_waveform,
    simulate_sine_response,
)


def _cfg_for_thd(args: argparse.Namespace) -> OpampConfig:
    """Build config with default weak nonlinearity for THD when coeffs are unset."""
    cfg = build_opamp_config(args)
    if args.ideal:
        return OpampConfig(
            a0_db=cfg.a0_db,
            gbw_hz=cfg.gbw_hz,
            cmrr_db=cfg.cmrr_db,
            psrr_db=cfg.psrr_db,
            rin_ohm=cfg.rin_ohm,
            rout_ohm=cfg.rout_ohm,
            loop_beta=cfg.loop_beta,
            nl_a2=0.0,
            nl_a3=0.0,
            vswing_high_v=cfg.vswing_high_v,
            vswing_low_v=cfg.vswing_low_v,
            vcm_v=cfg.vcm_v,
        )
    if cfg.nl_a2 == 0.0 and cfg.nl_a3 == 0.0:
        return OpampConfig(
            a0_db=cfg.a0_db,
            gbw_hz=cfg.gbw_hz,
            cmrr_db=cfg.cmrr_db,
            psrr_db=cfg.psrr_db,
            rin_ohm=cfg.rin_ohm,
            rout_ohm=cfg.rout_ohm,
            loop_beta=cfg.loop_beta,
            nl_a2=1.0e5,
            nl_a3=1.0e7,
            vswing_high_v=cfg.vswing_high_v,
            vswing_low_v=cfg.vswing_low_v,
            vcm_v=cfg.vcm_v,
        )
    return cfg


def main() -> None:
    """Run THD simulation and write waveform CSV, spectrum SVG, and THD_REPORT.md."""
    parser = argparse.ArgumentParser(description="Op-amp THD / sine transient testbench.")
    add_opamp_args(parser)
    add_noise_args(parser)
    add_simulator_args(parser)
    add_output_args(parser)
    parser.add_argument(
        "--freq-hz",
        type=float,
        default=1.0e3,
        help="Sine frequency (Hz, default: 1 kHz).",
    )
    parser.add_argument(
        "--amplitude-v",
        type=float,
        default=1.0e-3,
        help="Peak input sine amplitude (V, default: 1 mV).",
    )
    parser.add_argument(
        "--cycles",
        type=float,
        default=20.0,
        help="Number of sine periods simulated (default: 20).",
    )
    parser.add_argument(
        "--dt-s",
        type=float,
        default=1.0e-6,
        help="Time step (s, default: 1 us).",
    )
    args = parser.parse_args()

    cfg = _cfg_for_thd(args)
    noise = build_noise_config(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = SimulationLog(out_dir / "logs" / "thd.log")
    log.write_header("thd", resolve_engine_label(args.simulator), cfg, noise)
    log_run_context(log)

    try:
        if args.simulator == "python":
            sine = simulate_sine_response(
                cfg,
                noise,
                amplitude_v=args.amplitude_v,
                freq_hz=args.freq_hz,
                cycles=args.cycles,
                dt_s=args.dt_s,
            )
            skip_thd = is_thd_ideal(cfg, ideal_flag=args.ideal)
            thd = None if skip_thd else compute_thd(
                sine["time_s"],
                sine["vout_v"],
                args.freq_hz,
            )
        elif args.simulator == "ngspice":
            sine, thd = simulate_thd_ngspice(
                cfg,
                out_dir,
                amplitude_v=args.amplitude_v,
                freq_hz=args.freq_hz,
                cycles=args.cycles,
                dt_s=args.dt_s,
                ideal_flag=args.ideal,
                noise=noise if noise.enabled else None,
            )
        elif args.simulator == "spectre":
            sine, thd = simulate_thd_spectre(
                cfg,
                out_dir,
                amplitude_v=args.amplitude_v,
                freq_hz=args.freq_hz,
                cycles=args.cycles,
                dt_s=args.dt_s,
                ideal_flag=args.ideal,
                noise=noise if noise.enabled else None,
            )
        else:
            msg = f"Unsupported simulator for THD: {args.simulator}"
            raise ValueError(msg)
    except (NgspiceNotFoundError, SpectreNotFoundError, SpectreLicenseError) as exc:
        log.write(str(exc))
        log.close()
        raise SystemExit(str(exc)) from exc

    time_s = sine["time_s"]
    vout_v = sine["vout_v"]
    noise_rms_v = transient_noise_rms(sine["noise_v"]) if noise.enabled else 0.0

    csv_path = out_dir / "thd_waveform.csv"
    write_thd_waveform_csv(csv_path, time_s, vout_v)

    waveform_svg = out_dir / "thd_waveform.svg"
    spectrum_svg = out_dir / "thd_spectrum.svg"
    plot_thd_waveform(
        time_s,
        vout_v,
        waveform_svg,
        freq_hz=args.freq_hz,
        vout_clean_v=sine["vout_clean_v"],
        noise_rms_v=noise_rms_v,
    )
    plot_thd_spectrum(
        time_s,
        vout_v,
        args.freq_hz,
        spectrum_svg,
        thd=thd,
        vout_clean_v=sine["vout_clean_v"],
    )

    report = build_metrics_report(
        cfg,
        noise,
        engine=args.simulator,
        thd=thd,
        ideal_flag=args.ideal,
    )
    report = preserve_metrics_sections(report, read_metrics_json(out_dir / "opamp_metrics.json"))
    write_metrics_json(out_dir / "opamp_metrics.json", report)

    thd_md = write_thd_report(
        out_dir,
        engine=args.simulator,
        cfg=cfg,
        noise=noise,
        report=report,
        waveform_svg=waveform_svg,
        spectrum_svg=spectrum_svg,
        thd=thd,
        freq_hz=args.freq_hz,
        amplitude_v=args.amplitude_v,
        noise_rms_v=noise_rms_v,
    )
    engine_md = write_engine_report(out_dir, engine=args.simulator)
    archive_veriloga_artifacts(package_root(), out_dir)

    log.write(f"wrote {csv_path}")
    log.write(f"wrote {waveform_svg}")
    log.write(f"wrote {spectrum_svg}")
    log.write(f"wrote {thd_md}")
    log.write(f"wrote {out_dir / 'opamp_metrics.json'}")
    if engine_md is not None:
        log.write(f"wrote {engine_md}")
    log.close()

    print(f"Wrote {csv_path}")
    print(f"Wrote {waveform_svg}")
    print(f"Wrote {spectrum_svg}")
    print(f"Wrote {thd_md}")
    print(f"Wrote {out_dir / 'opamp_metrics.json'}")
    print(format_metrics_table(report))


if __name__ == "__main__":
    main()
