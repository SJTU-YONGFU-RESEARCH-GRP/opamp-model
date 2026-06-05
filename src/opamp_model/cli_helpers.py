"""Shared CLI helpers for opamp-model testbench scripts."""

from __future__ import annotations

import argparse

from opamp_model.config import (
    GmConfig,
    OpampConfig,
    OpampNoiseConfig,
    TiaConfig,
)


def add_opamp_args(parser: argparse.ArgumentParser) -> None:
    """Register core op-amp macromodel arguments."""
    parser.add_argument("--a0-db", type=float, default=80.0, help="DC open-loop gain (dB).")
    parser.add_argument("--gbw-hz", type=float, default=10.0e6, help="Gain-bandwidth (Hz).")
    parser.add_argument("--cmrr-db", type=float, default=90.0, help="CMRR (dB).")
    parser.add_argument("--psrr-db", type=float, default=80.0, help="PSRR at DC (dB).")
    parser.add_argument(
        "--psrr-pole-hz",
        type=float,
        default=100.0,
        help="PSRR feedthrough pole (Hz).",
    )
    parser.add_argument("--rin-ohm", type=float, default=1.0e12, help="Differential input R (ohm).")
    parser.add_argument("--rout-ohm", type=float, default=100.0, help="Output R (ohm).")
    parser.add_argument(
        "--loop-beta",
        type=float,
        default=1.0,
        help="Feedback factor for STB loop gain (default: unity).",
    )


def add_noise_args(parser: argparse.ArgumentParser) -> None:
    """Register noise and nonlinearity arguments."""
    parser.add_argument(
        "--ideal",
        action="store_true",
        help="Disable noise and weak nonlinearity (linear small-signal only).",
    )
    parser.add_argument(
        "--en-white-nv-per-sqrt-hz",
        type=float,
        default=5.0,
        dest="en_white_nv",
        help="Input-referred white noise (nV/sqrt(Hz)).",
    )
    parser.add_argument(
        "--en-flicker-1hz-nv-per-sqrt-hz",
        type=float,
        default=None,
        dest="en_flicker_1hz_nv",
        help="Input-referred flicker density at 1 Hz (nV/sqrt(Hz)).",
    )
    parser.add_argument(
        "--en-flicker-ef",
        type=float,
        default=1.0,
        help="Flicker exponent EF (density scales as 1/f^(EF/2)).",
    )
    parser.add_argument(
        "--en-flicker-corner-hz",
        type=float,
        default=None,
        dest="en_flicker_corner_hz",
        help="Legacy: derive flicker @1 Hz from white and corner when 1 Hz not set.",
    )
    parser.add_argument(
        "--kf",
        type=float,
        default=0.0,
        help="Coram flicker coefficient (power at 1 Hz = KF * |I|^AF when > 0).",
    )
    parser.add_argument(
        "--af",
        type=float,
        default=1.0,
        help="Coram flicker current exponent AF.",
    )
    parser.add_argument(
        "--bias-current-a",
        type=float,
        default=0.0,
        help="Bias current (A) for Coram flicker model.",
    )
    parser.add_argument(
        "--nl-a2",
        type=float,
        default=0.0,
        help="Second-order nonlinearity coefficient.",
    )
    parser.add_argument("--nl-a3", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=1)


def add_simulator_args(parser: argparse.ArgumentParser) -> None:
    """Register mutually exclusive simulator selection arguments."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--simulator",
        choices=("python", "spectre", "ngspice"),
        default="python",
        help="Simulation engine (default: python).",
    )
    group.add_argument(
        "--spectre",
        action="store_const",
        const="spectre",
        dest="simulator",
        help="Use Cadence Spectre.",
    )
    group.add_argument(
        "--ngspice",
        action="store_const",
        const="ngspice",
        dest="simulator",
        help="Use ngspice netlists.",
    )


def add_output_args(parser: argparse.ArgumentParser) -> None:
    """Register common output directory argument."""
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/python",
        help="Directory for CSV, SVG, and logs.",
    )


def resolve_engine_label(simulator: str) -> str:
    """Return a human-readable simulator label."""
    labels = {
        "python": "Python behavioral model",
        "spectre": "Cadence Spectre",
        "ngspice": "ngspice netlist",
    }
    return labels.get(simulator, simulator)


def build_opamp_config(args: argparse.Namespace) -> OpampConfig:
    """Build ``OpampConfig`` from parsed CLI arguments."""
    nl_a2 = 0.0 if args.ideal else args.nl_a2
    nl_a3 = 0.0 if args.ideal else args.nl_a3
    return OpampConfig(
        a0_db=args.a0_db,
        gbw_hz=args.gbw_hz,
        cmrr_db=args.cmrr_db,
        psrr_db=args.psrr_db,
        psrr_pole_hz=args.psrr_pole_hz,
        rin_ohm=args.rin_ohm,
        rout_ohm=args.rout_ohm,
        loop_beta=args.loop_beta,
        nl_a2=nl_a2,
        nl_a3=nl_a3,
    )


def build_noise_config(args: argparse.Namespace) -> OpampNoiseConfig:
    """Build ``OpampNoiseConfig`` from parsed CLI arguments."""
    if args.ideal:
        return OpampNoiseConfig(
            en_white_v_per_sqrt_hz=0.0,
            en_flicker_1hz_v_per_sqrt_hz=0.0,
            in_white_a_per_sqrt_hz=0.0,
            in_flicker_1hz_a_per_sqrt_hz=0.0,
            noise_seed=args.noise_seed,
        )
    kwargs: dict[str, float | int] = {
        "en_white_v_per_sqrt_hz": args.en_white_nv * 1.0e-9,
        "en_flicker_ef": args.en_flicker_ef,
        "kf": args.kf,
        "af": args.af,
        "bias_current_a": args.bias_current_a,
        "noise_seed": args.noise_seed,
    }
    if args.en_flicker_1hz_nv is not None:
        kwargs["en_flicker_1hz_v_per_sqrt_hz"] = args.en_flicker_1hz_nv * 1.0e-9
    if args.en_flicker_corner_hz is not None:
        kwargs["en_flicker_corner_hz"] = args.en_flicker_corner_hz
    return OpampNoiseConfig(**kwargs)


def add_tia_args(parser: argparse.ArgumentParser) -> None:
    """Register TIA feedback network arguments."""
    parser.add_argument("--rf-ohm", type=float, default=100.0e3, help="Feedback R (ohm).")
    parser.add_argument("--cf-f", type=float, default=100.0e-15, help="Feedback C (F).")
    parser.add_argument("--cs-f", type=float, default=0.0, help="Sensor/input shunt C (F).")
    parser.add_argument(
        "--voltage-input",
        action="store_true",
        help="Voltage-input TIA (Vout/Vin) instead of current input.",
    )


def build_tia_config(args: argparse.Namespace) -> TiaConfig:
    """Build ``TiaConfig`` when TIA-specific args are present."""
    return TiaConfig(
        rf_ohm=args.rf_ohm,
        cf_f=args.cf_f,
        cs_f=args.cs_f,
        current_input=not getattr(args, "voltage_input", False),
    )


def build_gm_config(args: argparse.Namespace) -> GmConfig:
    """Build ``GmConfig`` when Gm-specific args are present."""
    return GmConfig(
        gm_s=getattr(args, "gm_s", 1.0e-3),
        rout_ohm=getattr(args, "gm_rout_ohm", 1.0e6),
        cout_f=getattr(args, "gm_cout_f", 500.0e-15),
    )

