"""Transconductance (Gm / OTA) macromodel and AC bench helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from opamp_model.ac import _interp_crossing_frequency
from opamp_model.config import BenchSweepConfig, GmConfig, OpampNoiseConfig
from opamp_model.io import log_frequency_sweep
from opamp_model.noise import input_referred_en
from opamp_model.plot_style import (
    FIGSIZE,
    LINE_COLORS,
    LINEWIDTH_MAIN,
    LINEWIDTH_SECONDARY,
    apply_rcparams,
    apply_style,
)


class GmMetrics(TypedDict):
    """Scalar Gm AC summary metrics."""

    gm_s: float
    rout_ohm: float
    gain_db: float
    bandwidth_hz: float


class GmAcSimulationResult(TypedDict):
    """Gm AC simulation result bundle."""

    frequency_hz: NDArray[np.float64]
    gain_db: NDArray[np.float64]
    phase_deg: NDArray[np.float64]
    gm_s: NDArray[np.float64]
    metrics: GmMetrics


def zout_gm(
    gm_cfg: GmConfig,
    frequency_hz: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Return output impedance vs frequency (Rout || Cout)."""
    omega = 2.0 * np.pi * frequency_hz.astype(np.float64)
    cap = 1.0 / (1j * omega * gm_cfg.cout_f) if gm_cfg.cout_f > 0.0 else np.inf
    return 1.0 / (1.0 / gm_cfg.rout_ohm + 1.0 / cap)


def transconductance_transfer(
    gm_cfg: GmConfig,
    frequency_hz: NDArray[np.float64] | float,
    noise: OpampNoiseConfig | None = None,
) -> NDArray[np.complex128]:
    """Return loaded voltage transfer V_out / V_diff.

    Ideal VCCS ``I_out = gm * (Vin+ - Vin-)`` with output shunt ``Rout || Cout``:

    ``H(s) = gm * Z_out(s)``.
    """
    _ = noise
    f = np.atleast_1d(np.asarray(frequency_hz, dtype=np.float64))
    return np.atleast_1d(
        np.asarray(gm_cfg.gm_s * zout_gm(gm_cfg, f), dtype=np.complex128)
    )


def gm_vs_frequency(
    gm_cfg: GmConfig,
    frequency_hz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return effective transconductance magnitude (S) vs frequency.

    For the passive-loaded macromodel, ``|gm_eff| = |H| / |Z_out|`` equals ``gm_s``.
    """
    return np.full(frequency_hz.shape, gm_cfg.gm_s, dtype=np.float64)


def extract_gm_metrics(
    gm_cfg: GmConfig,
    frequency_hz: NDArray[np.float64],
    gain_db: NDArray[np.float64],
) -> GmMetrics:
    """Extract DC gain, Rout, and −3 dB bandwidth from the loaded Bode sweep."""
    gain_db_arr = gain_db.astype(np.float64)
    gain_dc_db = float(gain_db_arr[0]) if len(gain_db_arr) else float("nan")
    target_db = gain_dc_db - 3.0
    bandwidth_hz = _interp_crossing_frequency(frequency_hz, gain_db_arr, target_db)
    return GmMetrics(
        gm_s=gm_cfg.gm_s,
        rout_ohm=gm_cfg.rout_ohm,
        gain_db=gain_dc_db,
        bandwidth_hz=bandwidth_hz,
    )


def gm_frequency_grid(
    sweep: BenchSweepConfig | None = None,
) -> NDArray[np.float64]:
    """Return the log-spaced frequency grid for Gm AC benches."""
    cfg = sweep or BenchSweepConfig()
    return log_frequency_sweep(cfg.f_start_hz, cfg.f_stop_hz, cfg.points_per_decade)


def simulate_gm_ac(
    gm_cfg: GmConfig,
    noise: OpampNoiseConfig | None = None,
    *,
    sweep: BenchSweepConfig | None = None,
) -> GmAcSimulationResult:
    """Run Gm AC simulation (loaded V_out/V_diff Bode and gm vs f)."""
    frequency_hz = gm_frequency_grid(sweep)
    h = transconductance_transfer(gm_cfg, frequency_hz, noise)
    gain_db = (20.0 * np.log10(np.maximum(np.abs(h), 1.0e-30))).astype(np.float64)
    phase_deg = np.degrees(np.angle(h)).astype(np.float64)
    gm_s = gm_vs_frequency(gm_cfg, frequency_hz)
    metrics = extract_gm_metrics(gm_cfg, frequency_hz, gain_db)
    return GmAcSimulationResult(
        frequency_hz=frequency_hz,
        gain_db=gain_db,
        phase_deg=phase_deg,
        gm_s=gm_s,
        metrics=metrics,
    )


def output_referred_noise_density(
    gm_cfg: GmConfig,
    noise: OpampNoiseConfig,
    frequency_hz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return output-referred voltage noise density (V/√Hz) vs frequency."""
    en = input_referred_en(frequency_hz, noise)
    h = transconductance_transfer(gm_cfg, frequency_hz, noise)
    return (en * np.abs(h) / np.maximum(gm_cfg.gm_s, 1.0e-30)).astype(np.float64)


def write_gm_ac_csv(
    path: Path,
    frequency_hz: NDArray[np.float64],
    gain_db: NDArray[np.float64],
    phase_deg: NDArray[np.float64],
    gm_s: NDArray[np.float64],
) -> None:
    """Write Gm AC CSV (frequency, gain_db, phase_deg, gm_s)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([frequency_hz, gain_db, phase_deg, gm_s])
    header = "frequency_hz,gain_db,phase_deg,gm_s"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def plot_gm_bode(
    result: GmAcSimulationResult,
    output_path: Path,
    *,
    title: str = "Gm loaded transfer (Vout/Vdiff)",
) -> Path:
    """Plot gain and phase vs frequency and write an SVG."""
    apply_rcparams()
    metrics = result["metrics"]
    fig, (ax_gain, ax_phase) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    ax_gain.semilogx(
        result["frequency_hz"],
        result["gain_db"],
        color=LINE_COLORS["gain"],
        linewidth=LINEWIDTH_MAIN,
    )
    ax_gain.set_ylabel("Gain (dB)", fontsize=13)
    ax_gain.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")

    ax_phase.semilogx(
        result["frequency_hz"],
        result["phase_deg"],
        color=LINE_COLORS["phase"],
        linewidth=LINEWIDTH_SECONDARY,
    )
    ax_phase.set_ylabel("Phase (deg)", fontsize=13)
    ax_phase.set_xlabel("Frequency (Hz)", fontsize=13)
    ax_phase.axhline(-90.0, color="#888888", linewidth=1.0, linestyle="--")

    subtitle = (
        f"gm={metrics['gm_s']:.3g} S  "
        f"Rout={metrics['rout_ohm']:.3g} Ω  "
        f"Gain(DC)={metrics['gain_db']:.1f} dB"
    )
    if np.isfinite(metrics["bandwidth_hz"]):
        subtitle += f"  BW−3dB={metrics['bandwidth_hz']:.3g} Hz"
    ax_gain.set_title(f"{title}\n{subtitle}", fontsize=14)

    apply_style(ax_gain)
    apply_style(ax_phase)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def plot_gm_vs_f(
    result: GmAcSimulationResult,
    output_path: Path,
    *,
    title: str = "Transconductance vs frequency",
) -> Path:
    """Plot effective gm (S) vs frequency and write an SVG."""
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogx(
        result["frequency_hz"],
        result["gm_s"] * 1.0e3,
        color=LINE_COLORS["gain"],
        linewidth=LINEWIDTH_MAIN,
    )
    ax.set_ylabel("gm (mS)", fontsize=13)
    ax.set_xlabel("Frequency (Hz)", fontsize=13)
    ax.set_title(
        f"{title}\nDC gm = {result['metrics']['gm_s'] * 1.0e3:.3g} mS",
        fontsize=14,
    )
    apply_style(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def write_gm_ac_report(
    output_dir: Path,
    *,
    gm_cfg: GmConfig,
    result: GmAcSimulationResult,
    bode_svg: Path,
    gm_svg: Path,
) -> Path:
    """Write ``GM_AC_REPORT.md`` with figures and scalar metrics."""
    path = output_dir / "GM_AC_REPORT.md"
    metrics = result["metrics"]
    lines = [
        "# Gm AC bench",
        "",
        "- **Engine:** Python behavioral model (`python`)",
        "",
        "## Macromodel",
        "",
        "| Parameter | Value | Unit |",
        "| --- | --- | --- |",
        f"| gm | {gm_cfg.gm_s:.6g} | S |",
        f"| Rout | {gm_cfg.rout_ohm:.6g} | Ω |",
        f"| Cout | {gm_cfg.cout_f:.6g} | F |",
        "",
        "Loaded transfer: ``V_out / V_diff = gm · (Rout || Cout)``.",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Unit |",
        "| --- | --- | --- |",
        f"| gm | {metrics['gm_s']:.6g} | S |",
        f"| Rout | {metrics['rout_ohm']:.6g} | Ω |",
        f"| Gain (DC) | {metrics['gain_db']:.3g} | dB |",
        f"| Bandwidth (−3 dB) | {metrics['bandwidth_hz']:.6g} | Hz |",
        "",
        "## Figures",
        "",
        f"![Loaded Bode]({bode_svg.name})",
        "",
        f"![gm vs f]({gm_svg.name})",
        "",
        "## Artifacts",
        "",
        "| File | Description |",
        "| --- | --- |",
        "| `gm_ac_bode.csv` | frequency, gain_db, phase_deg, gm_s |",
        f"| `{bode_svg.name}` | Loaded transfer Bode |",
        f"| `{gm_svg.name}` | gm vs frequency |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
