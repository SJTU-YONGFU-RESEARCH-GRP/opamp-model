"""AC analysis, Bode metrics, and plotting."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from opamp_model.cm_ps import CmrrSimulationResult
from opamp_model.plot_style import (
    FIGSIZE,
    LINE_COLORS,
    LINEWIDTH_MAIN,
    LINEWIDTH_SECONDARY,
    apply_rcparams,
    apply_style,
)


class AcMetrics(TypedDict):
    """Scalar AC / STB summary metrics."""

    gbw_hz: float
    phase_margin_deg: float
    gain_margin_db: float
    a0_db: float


def _interp_crossing_frequency(
    frequency_hz: NDArray[np.float64],
    y_db: NDArray[np.float64],
    target_db: float,
) -> float:
    """Linearly interpolate the frequency where ``y_db`` crosses ``target_db``."""
    f = frequency_hz.astype(np.float64)
    y = y_db.astype(np.float64)
    for idx in range(len(y) - 1):
        y0, y1 = y[idx], y[idx + 1]
        if (y0 - target_db) * (y1 - target_db) <= 0.0 and y1 != y0:
            frac = (target_db - y0) / (y1 - y0)
            return float(f[idx] + frac * (f[idx + 1] - f[idx]))
    return float("nan")


def _interp_phase_at_frequency(
    frequency_hz: NDArray[np.float64],
    phase_deg: NDArray[np.float64],
    freq_hz: float,
) -> float:
    """Linearly interpolate phase (degrees) at ``freq_hz``."""
    if not np.isfinite(freq_hz):
        return float("nan")
    return float(np.interp(freq_hz, frequency_hz.astype(np.float64), phase_deg.astype(np.float64)))


def extract_gbw_phase_margin(
    frequency_hz: NDArray[np.float64],
    gain_db: NDArray[np.float64],
    phase_deg: NDArray[np.float64],
) -> AcMetrics:
    """Extract GBW, phase margin, gain margin, and low-frequency gain.

    Phase margin uses ``PM = 180 + phase(loop)`` at the 0 dB gain crossover.
    Gain margin is the gain (dB) when phase crosses ``-180`` degrees.
    """
    a0_db = float(gain_db[0]) if len(gain_db) else float("nan")
    gbw_hz = _interp_crossing_frequency(frequency_hz, gain_db, 0.0)
    phase_at_gbw = _interp_phase_at_frequency(frequency_hz, phase_deg, gbw_hz)
    phase_margin_deg = 180.0 + phase_at_gbw

    phase_cross_hz = _interp_crossing_frequency(frequency_hz, phase_deg, -180.0)
    gain_at_phase_cross = _interp_phase_at_frequency(frequency_hz, gain_db, phase_cross_hz)
    gain_margin_db = gain_at_phase_cross

    return AcMetrics(
        gbw_hz=gbw_hz,
        phase_margin_deg=phase_margin_deg,
        gain_margin_db=gain_margin_db,
        a0_db=a0_db,
    )


def extract_cmrr(
    frequency_hz: NDArray[np.float64],
    gain_db: NDArray[np.float64],
    *,
    cmrr_linear: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Derive ACM and CMRR curves from open-loop gain samples.

    Uses ``ACM = Aol / CMRR_linear`` with constant ``CMRR_linear``.
    Returns ``(acm_db, cmrr_db, cmrr_dc_db)``.
    """
    cmrr_db_const = 20.0 * np.log10(max(cmrr_linear, 1.0e-30))
    acm_db = (gain_db - cmrr_db_const).astype(np.float64)
    cmrr_db = (gain_db - acm_db).astype(np.float64)
    _ = frequency_hz
    return acm_db, cmrr_db, float(cmrr_db[0]) if len(cmrr_db) else float("nan")


def cmrr_from_open_loop(
    frequency_hz: NDArray[np.float64],
    gain_db: NDArray[np.float64],
    *,
    cmrr_linear: float,
) -> CmrrSimulationResult:
    """Build a CMRR result bundle from an existing open-loop Bode sweep."""
    acm_db, cmrr_db, cmrr_dc_db = extract_cmrr(
        frequency_hz,
        gain_db,
        cmrr_linear=cmrr_linear,
    )
    return CmrrSimulationResult(
        frequency_hz=frequency_hz.astype(np.float64),
        acm_db=acm_db,
        cmrr_db=cmrr_db,
        cmrr_dc_db=cmrr_dc_db,
    )


def plot_cmrr(
    cmrr_result: CmrrSimulationResult,
    output_path: Path,
    *,
    title: str = "Common-mode rejection",
) -> Path:
    """Plot ACM and CMRR vs frequency and write an SVG."""
    apply_rcparams()
    f = cmrr_result["frequency_hz"]
    fig, (ax_acm, ax_cmrr) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    ax_acm.semilogx(
        f,
        cmrr_result["acm_db"],
        color=LINE_COLORS["gain"],
        linewidth=LINEWIDTH_MAIN,
    )
    ax_acm.set_ylabel("ACM (dB)", fontsize=13)
    ax_acm.set_title(
        f"{title}\nCMRR(DC)={cmrr_result['cmrr_dc_db']:.1f} dB",
        fontsize=14,
    )

    ax_cmrr.semilogx(
        f,
        cmrr_result["cmrr_db"],
        color=LINE_COLORS["cmrr"],
        linewidth=LINEWIDTH_SECONDARY,
    )
    ax_cmrr.set_ylabel("CMRR (dB)", fontsize=13)
    ax_cmrr.set_xlabel("Frequency (Hz)", fontsize=13)

    apply_style(ax_acm)
    apply_style(ax_cmrr)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def plot_bode(
    frequency_hz: NDArray[np.float64],
    gain_db: NDArray[np.float64],
    phase_deg: NDArray[np.float64],
    output_path: Path,
    *,
    title: str = "Open-loop Bode",
    metrics: AcMetrics | None = None,
) -> Path:
    """Write a two-panel Bode SVG."""
    apply_rcparams()
    fig, (ax_gain, ax_phase) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    ax_gain.semilogx(frequency_hz, gain_db, color=LINE_COLORS["gain"], linewidth=LINEWIDTH_MAIN)
    ax_gain.set_ylabel("Gain (dB)", fontsize=13)
    ax_gain.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")

    ax_phase.semilogx(
        frequency_hz,
        phase_deg,
        color=LINE_COLORS["phase"],
        linewidth=LINEWIDTH_SECONDARY,
    )
    ax_phase.set_ylabel("Phase (deg)", fontsize=13)
    ax_phase.set_xlabel("Frequency (Hz)", fontsize=13)
    ax_phase.axhline(-180.0, color="#888888", linewidth=1.0, linestyle="--")

    if metrics is not None and np.isfinite(metrics["gbw_hz"]):
        ax_gain.axvline(metrics["gbw_hz"], color="#888888", linewidth=1.0, linestyle=":")
        subtitle = (
            f"GBW={metrics['gbw_hz']:.3g} Hz  "
            f"PM={metrics['phase_margin_deg']:.1f}°  "
            f"A0={metrics['a0_db']:.1f} dB"
        )
        ax_gain.set_title(f"{title}\n{subtitle}", fontsize=14)
    else:
        ax_gain.set_title(title, fontsize=14)

    apply_style(ax_gain)
    apply_style(ax_phase)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path
