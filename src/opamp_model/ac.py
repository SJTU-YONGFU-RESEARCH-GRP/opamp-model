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


class AcCompMetrics(TypedDict):
    """Gain-peaking summary near the open-loop unity-gain frequency."""

    peak_db: float
    peak_freq_hz: float
    gbw_hz: float
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


def extract_gain_peaking_near_gbw(
    frequency_hz: NDArray[np.float64],
    gain_db: NDArray[np.float64],
    phase_deg: NDArray[np.float64],
    *,
    window_low_factor: float = 0.1,
    window_high_factor: float = 10.0,
) -> AcCompMetrics:
    """Extract gain peaking (dB) in a band around the 0 dB crossover.

    ``peak_db`` is the maximum excess of measured open-loop gain over an ideal
    single-pole rolloff anchored at ``a0_db`` and the dominant pole
    ``f_p = gbw_hz / a0_linear``. Phase samples are accepted for API symmetry
    with :func:`extract_gbw_phase_margin` but are not used in the calculation.
    """
    _ = phase_deg
    ac = extract_gbw_phase_margin(frequency_hz, gain_db, phase_deg)
    gbw_hz = ac["gbw_hz"]
    a0_db = ac["a0_db"]
    nan = float("nan")
    if not np.isfinite(gbw_hz) or gbw_hz <= 0.0 or not np.isfinite(a0_db):
        return AcCompMetrics(
            peak_db=nan,
            peak_freq_hz=nan,
            gbw_hz=gbw_hz,
            a0_db=a0_db,
        )

    f = frequency_hz.astype(np.float64)
    g = gain_db.astype(np.float64)
    a0_linear = 10.0 ** (a0_db / 20.0)
    fp_hz = gbw_hz / max(a0_linear, 1.0)
    f_low = max(float(f[0]), window_low_factor * gbw_hz, fp_hz)
    f_high = min(float(f[-1]), window_high_factor * gbw_hz)
    if f_high <= f_low:
        return AcCompMetrics(
            peak_db=nan,
            peak_freq_hz=nan,
            gbw_hz=gbw_hz,
            a0_db=a0_db,
        )

    mask = (f >= f_low) & (f <= f_high)
    if not np.any(mask):
        return AcCompMetrics(
            peak_db=nan,
            peak_freq_hz=nan,
            gbw_hz=gbw_hz,
            a0_db=a0_db,
        )

    f_band = f[mask]
    g_band = g[mask]
    ideal_db = a0_db - 20.0 * np.log10(np.maximum(f_band / fp_hz, 1.0e-30))
    excess_db = g_band - ideal_db
    peak_idx = int(np.argmax(excess_db))
    return AcCompMetrics(
        peak_db=float(excess_db[peak_idx]),
        peak_freq_hz=float(f_band[peak_idx]),
        gbw_hz=gbw_hz,
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


def plot_ac_comp(
    frequency_hz: NDArray[np.float64],
    gain_db: NDArray[np.float64],
    phase_deg: NDArray[np.float64],
    output_path: Path,
    *,
    title: str = "Open-loop gain peaking",
    comp_metrics: AcCompMetrics | None = None,
) -> Path:
    """Plot open-loop Bode with GBW / peaking markers and write an SVG."""
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

    if comp_metrics is not None:
        gbw = comp_metrics["gbw_hz"]
        peak_f = comp_metrics["peak_freq_hz"]
        peak_db = comp_metrics["peak_db"]
        if np.isfinite(gbw) and gbw > 0.0:
            ax_gain.axvline(gbw, color="#888888", linewidth=1.0, linestyle=":")
        if np.isfinite(peak_f) and peak_f > 0.0:
            ax_gain.axvline(peak_f, color=LINE_COLORS["phase"], linewidth=1.0, linestyle="--")
        subtitle_parts = []
        if np.isfinite(gbw):
            subtitle_parts.append(f"GBW={gbw:.3g} Hz")
        if np.isfinite(peak_db):
            subtitle_parts.append(f"peak={peak_db:.2f} dB")
        if np.isfinite(peak_f):
            subtitle_parts.append(f"@ {peak_f:.3g} Hz")
        if subtitle_parts:
            ax_gain.set_title(f"{title}\n{'  '.join(subtitle_parts)}", fontsize=14)
        else:
            ax_gain.set_title(title, fontsize=14)
    else:
        ax_gain.set_title(title, fontsize=14)

    apply_style(ax_gain)
    apply_style(ax_phase)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path
