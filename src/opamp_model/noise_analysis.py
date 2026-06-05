"""Integrated and spot noise analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.core import open_loop_transfer
from opamp_model.noise import flicker_corner_frequency, flicker_voltage_density, input_referred_en
from opamp_model.plot_style import (
    FIGSIZE,
    LABEL_SIZE,
    LINE_COLORS,
    LINEWIDTH_MAIN,
    LINEWIDTH_SECONDARY,
    TITLE_SIZE,
    apply_rcparams,
    apply_style,
    downsample_stride,
)


class NoiseMetrics(TypedDict):
    """Scalar noise summary metrics."""

    integrated_noise_rms_v: float
    en_out_spot_1khz_v_per_sqrt_hz: float
    en_in_spot_1khz_v_per_sqrt_hz: float


class NoiseBreakdown(TypedDict):
    """Input- and output-referred noise density components vs frequency."""

    frequency_hz: NDArray[np.float64]
    en_in_white_v_per_sqrt_hz: NDArray[np.float64]
    en_in_flicker_v_per_sqrt_hz: NDArray[np.float64]
    en_in_total_v_per_sqrt_hz: NDArray[np.float64]
    en_out_white_v_per_sqrt_hz: NDArray[np.float64]
    en_out_flicker_v_per_sqrt_hz: NDArray[np.float64]
    en_out_total_v_per_sqrt_hz: NDArray[np.float64]
    flicker_corner_hz: float


def integrate_noise_rms(
    frequency_hz: NDArray[np.float64],
    density_v_per_sqrt_hz: NDArray[np.float64],
) -> float:
    """Integrate one-sided noise density to RMS (V) via trapezoidal rule."""
    f = frequency_hz.astype(np.float64)
    d = density_v_per_sqrt_hz.astype(np.float64)
    if len(f) < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        power = float(np.trapezoid(d**2, f))
    else:
        power = float(np.trapz(d**2, f))  # type: ignore[attr-defined]
    return float(np.sqrt(max(power, 0.0)))


def spot_noise_at_frequency(
    frequency_hz: NDArray[np.float64],
    density_v_per_sqrt_hz: NDArray[np.float64],
    spot_hz: float,
) -> float:
    """Interpolate noise density (V/√Hz) at ``spot_hz`` on a log-spaced grid."""
    f = np.asarray(frequency_hz, dtype=np.float64).ravel()
    d = np.asarray(density_v_per_sqrt_hz, dtype=np.float64).ravel()
    if f.size == 0:
        return float("nan")
    if f.size == 1:
        return float(d[0])
    log_f = np.log10(f)
    log_spot = np.log10(max(spot_hz, float(f[0])))
    return float(np.interp(log_spot, log_f, d))


def output_noise_spectrum(
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
    frequency_hz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return output-referred noise density (V/√Hz) scaled by |A_open(f)|."""
    if not noise.enabled:
        return np.zeros_like(frequency_hz, dtype=np.float64)
    en_in = input_referred_en(frequency_hz, noise)
    a_mag = np.abs(open_loop_transfer(cfg, frequency_hz))
    return (en_in * a_mag).astype(np.float64)


def integrated_output_noise_rms(
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
    frequency_hz: NDArray[np.float64],
    spectrum_v_per_sqrt_hz: NDArray[np.float64],
) -> float:
    """Integrate output-referred noise over the analysis band (cfg.sweep limits)."""
    f = frequency_hz.astype(np.float64)
    mask = (f >= cfg.sweep.f_start_hz) & (f <= cfg.sweep.f_stop_hz)
    if not np.any(mask):
        return 0.0
    return integrate_noise_rms(f[mask], spectrum_v_per_sqrt_hz[mask])


def extract_noise_metrics(
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
    frequency_hz: NDArray[np.float64],
    spectrum_v_per_sqrt_hz: NDArray[np.float64],
    *,
    spot_hz: float = 1.0e3,
) -> NoiseMetrics:
    """Extract integrated RMS and spot densities at ``spot_hz``."""
    if not noise.enabled:
        return NoiseMetrics(
            integrated_noise_rms_v=0.0,
            en_out_spot_1khz_v_per_sqrt_hz=0.0,
            en_in_spot_1khz_v_per_sqrt_hz=0.0,
        )
    en_in = input_referred_en(frequency_hz, noise)
    return NoiseMetrics(
        integrated_noise_rms_v=integrated_output_noise_rms(
            cfg, noise, frequency_hz, spectrum_v_per_sqrt_hz
        ),
        en_out_spot_1khz_v_per_sqrt_hz=spot_noise_at_frequency(
            frequency_hz, spectrum_v_per_sqrt_hz, spot_hz
        ),
        en_in_spot_1khz_v_per_sqrt_hz=spot_noise_at_frequency(
            frequency_hz, en_in, spot_hz
        ),
    )


def compute_noise_breakdown(
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
    frequency_hz: NDArray[np.float64],
) -> NoiseBreakdown:
    """Split white and flicker contributions at input and output."""
    f = np.asarray(frequency_hz, dtype=np.float64)
    if not noise.enabled:
        zeros = np.zeros_like(f)
        return NoiseBreakdown(
            frequency_hz=f,
            en_in_white_v_per_sqrt_hz=zeros,
            en_in_flicker_v_per_sqrt_hz=zeros,
            en_in_total_v_per_sqrt_hz=zeros,
            en_out_white_v_per_sqrt_hz=zeros,
            en_out_flicker_v_per_sqrt_hz=zeros,
            en_out_total_v_per_sqrt_hz=zeros,
            flicker_corner_hz=float("nan"),
        )

    en_in_white = np.full_like(f, noise.en_white_v_per_sqrt_hz)
    en_in_flicker = flicker_voltage_density(f, noise)
    en_in_total = input_referred_en(f, noise)
    a_mag = np.abs(open_loop_transfer(cfg, f))
    return NoiseBreakdown(
        frequency_hz=f,
        en_in_white_v_per_sqrt_hz=en_in_white.astype(np.float64),
        en_in_flicker_v_per_sqrt_hz=en_in_flicker.astype(np.float64),
        en_in_total_v_per_sqrt_hz=en_in_total.astype(np.float64),
        en_out_white_v_per_sqrt_hz=(en_in_white * a_mag).astype(np.float64),
        en_out_flicker_v_per_sqrt_hz=(en_in_flicker * a_mag).astype(np.float64),
        en_out_total_v_per_sqrt_hz=(en_in_total * a_mag).astype(np.float64),
        flicker_corner_hz=float(flicker_corner_frequency(noise)),
    )


def plot_noise_breakdown(
    breakdown: NoiseBreakdown,
    output_path: Path,
    *,
    title: str = "Noise density breakdown",
) -> Path:
    """Plot input- and output-referred white/flicker/total spectra (SVG)."""
    apply_rcparams()
    f = breakdown["frequency_hz"]
    stride = downsample_stride(len(f))
    f_plot = f[::stride]
    corner = breakdown["flicker_corner_hz"]
    scale = 1.0e9

    fig, (ax_in, ax_out) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)
    for ax, white, flicker, total, ylabel in (
        (
            ax_in,
            breakdown["en_in_white_v_per_sqrt_hz"],
            breakdown["en_in_flicker_v_per_sqrt_hz"],
            breakdown["en_in_total_v_per_sqrt_hz"],
            "Input-referred (nV/√Hz)",
        ),
        (
            ax_out,
            breakdown["en_out_white_v_per_sqrt_hz"],
            breakdown["en_out_flicker_v_per_sqrt_hz"],
            breakdown["en_out_total_v_per_sqrt_hz"],
            "Output-referred (nV/√Hz)",
        ),
    ):
        ax.loglog(
            f_plot,
            white[::stride] * scale,
            color=LINE_COLORS["gain"],
            linewidth=LINEWIDTH_MAIN,
            label="White",
        )
        ax.loglog(
            f_plot,
            flicker[::stride] * scale,
            color=LINE_COLORS["cmrr"],
            linewidth=LINEWIDTH_SECONDARY,
            label="Flicker",
        )
        ax.loglog(
            f_plot,
            total[::stride] * scale,
            color=LINE_COLORS["psrr"],
            linewidth=LINEWIDTH_SECONDARY,
            linestyle="--",
            label="Total (RSS)",
        )
        if np.isfinite(corner) and corner > 0.0:
            ax.axvline(
                corner,
                color=LINE_COLORS["phase"],
                linewidth=1.5,
                linestyle=":",
                label="Flicker corner",
            )
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.legend(fontsize=9, loc="best")
        apply_style(ax, grid_axis="both")

    ax_out.set_xlabel("Frequency (Hz)", fontsize=LABEL_SIZE)
    corner_str = f"{corner:.4g} Hz" if np.isfinite(corner) else "—"
    fig.suptitle(f"{title}\nFlicker corner (white = flicker): {corner_str}", fontsize=TITLE_SIZE)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def plot_noise_spectrum(
    frequency_hz: NDArray[np.float64],
    spectrum_v_per_sqrt_hz: NDArray[np.float64],
    output_path: Path,
    *,
    title: str = "Output noise spectrum",
    metrics: NoiseMetrics | None = None,
) -> Path:
    """Plot output-referred noise density vs frequency and write an SVG."""
    apply_rcparams()
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    stride = downsample_stride(len(frequency_hz))
    f_plot = frequency_hz[::stride]
    n_plot = spectrum_v_per_sqrt_hz[::stride] * 1.0e9
    ax.loglog(
        f_plot,
        n_plot,
        color=LINE_COLORS["noise"],
        linewidth=LINEWIDTH_MAIN,
    )
    ax.set_xlabel("Frequency (Hz)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Output noise (nV/√Hz)", fontsize=LABEL_SIZE)
    if metrics is not None and np.isfinite(metrics["integrated_noise_rms_v"]):
        subtitle = (
            f"∫ RMS={metrics['integrated_noise_rms_v']:.3g} V  "
            f"@1 kHz={metrics['en_out_spot_1khz_v_per_sqrt_hz'] * 1.0e9:.3g} nV/√Hz"
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=TITLE_SIZE)
    else:
        ax.set_title(title, fontsize=TITLE_SIZE)
    apply_style(ax, grid_axis="both")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path
