"""Integrated and spot noise analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.core import open_loop_transfer
from opamp_model.noise import input_referred_en
from opamp_model.plot_style import (
    FIGSIZE,
    LABEL_SIZE,
    LINE_COLORS,
    LINEWIDTH_MAIN,
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


def integrate_noise_rms(
    frequency_hz: NDArray[np.float64],
    density_v_per_sqrt_hz: NDArray[np.float64],
) -> float:
    """Integrate one-sided noise density to RMS (V) via trapezoidal rule."""
    f = frequency_hz.astype(np.float64)
    d = density_v_per_sqrt_hz.astype(np.float64)
    if len(f) < 2:
        return 0.0
    power = float(np.trapezoid(d**2, f))
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
