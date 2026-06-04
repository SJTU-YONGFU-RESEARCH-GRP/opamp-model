"""Input and output impedance helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from opamp_model.config import OpampConfig
from opamp_model.io import log_frequency_sweep
from opamp_model.plot_style import (
    FIGSIZE,
    LINE_COLORS,
    LINEWIDTH_MAIN,
    LINEWIDTH_SECONDARY,
    apply_rcparams,
    apply_style,
)


def zin_diff(cfg: OpampConfig, frequency_hz: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Return differential input impedance vs frequency (Rin || Cin)."""
    omega = 2.0 * np.pi * frequency_hz.astype(np.float64)
    cap = 1.0 / (1j * omega * cfg.cin_f) if cfg.cin_f > 0.0 else np.inf
    return 1.0 / (1.0 / cfg.rin_ohm + 1.0 / cap)


def zout(cfg: OpampConfig, frequency_hz: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Return output impedance vs frequency (Rout || Cout)."""
    omega = 2.0 * np.pi * frequency_hz.astype(np.float64)
    cap = 1.0 / (1j * omega * cfg.cout_f) if cfg.cout_f > 0.0 else np.inf
    return 1.0 / (1.0 / cfg.rout_ohm + 1.0 / cap)


def impedance_frequency_grid(cfg: OpampConfig) -> NDArray[np.float64]:
    """Return the log-spaced frequency grid used for impedance plots."""
    sweep = cfg.sweep
    return log_frequency_sweep(sweep.f_start_hz, sweep.f_stop_hz, sweep.points_per_decade)


def plot_impedance(
    cfg: OpampConfig,
    output_path: Path,
    *,
    title: str = "Input / output impedance",
) -> Path:
    """Plot |Zin| and |Zout| vs frequency and write an SVG."""
    apply_rcparams()
    frequency_hz = impedance_frequency_grid(cfg)
    zin_abs = np.abs(zin_diff(cfg, frequency_hz))
    zout_abs = np.abs(zout(cfg, frequency_hz))

    fig, (ax_in, ax_out) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)
    ax_in.semilogx(
        frequency_hz,
        zin_abs,
        color=LINE_COLORS["gain"],
        linewidth=LINEWIDTH_MAIN,
        label="|Zin|",
    )
    ax_in.set_ylabel("|Zin| (Ω)", fontsize=13)
    ax_in.set_title(title, fontsize=14)

    ax_out.semilogx(
        frequency_hz,
        zout_abs,
        color=LINE_COLORS["phase"],
        linewidth=LINEWIDTH_SECONDARY,
        label="|Zout|",
    )
    ax_out.set_ylabel("|Zout| (Ω)", fontsize=13)
    ax_out.set_xlabel("Frequency (Hz)", fontsize=13)

    apply_style(ax_in)
    apply_style(ax_out)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path
