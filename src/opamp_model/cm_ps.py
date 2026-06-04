"""Common-mode and power-supply rejection transfer functions (Phase 4)."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from opamp_model.config import OpampConfig
from opamp_model.core import bode_from_transfer, open_loop_transfer
from opamp_model.io import log_frequency_sweep
from opamp_model.plot_style import (
    FIGSIZE,
    LINE_COLORS,
    LINEWIDTH_MAIN,
    apply_rcparams,
    apply_style,
)


class CmrrSimulationResult(TypedDict):
    """Common-mode rejection simulation bundle."""

    frequency_hz: NDArray[np.float64]
    acm_db: NDArray[np.float64]
    cmrr_db: NDArray[np.float64]
    cmrr_dc_db: float


class PsrrSimulationResult(TypedDict):
    """Power-supply rejection simulation bundle."""

    frequency_hz: NDArray[np.float64]
    psrr_db: NDArray[np.float64]
    psrr_dc_db: float


def common_mode_transfer(
    cfg: OpampConfig,
    frequency_hz: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Return common-mode gain ``ACM(s) = Aol(s) / CMRR_linear``."""
    aol = open_loop_transfer(cfg, frequency_hz)
    return (aol / cfg.cmrr_linear).astype(np.complex128)


def psrr_transfer(
    cfg: OpampConfig,
    frequency_hz: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Return supply-to-output feedthrough ``H_ps(s) = (1/PSRR) / (1 + s/wp_psrr)``."""
    omega = 2.0 * np.pi * frequency_hz.astype(np.float64)
    wp = 2.0 * np.pi * max(cfg.psrr_pole_hz, 1.0e-30)
    s = 1j * omega
    h = (1.0 / cfg.psrr_linear) / (1.0 + s / wp)
    return h.astype(np.complex128)


def psrr_db_from_transfer(transfer: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convert supply feedthrough to PSRR in dB (``20*log10(1/|H_ps|)``)."""
    return (-20.0 * np.log10(np.abs(transfer) + 1.0e-30)).astype(np.float64)


def simulate_cmrr(cfg: OpampConfig) -> CmrrSimulationResult:
    """Simulate ACM and CMRR vs frequency from the macromodel."""
    f = log_frequency_sweep(
        cfg.sweep.f_start_hz,
        cfg.sweep.f_stop_hz,
        cfg.sweep.points_per_decade,
    )
    acm = common_mode_transfer(cfg, f)
    acm_db, _ = bode_from_transfer(acm)
    aol = open_loop_transfer(cfg, f)
    aol_db, _ = bode_from_transfer(aol)
    cmrr_db = (aol_db - acm_db).astype(np.float64)
    return CmrrSimulationResult(
        frequency_hz=f,
        acm_db=acm_db,
        cmrr_db=cmrr_db,
        cmrr_dc_db=float(cmrr_db[0]),
    )


def simulate_psrr(cfg: OpampConfig) -> PsrrSimulationResult:
    """Simulate PSRR vs frequency from the single-pole supply feedthrough model."""
    f = log_frequency_sweep(
        cfg.sweep.f_start_hz,
        cfg.sweep.f_stop_hz,
        cfg.sweep.points_per_decade,
    )
    h = psrr_transfer(cfg, f)
    psrr_db = psrr_db_from_transfer(h)
    return PsrrSimulationResult(
        frequency_hz=f,
        psrr_db=psrr_db,
        psrr_dc_db=float(psrr_db[0]),
    )


def plot_psrr(
    psrr_result: PsrrSimulationResult,
    output_path: Path,
    *,
    title: str = "PSRR vs frequency",
) -> Path:
    """Plot PSRR vs frequency and write an SVG."""
    apply_rcparams()
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.semilogx(
        psrr_result["frequency_hz"],
        psrr_result["psrr_db"],
        color=LINE_COLORS["psrr"],
        linewidth=LINEWIDTH_MAIN,
    )
    ax.set_ylabel("PSRR (dB)", fontsize=13)
    ax.set_xlabel("Frequency (Hz)", fontsize=13)
    ax.set_title(
        f"{title}\nPSRR(DC)={psrr_result['psrr_dc_db']:.1f} dB",
        fontsize=14,
    )
    apply_style(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path
