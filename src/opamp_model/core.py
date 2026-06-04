"""Small-signal op-amp core: dominant-pole open-loop and loop gain."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.io import log_frequency_sweep


def dominant_pole_rad_s(cfg: OpampConfig) -> float:
    """Return dominant-pole frequency (rad/s) for the one-pole AOL model.

    Uses ``GBW`` as the unity-gain frequency in Hz: ``f_ug = gbw_hz``,
    ``wp = 2*pi*f_ug/a0_linear``.
    """
    return 2.0 * np.pi * cfg.gbw_hz / max(cfg.a0_linear, 1.0)


def open_loop_transfer(
    cfg: OpampConfig,
    frequency_hz: NDArray[np.float64],
    noise: OpampNoiseConfig | None = None,
) -> NDArray[np.complex128]:
    """Return open-loop differential voltage gain vs frequency.

    ``H(s) = A0 / (1 + s/wp)`` with optional second pole and zero when configured.
    Input/output loading (Rin, Cin, Rout, Cout) does not modify the ideal voltage
    gain in Phase 1; impedances are exposed via :mod:`opamp_model.impedance`.

    Args:
        cfg: Op-amp macromodel parameters.
        frequency_hz: Frequency samples (Hz).
        noise: Reserved for Phase 3 (ignored in Phase 1).

    Returns:
        Complex open-loop gain at each frequency.
    """
    _ = noise
    omega = 2.0 * np.pi * frequency_hz.astype(np.float64)
    wp = dominant_pole_rad_s(cfg)
    s = 1j * omega
    h = cfg.a0_linear / (1.0 + s / wp)

    if cfg.fp2_hz > 0.0:
        wp2 = 2.0 * np.pi * cfg.fp2_hz
        h /= 1.0 + s / wp2

    if cfg.fz_hz > 0.0:
        wz = 2.0 * np.pi * cfg.fz_hz
        h *= 1.0 + s / wz

    return h.astype(np.complex128)


def loop_transfer(
    cfg: OpampConfig,
    frequency_hz: NDArray[np.float64],
    noise: OpampNoiseConfig | None = None,
) -> NDArray[np.complex128]:
    """Return loop gain ``beta * A_open(s)`` for STB analysis."""
    return (cfg.loop_beta * open_loop_transfer(cfg, frequency_hz, noise)).astype(
        np.complex128
    )


def bode_from_transfer(
    transfer: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert complex transfer samples to gain (dB) and phase (degrees)."""
    gain_db = 20.0 * np.log10(np.abs(transfer) + 1.0e-30)
    phase_deg = np.angle(transfer, deg=True)
    return gain_db.astype(np.float64), phase_deg.astype(np.float64)


def open_loop_bode(
    cfg: OpampConfig,
    noise: OpampNoiseConfig | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(frequency_hz, gain_db, phase_deg)`` for open-loop AC."""
    f = log_frequency_sweep(
        cfg.sweep.f_start_hz,
        cfg.sweep.f_stop_hz,
        cfg.sweep.points_per_decade,
    )
    h = open_loop_transfer(cfg, f, noise)
    gain_db, phase_deg = bode_from_transfer(h)
    return f, gain_db, phase_deg


def loop_bode(
    cfg: OpampConfig,
    noise: OpampNoiseConfig | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(frequency_hz, gain_db, phase_deg)`` for loop-gain STB."""
    f = log_frequency_sweep(
        cfg.sweep.f_start_hz,
        cfg.sweep.f_stop_hz,
        cfg.sweep.points_per_decade,
    )
    h = loop_transfer(cfg, f, noise)
    gain_db, phase_deg = bode_from_transfer(h)
    return f, gain_db, phase_deg
