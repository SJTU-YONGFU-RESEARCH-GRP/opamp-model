"""Simulation orchestration entry points."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from opamp_model.ac import AcMetrics, extract_gbw_phase_margin
from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.core import loop_bode, open_loop_bode
from opamp_model.noise_analysis import NoiseMetrics, extract_noise_metrics, output_noise_spectrum


class AcSimulationResult(TypedDict):
    """AC simulation result bundle."""

    frequency_hz: NDArray[np.float64]
    gain_db: NDArray[np.float64]
    phase_deg: NDArray[np.float64]
    metrics: AcMetrics


def simulate_ac(
    cfg: OpampConfig,
    noise: OpampNoiseConfig | None = None,
) -> AcSimulationResult:
    """Run open-loop AC / Bode simulation (Python macromodel)."""
    _ = noise
    f, gain_db, phase_deg = open_loop_bode(cfg, noise)
    metrics = extract_gbw_phase_margin(f, gain_db, phase_deg)
    return AcSimulationResult(
        frequency_hz=f,
        gain_db=gain_db,
        phase_deg=phase_deg,
        metrics=metrics,
    )


def simulate_stb(
    cfg: OpampConfig,
    noise: OpampNoiseConfig | None = None,
) -> AcSimulationResult:
    """Run STB / loop-gain Bode simulation (``beta * A_open``)."""
    _ = noise
    f, gain_db, phase_deg = loop_bode(cfg, noise)
    metrics = extract_gbw_phase_margin(f, gain_db, phase_deg)
    return AcSimulationResult(
        frequency_hz=f,
        gain_db=gain_db,
        phase_deg=phase_deg,
        metrics=metrics,
    )


class NoiseSimulationResult(TypedDict):
    """Noise simulation result bundle."""

    frequency_hz: NDArray[np.float64]
    noise_v_per_sqrt_hz: NDArray[np.float64]
    metrics: NoiseMetrics


def simulate_noise(
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
) -> NoiseSimulationResult:
    """Run noise analysis (Python macromodel)."""
    from opamp_model.io import log_frequency_sweep

    f = log_frequency_sweep(
        cfg.sweep.f_start_hz,
        cfg.sweep.f_stop_hz,
        cfg.sweep.points_per_decade,
    )
    spectrum = output_noise_spectrum(cfg, noise, f)
    metrics = extract_noise_metrics(cfg, noise, f, spectrum)
    return NoiseSimulationResult(
        frequency_hz=f,
        noise_v_per_sqrt_hz=spectrum,
        metrics=metrics,
    )
