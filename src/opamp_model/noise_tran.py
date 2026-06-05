"""Time-domain noise samples for transient simulation overlays."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from opamp_model.config import OpampNoiseConfig
from opamp_model.noise import flicker_corner_frequency, flicker_voltage_density


def _sample_dt(time_s: NDArray[np.float64]) -> float:
    """Return median time step (s) from a uniform or near-uniform grid."""
    if time_s.size < 2:
        return 1.0e-9
    dt = float(np.median(np.diff(time_s.astype(np.float64))))
    return max(dt, 1.0e-18)


def _white_noise_samples(
    dt_s: float,
    en_white_v_per_sqrt_hz: float,
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate white noise samples with one-sided PSD ``en_white^2`` (V²/Hz)."""
    sigma = en_white_v_per_sqrt_hz / np.sqrt(2.0 * dt_s)
    return rng.normal(0.0, sigma, size=n).astype(np.float64)


def _flicker_noise_samples(
    dt_s: float,
    noise: OpampNoiseConfig,
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate low-frequency flicker samples via a first-order shaping filter."""
    en_1hz = float(flicker_voltage_density(np.array([1.0]), noise)[0])
    if en_1hz <= 0.0:
        return np.zeros(n, dtype=np.float64)

    fc = flicker_corner_frequency(noise)
    if not np.isfinite(fc) or fc <= 0.0:
        fc = 1.0
    tau = 1.0 / (2.0 * np.pi * fc)
    alpha = np.exp(-dt_s / tau)
    beta = np.sqrt(max(1.0 - alpha**2, 0.0))

    out = np.empty(n, dtype=np.float64)
    state = 0.0
    drive = rng.normal(0.0, 1.0, size=n)
    for idx in range(n):
        state = alpha * state + beta * drive[idx]
        out[idx] = state

    std = float(np.std(out))
    if std < 1.0e-30:
        return np.zeros(n, dtype=np.float64)
    scale = en_1hz * np.sqrt(dt_s) / std
    return (out * scale).astype(np.float64)


def input_referred_transient_noise(
    time_s: NDArray[np.float64],
    noise: OpampNoiseConfig,
) -> NDArray[np.float64]:
    """Return input-referred noise voltage samples for transient overlay (V).

    Uses ``noise_seed`` for reproducibility. White samples follow
    ``en_white_v_per_sqrt_hz``; flicker is shaped to match ``flicker_voltage_density``.
    """
    if not noise.enabled:
        return np.zeros_like(time_s, dtype=np.float64)

    rng = np.random.default_rng(noise.noise_seed)
    n = int(time_s.size)
    dt_s = _sample_dt(time_s)
    white = _white_noise_samples(dt_s, noise.en_white_v_per_sqrt_hz, n, rng)
    flicker = _flicker_noise_samples(dt_s, noise, n, rng)
    return (white + flicker).astype(np.float64)


def transient_noise_rms(samples_v: NDArray[np.float64]) -> float:
    """Return RMS of a noise-only transient waveform (V)."""
    if samples_v.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples_v.astype(np.float64) ** 2)))
