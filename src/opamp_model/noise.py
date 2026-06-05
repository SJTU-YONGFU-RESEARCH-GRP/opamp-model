"""Noise density models: white, flicker, shot."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from opamp_model.config import OpampNoiseConfig


def _voltage_flicker_density_at_1hz(noise: OpampNoiseConfig) -> float:
    if noise.kf > 0.0:
        current = abs(noise.bias_current_a)
        return math.sqrt(noise.kf * current ** noise.af)
    return noise.en_flicker_1hz_v_per_sqrt_hz


def flicker_power_at_1hz(noise: OpampNoiseConfig, *, voltage: bool = True) -> float:
    """Return flicker noise power spectral density at 1 Hz (V²/Hz or A²/Hz)."""
    if voltage:
        en = _voltage_flicker_density_at_1hz(noise)
        return en * en if en > 0.0 else 0.0
    in_1hz = _current_flicker_density_at_1hz(noise)
    return in_1hz * in_1hz if in_1hz > 0.0 else 0.0


def _current_flicker_density_at_1hz(noise: OpampNoiseConfig) -> float:
    return noise.in_flicker_1hz_a_per_sqrt_hz


def flicker_voltage_density(
    frequency_hz: NDArray[np.float64] | float,
    noise: OpampNoiseConfig,
) -> NDArray[np.float64]:
    """Return input-referred voltage flicker density (V/√Hz)."""
    en_1hz = _voltage_flicker_density_at_1hz(noise)
    if en_1hz <= 0.0 or noise.en_flicker_ef <= 0.0:
        f = np.asarray(frequency_hz, dtype=np.float64)
        return np.zeros_like(f)
    f = np.maximum(np.asarray(frequency_hz, dtype=np.float64), 1.0e-30)
    return (en_1hz / f ** (noise.en_flicker_ef / 2.0)).astype(np.float64)


def flicker_current_density(
    frequency_hz: NDArray[np.float64] | float,
    noise: OpampNoiseConfig,
) -> NDArray[np.float64]:
    """Return input-referred current flicker density (A/√Hz)."""
    in_1hz = _current_flicker_density_at_1hz(noise)
    if in_1hz <= 0.0 or noise.en_flicker_ef <= 0.0:
        f = np.asarray(frequency_hz, dtype=np.float64)
        return np.zeros_like(f)
    f = np.maximum(np.asarray(frequency_hz, dtype=np.float64), 1.0e-30)
    return (in_1hz / f ** (noise.en_flicker_ef / 2.0)).astype(np.float64)


def flicker_corner_frequency(noise: OpampNoiseConfig) -> float:
    """Return voltage flicker corner where white density equals flicker density."""
    white = noise.en_white_v_per_sqrt_hz
    flicker_1hz = _voltage_flicker_density_at_1hz(noise)
    ef = noise.en_flicker_ef
    if white <= 0.0 or flicker_1hz <= 0.0 or ef <= 0.0:
        return float("nan")
    return float((flicker_1hz / white) ** (2.0 / ef))


def in_flicker_corner_frequency(noise: OpampNoiseConfig) -> float:
    """Return current flicker corner where white density equals flicker density."""
    white = noise.in_white_a_per_sqrt_hz
    flicker_1hz = _current_flicker_density_at_1hz(noise)
    ef = noise.en_flicker_ef
    if white <= 0.0 or flicker_1hz <= 0.0 or ef <= 0.0:
        return float("nan")
    return float((flicker_1hz / white) ** (2.0 / ef))


def input_referred_en(
    frequency_hz: NDArray[np.float64],
    noise: OpampNoiseConfig,
) -> NDArray[np.float64]:
    """Return input-referred voltage noise density (V/√Hz) vs ``frequency_hz``.

    White plus flicker (paper: ``en_1hz / f^(EF/2)``, RSS with white).
    """
    f = np.asarray(frequency_hz, dtype=np.float64)
    white = np.full_like(f, noise.en_white_v_per_sqrt_hz)
    flicker = flicker_voltage_density(f, noise)
    return np.sqrt(white**2 + flicker**2)


def input_referred_in(
    frequency_hz: NDArray[np.float64],
    noise: OpampNoiseConfig,
) -> NDArray[np.float64]:
    """Return input-referred current noise density (A/√Hz) vs ``frequency_hz``."""
    f = np.asarray(frequency_hz, dtype=np.float64)
    white = np.full_like(f, noise.in_white_a_per_sqrt_hz)
    flicker = flicker_current_density(f, noise)
    return np.sqrt(white**2 + flicker**2)
