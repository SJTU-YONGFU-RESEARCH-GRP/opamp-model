"""Noise density models: white, flicker, shot."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from opamp_model.config import OpampNoiseConfig


def input_referred_en(
    frequency_hz: NDArray[np.float64],
    noise: OpampNoiseConfig,
) -> NDArray[np.float64]:
    """Return input-referred voltage noise density (V/√Hz) vs ``frequency_hz``.

    White plus flicker terms; tests validate units and integration.
    """
    f = np.maximum(np.asarray(frequency_hz, dtype=np.float64), 1.0e-30)
    white = np.full_like(f, noise.en_white_v_per_sqrt_hz)
    flicker = np.zeros_like(f)
    if noise.en_flicker_at_1hz_v_per_sqrt_hz > 0.0 and noise.en_flicker_corner_hz > 0.0:
        flicker = noise.en_flicker_at_1hz_v_per_sqrt_hz * np.sqrt(
            noise.en_flicker_corner_hz / f
        )
    return np.sqrt(white**2 + flicker**2)
