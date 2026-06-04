"""Tests for noise density helpers."""

from __future__ import annotations

import numpy as np
import pytest

from opamp_model.config import OpampNoiseConfig
from opamp_model.noise import input_referred_en


def test_white_noise_floor_at_high_frequency() -> None:
    """At high f, density approaches white floor."""
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=5.0e-9,
        en_flicker_at_1hz_v_per_sqrt_hz=50.0e-9,
        en_flicker_corner_hz=100.0,
    )
    f = np.array([1.0e6])
    en = input_referred_en(f, noise)
    assert en[0] == pytest.approx(5.0e-9, rel=0.01)
