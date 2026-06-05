"""Tests for time-domain noise sample generation."""

from __future__ import annotations

import numpy as np
import pytest

from opamp_model.config import OpampNoiseConfig
from opamp_model.noise_tran import (
    input_referred_transient_noise,
    transient_noise_rms,
)


def test_transient_noise_zero_when_disabled() -> None:
    """Disabled noise config yields an all-zero waveform."""
    time_s = np.linspace(0.0, 1.0e-3, 1000)
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=0.0,
        en_flicker_1hz_v_per_sqrt_hz=0.0,
    )
    samples = input_referred_transient_noise(time_s, noise)
    assert np.all(samples == 0.0)
    assert transient_noise_rms(samples) == 0.0


def test_transient_noise_reproducible_with_seed() -> None:
    """Identical seeds produce identical noise traces."""
    time_s = np.linspace(0.0, 2.0e-3, 2000)
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=5.0e-6,
        en_flicker_1hz_v_per_sqrt_hz=20.0e-6,
        noise_seed=42,
    )
    a = input_referred_transient_noise(time_s, noise)
    b = input_referred_transient_noise(time_s, noise)
    np.testing.assert_allclose(a, b)


def test_transient_noise_positive_rms_when_enabled() -> None:
    """Enabled white noise yields finite RMS on a long window."""
    time_s = np.linspace(0.0, 1.0e-2, 10_000)
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=1.0e-5,
        en_flicker_1hz_v_per_sqrt_hz=0.0,
        noise_seed=7,
    )
    samples = input_referred_transient_noise(time_s, noise)
    rms = transient_noise_rms(samples)
    assert rms > 0.0
    assert rms == pytest.approx(np.std(samples), rel=0.01)
