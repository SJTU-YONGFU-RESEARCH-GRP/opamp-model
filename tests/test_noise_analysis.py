"""Tests for noise integration and output-referred spectrum."""

from __future__ import annotations

import numpy as np
import pytest

from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.io import log_frequency_sweep
from opamp_model.model import simulate_noise
from opamp_model.noise import input_referred_en
from opamp_model.noise_analysis import (
    extract_noise_metrics,
    integrate_noise_rms,
    output_noise_spectrum,
    spot_noise_at_frequency,
)


def test_integrate_noise_rms_white_floor() -> None:
    """Flat density integrates to RMS = density * sqrt(bandwidth)."""
    f = np.logspace(0, 4, 500)
    density = np.full_like(f, 2.0e-9)
    rms = integrate_noise_rms(f, density)
    bw = f[-1] - f[0]
    assert rms == pytest.approx(2.0e-9 * np.sqrt(bw), rel=0.02)


def test_output_noise_scales_with_open_loop_gain() -> None:
    """Output spectrum exceeds input at low frequency where |A| > 1."""
    cfg = OpampConfig(a0_db=80.0, gbw_hz=10.0e6)
    noise = OpampNoiseConfig(en_white_v_per_sqrt_hz=5.0e-9)
    f = log_frequency_sweep(1.0, 100.0e6, 10)
    en_in = input_referred_en(f, noise)
    en_out = output_noise_spectrum(cfg, noise, f)
    assert en_out[0] > en_in[0]
    high_f_idx = len(f) // 2
    assert en_out[high_f_idx] < en_in[high_f_idx] * cfg.a0_linear


def test_spot_noise_at_1khz() -> None:
    """Spot interpolation returns white floor at 1 kHz for white-only noise."""
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=7.0e-9,
        en_flicker_at_1hz_v_per_sqrt_hz=0.0,
    )
    f = log_frequency_sweep(1.0, 1.0e6, 10)
    en = input_referred_en(f, noise)
    spot = spot_noise_at_frequency(f, en, 1.0e3)
    assert spot == pytest.approx(7.0e-9, rel=0.01)


def test_simulate_noise_finite_with_noise_enabled() -> None:
    """Integrated RMS is positive when noise is enabled (not --ideal)."""
    cfg = OpampConfig(a0_db=60.0, gbw_hz=1.0e6)
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=5.0e-9,
        en_flicker_at_1hz_v_per_sqrt_hz=0.0,
    )
    result = simulate_noise(cfg, noise)
    metrics = result["metrics"]
    assert metrics["integrated_noise_rms_v"] > 0.0
    assert metrics["en_out_spot_1khz_v_per_sqrt_hz"] > 0.0
    assert metrics["en_in_spot_1khz_v_per_sqrt_hz"] == pytest.approx(5.0e-9, rel=0.05)


def test_simulate_noise_zero_when_ideal() -> None:
    """Disabled noise yields zero spectrum and integrated RMS."""
    cfg = OpampConfig()
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=0.0,
        en_flicker_at_1hz_v_per_sqrt_hz=0.0,
    )
    result = simulate_noise(cfg, noise)
    assert np.all(result["noise_v_per_sqrt_hz"] == 0.0)
    assert result["metrics"]["integrated_noise_rms_v"] == 0.0


def test_extract_noise_metrics_matches_simulate() -> None:
    """extract_noise_metrics agrees with simulate_noise bundle."""
    cfg = OpampConfig(a0_db=40.0)
    noise = OpampNoiseConfig(en_white_v_per_sqrt_hz=1.0e-8)
    sim = simulate_noise(cfg, noise)
    direct = extract_noise_metrics(
        cfg, noise, sim["frequency_hz"], sim["noise_v_per_sqrt_hz"]
    )
    assert direct["integrated_noise_rms_v"] == pytest.approx(
        sim["metrics"]["integrated_noise_rms_v"]
    )
