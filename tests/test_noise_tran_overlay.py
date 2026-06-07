"""Transient noise overlay helpers for multi-engine TRAN."""

from __future__ import annotations

import numpy as np

from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.tran import (
    TranSineResult,
    TranStepResult,
    overlay_transient_noise_on_sine,
    overlay_transient_noise_on_step,
    simulate_step_response,
)


def test_overlay_step_matches_python_macromodel() -> None:
    """SPICE-style step overlay matches Python unity-gain noise addition."""
    cfg = OpampConfig()
    noise = OpampNoiseConfig(noise_seed=11)
    py = simulate_step_response(cfg, noise, step_v=0.5, duration_s=1.0e-6, dt_s=2.0e-9)
    clean = TranStepResult(
        time_s=py["time_s"],
        vout_v=py["vout_clean_v"],
        vout_clean_v=py["vout_clean_v"],
        noise_v=np.zeros_like(py["time_s"]),
    )
    overlay = overlay_transient_noise_on_step(clean, cfg, noise)
    np.testing.assert_allclose(overlay["noise_v"], py["noise_v"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(overlay["vout_v"], py["vout_v"], rtol=0.0, atol=0.0)


def test_overlay_sine_populates_noise_trace() -> None:
    """Open-loop sine overlay produces non-zero noise samples when enabled."""
    cfg = OpampConfig(a0_db=70.0, nl_a2=1.0e4)
    noise = OpampNoiseConfig(noise_seed=3)
    n = 100
    time_s = np.linspace(0.0, 1.0e-4, n, dtype=np.float64)
    clean = TranSineResult(
        time_s=time_s,
        vout_v=np.ones(n),
        vout_clean_v=np.ones(n),
        noise_v=np.zeros(n),
    )
    overlay = overlay_transient_noise_on_sine(
        clean,
        cfg,
        noise,
        amplitude_v=1.0e-3,
        freq_hz=1.0e3,
    )
    assert np.max(np.abs(overlay["noise_v"])) > 0.0
    assert not np.allclose(overlay["vout_v"], overlay["vout_clean_v"])
