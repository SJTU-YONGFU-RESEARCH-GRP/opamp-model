"""Tests for THD transient sine simulation and FFT analysis."""

from __future__ import annotations

import numpy as np

from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.tran import (
    compute_thd,
    is_thd_ideal,
    simulate_sine_response,
    transient_noise_rms,
)


def _run_sine_thd(
    cfg: OpampConfig,
    *,
    nl_a3: float = 0.0,
    freq_hz: float = 1.0e3,
    amplitude_v: float = 1.0e-3,
) -> float:
    """Helper: simulate and return THD (dB)."""
    cfg_nl = OpampConfig(
        a0_db=cfg.a0_db,
        gbw_hz=cfg.gbw_hz,
        nl_a2=cfg.nl_a2,
        nl_a3=nl_a3,
    )
    result = simulate_sine_response(
        cfg_nl,
        OpampNoiseConfig(
            en_white_v_per_sqrt_hz=0.0,
            en_flicker_1hz_v_per_sqrt_hz=0.0,
        ),
        amplitude_v=amplitude_v,
        freq_hz=freq_hz,
        cycles=30.0,
        dt_s=2.0e-6,
    )
    return compute_thd(result["time_s"], result["vout_v"], freq_hz)["thd_db"]


def test_linear_sine_thd_very_low() -> None:
    """Pure linear gain yields negligible harmonic content."""
    cfg = OpampConfig(a0_db=40.0, nl_a2=0.0, nl_a3=0.0)
    thd_db = _run_sine_thd(cfg)
    assert thd_db < -60.0


def test_cubic_nonlinearity_raises_thd() -> None:
    """Third-order term generates harmonics and raises THD."""
    cfg = OpampConfig(a0_db=40.0, nl_a2=0.0, nl_a3=0.0)
    thd_linear = _run_sine_thd(cfg, nl_a3=0.0)
    thd_nl = _run_sine_thd(cfg, nl_a3=1.0e4)
    assert thd_nl > thd_linear + 10.0


def test_is_thd_ideal_flag() -> None:
    """Ideal mode or zero NL coefficients skip distortion reporting."""
    cfg = OpampConfig(nl_a2=0.0, nl_a3=0.0)
    assert is_thd_ideal(cfg, ideal_flag=True)
    assert is_thd_ideal(cfg, ideal_flag=False)
    cfg_nl = OpampConfig(nl_a2=0.0, nl_a3=1.0)
    assert not is_thd_ideal(cfg_nl, ideal_flag=False)
    assert is_thd_ideal(cfg_nl, ideal_flag=True)


def test_sine_response_adds_input_referred_noise() -> None:
    """Noise-enabled sine simulation differs from the clean trace."""
    cfg = OpampConfig(a0_db=40.0, nl_a3=0.0)
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=5.0e-4,
        en_flicker_1hz_v_per_sqrt_hz=0.0,
        noise_seed=99,
    )
    sine = simulate_sine_response(
        cfg,
        noise,
        amplitude_v=1.0e-3,
        freq_hz=1.0e3,
        cycles=5.0,
        dt_s=1.0e-6,
    )
    assert transient_noise_rms(sine["noise_v"]) > 0.0
    assert np.max(np.abs(sine["vout_v"] - sine["vout_clean_v"])) > 0.0


def test_compute_thd_returns_hd2_hd3() -> None:
    """THD metrics include HD2 and HD3 keys."""
    cfg = OpampConfig(a0_db=40.0, nl_a3=5.0e3)
    sine = simulate_sine_response(
        cfg,
        None,
        amplitude_v=2.0e-3,
        freq_hz=2.0e3,
        cycles=25.0,
        dt_s=1.0e-6,
    )
    result = compute_thd(sine["time_s"], sine["vout_v"], 2.0e3)
    assert np.isfinite(result["thd_db"])
    assert np.isfinite(result["hd2_db"])
    assert np.isfinite(result["hd3_db"])
    assert result["fundamental_v"] > 0.0
