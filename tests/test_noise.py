"""Tests for noise density helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from opamp_model.config import OpampNoiseConfig
from opamp_model.noise import (
    flicker_corner_frequency,
    flicker_power_at_1hz,
    flicker_voltage_density,
    input_referred_en,
    input_referred_in,
)


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


def test_flicker_density_paper_formula_ef1() -> None:
    """Flicker density follows en_1hz / sqrt(f) for EF=1."""
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=5.0e-9,
        en_flicker_1hz_v_per_sqrt_hz=50.0e-9,
        en_flicker_ef=1.0,
    )
    f = np.array([1.0, 100.0, 10.0e3])
    flicker = flicker_voltage_density(f, noise)
    assert flicker[0] == pytest.approx(50.0e-9)
    assert flicker[1] == pytest.approx(5.0e-9)
    assert flicker[2] == pytest.approx(0.5e-9)


def test_flicker_corner_matches_white_intersection() -> None:
    """Corner frequency is where white and flicker densities are equal."""
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=5.0e-9,
        en_flicker_1hz_v_per_sqrt_hz=50.0e-9,
        en_flicker_ef=1.0,
    )
    fc = flicker_corner_frequency(noise)
    assert fc == pytest.approx(100.0)
    f = np.array([fc])
    en = input_referred_en(f, noise)
    assert en[0] == pytest.approx(5.0e-9 * math.sqrt(2.0), rel=1.0e-6)


def test_flicker_ef_general_exponent() -> None:
    """EF=2 gives 1/f density roll-off (density ∝ 1/f)."""
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=1.0e-9,
        en_flicker_1hz_v_per_sqrt_hz=10.0e-9,
        en_flicker_ef=2.0,
    )
    f = np.array([1.0, 10.0])
    flicker = flicker_voltage_density(f, noise)
    assert flicker[0] == pytest.approx(10.0e-9)
    assert flicker[1] == pytest.approx(1.0e-9)
    fc = flicker_corner_frequency(noise)
    assert fc == pytest.approx(10.0)


def test_coram_flicker_power_and_density() -> None:
    """Coram model sets 1 Hz power to KF * |I|^AF."""
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=5.0e-9,
        kf=2.0e-24,
        af=1.0,
        bias_current_a=1.0e-6,
    )
    power = flicker_power_at_1hz(noise)
    assert power == pytest.approx(2.0e-24 * 1.0e-6)
    en_1hz = math.sqrt(power)
    f = np.array([1.0])
    assert flicker_voltage_density(f, noise)[0] == pytest.approx(en_1hz)
    assert noise.enabled is True


def test_legacy_constructor_preserves_spectrum() -> None:
    """Legacy en_flicker_at_1hz + corner kwargs match prior macromodel."""
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=5.0e-9,
        en_flicker_at_1hz_v_per_sqrt_hz=50.0e-9,
        en_flicker_corner_hz=100.0,
    )
    migrated_1hz = 50.0e-9 * math.sqrt(100.0)
    assert noise.en_flicker_1hz_v_per_sqrt_hz == pytest.approx(migrated_1hz)
    f = np.array([1.0e6])
    en = input_referred_en(f, noise)
    assert en[0] == pytest.approx(5.0e-9, rel=0.01)


def test_input_referred_current_noise() -> None:
    """Current noise combines white and flicker with the same EF."""
    noise = OpampNoiseConfig(
        in_white_a_per_sqrt_hz=1.0e-12,
        in_flicker_1hz_a_per_sqrt_hz=10.0e-12,
        en_flicker_ef=1.0,
    )
    f = np.array([1.0, 100.0])
    density = input_referred_in(f, noise)
    assert density[0] == pytest.approx(math.hypot(1.0e-12, 10.0e-12))
    assert density[1] == pytest.approx(math.hypot(1.0e-12, 1.0e-12))
