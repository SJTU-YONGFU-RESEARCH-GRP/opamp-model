"""Spectre noise netlist and engine tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.io import package_root
from opamp_model.spectre_engine import render_spectre_noise_netlist

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("spectre") is None
    and not Path("/eda/cadence/SPECTRE241/tools/bin/spectre").is_file(),
    reason="Spectre not available",
)


def test_render_noise_netlist_includes_noise_analysis() -> None:
    """Rendered Spectre deck runs ``noise`` on ``out`` with VA noise parameters."""
    cfg = OpampConfig(a0_db=70.0, gbw_hz=2.0e6)
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=4.0e-9,
        en_flicker_1hz_v_per_sqrt_hz=40.0e-9,
        en_flicker_ef=1.0,
    )
    template = package_root() / "testbench" / "spectre" / "noise_open_loop.scs"
    text = render_spectre_noise_netlist(template, cfg, noise)
    assert "noise noise start=" in text
    assert "oprobe=ROUT" in text
    assert "parameters en_white_v_per_sqrt_hz=4" in text
    assert "parameters en_flicker_1hz_v_per_sqrt_hz=" in text
    assert "4e-08" in text or "4e-8" in text
    assert "parameters enable_noise=1" in text
    assert "EN_FLICKER_1HZ_V_PER_SQRT_HZ=en_flicker_1hz_v_per_sqrt_hz" in text
    assert "ENABLE_NOISE=enable_noise" in text
    assert "parameters rn_white_ohm=1e12" in text or "parameters rn_white_ohm=1e+12" in text
    assert "RN (inp inn) resistor" in text
    assert "PLACEHOLDER_RN" not in text
    assert "ahdl_include" in text


def test_spectre_noise_simulation(tmp_path: Path) -> None:
    """Spectre noise run completes and returns finite metrics when PSF is readable."""
    pytest.importorskip("psf_parser")
    from opamp_model.spectre_engine import SpectreNotFoundError, simulate_noise_spectre

    cfg = OpampConfig(a0_db=60.0, gbw_hz=1.0e6)
    noise = OpampNoiseConfig()
    try:
        result = simulate_noise_spectre(cfg, tmp_path, noise)
    except SpectreNotFoundError:
        pytest.skip("Spectre not available")
    except RuntimeError as exc:
        msg = str(exc)
        skip_markers = ("PSF", "license", "LMC-", "failed with code")
        if any(marker.lower() in msg.lower() for marker in skip_markers):
            pytest.skip(f"Spectre noise not available: {exc}")
        raise
    assert result["metrics"]["integrated_noise_rms_v"] > 0.0
