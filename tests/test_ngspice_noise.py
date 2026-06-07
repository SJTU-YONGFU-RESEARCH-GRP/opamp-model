"""ngspice noise bench tests (skip when ngspice is not installed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.io import package_root
from opamp_model.model import simulate_noise
from opamp_model.ngspice_engine import (
    NgspiceNotFoundError,
    render_ngspice_noise_netlist,
    simulate_noise_ngspice,
)

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("ngspice") is None
    and __import__("shutil").which("ngspice-shared") is None,
    reason="ngspice not on PATH",
)


def test_render_noise_netlist_has_noise_and_ac() -> None:
    """Rendered noise netlist includes .noise, .ac, and thermal input resistor."""
    cfg = OpampConfig(gbw_hz=2.0e6)
    noise = OpampNoiseConfig(
        en_white_v_per_sqrt_hz=3.0e-9,
        en_flicker_1hz_v_per_sqrt_hz=30.0e-9,
    )
    template = package_root() / "testbench" / "ngspice" / "noise_open_loop.cir"
    text = render_ngspice_noise_netlist(template, cfg, noise)
    assert ".noise V(out)" in text
    assert ".ac dec" in text
    assert "Rn inp inn" in text
    assert "Eaux n1 0 inp inn" in text
    assert "OpampNoiseConfig" in text or "flicker" in text.lower()
    assert "wrdata noise_spectrum.raw" in text


def test_ngspice_noise_matches_python(tmp_path: Path) -> None:
    """ngspice noise metrics align with Python macromodel (same one-pole gain)."""
    cfg = OpampConfig()
    noise = OpampNoiseConfig()
    py = simulate_noise(cfg, noise)
    try:
        ng = simulate_noise_ngspice(cfg, tmp_path, noise)
    except NgspiceNotFoundError:
        pytest.skip("ngspice not available")
    assert ng["metrics"]["integrated_noise_rms_v"] == pytest.approx(
        py["metrics"]["integrated_noise_rms_v"],
        rel=0.02,
    )
    assert ng["metrics"]["en_out_spot_1khz_v_per_sqrt_hz"] == pytest.approx(
        py["metrics"]["en_out_spot_1khz_v_per_sqrt_hz"],
        rel=0.05,
    )
