"""Spectre netlist rendering tests (no simulator required)."""

from __future__ import annotations

from opamp_model.config import OpampConfig
from opamp_model.io import package_root
from opamp_model.spectre_engine import render_spectre_ac_netlist


def test_render_ac_netlist_contains_parameters() -> None:
    """Rendered AC netlist embeds CLI macromodel values."""
    cfg = OpampConfig(a0_db=70.0, gbw_hz=2.0e6)
    template = package_root() / "testbench" / "spectre" / "ac_open_loop.scs"
    text = render_spectre_ac_netlist(template, cfg)
    assert "parameters a0_db=70" in text
    assert "parameters gbw_hz=2000000" in text or "parameters gbw_hz=2e6" in text
    assert 'include "./testbench' not in text
    assert "ahdl_include" in text
    assert str(package_root() / "veriloga/configurable_opamp.va") in text


def test_veriloga_laplace_uses_array_variables() -> None:
    """Spectre requires laplace_nd coeffs in array variables, not inline expressions."""
    va = (package_root() / "veriloga/configurable_opamp.va").read_text(encoding="utf-8")
    assert "real numer[0:1]" in va
    assert "laplace_nd(vd, numer, denom)" in va
    assert "{a0_linear * wp" not in va
