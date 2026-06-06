"""Tests for Gm / OTA macromodel."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from opamp_model.config import GmConfig, OpampNoiseConfig
from opamp_model.gm import (
    bode_to_gm_result,
    extract_gm_metrics,
    simulate_gm_ac,
    transconductance_transfer,
    zout_gm,
)
from opamp_model.io import package_root
from opamp_model.metrics import build_gm_metric_entries
from opamp_model.ngspice_engine import (
    NgspiceNotFoundError,
    render_ngspice_gm_netlist,
    simulate_gm_ac_ngspice,
)
from opamp_model.spectre_engine import render_spectre_gm_netlist

_NGSPICE_AVAILABLE = (
    shutil.which("ngspice") is not None or shutil.which("ngspice-shared") is not None
)


def test_zout_near_rout_at_low_frequency() -> None:
    """At low frequency, |Zout| approaches Rout."""
    cfg = GmConfig(rout_ohm=1.0e6, cout_f=500.0e-15)
    z = zout_gm(cfg, np.array([1.0]))
    assert abs(z[0]) == pytest.approx(1.0e6, rel=0.01)


def test_transconductance_dc_gain() -> None:
    """Loaded DC gain equals gm * Rout in linear scale."""
    cfg = GmConfig(gm_s=2.0e-3, rout_ohm=50.0e3, cout_f=0.0)
    h = transconductance_transfer(cfg, np.array([10.0]))
    assert abs(h[0]) == pytest.approx(cfg.gm_s * cfg.rout_ohm, rel=1.0e-6)


def test_simulate_gm_ac_returns_bode_and_flat_gm() -> None:
    """simulate_gm_ac returns aligned vectors and constant gm."""
    cfg = GmConfig(gm_s=1.0e-3, rout_ohm=1.0e6, cout_f=500.0e-15)
    result = simulate_gm_ac(cfg)
    n = len(result["frequency_hz"])
    assert n == len(result["gain_db"]) == len(result["phase_deg"]) == len(result["gm_s"])
    assert np.allclose(result["gm_s"], cfg.gm_s)
    assert result["metrics"]["gm_s"] == cfg.gm_s
    assert result["frequency_hz"][0] > 0.0


def test_bandwidth_near_output_rc_pole() -> None:
    """−3 dB bandwidth tracks 1/(2π Rout Cout) within a decade."""
    cfg = GmConfig(gm_s=1.0e-3, rout_ohm=100.0e3, cout_f=1.0e-12)
    result = simulate_gm_ac(cfg)
    f_expected = 1.0 / (2.0 * np.pi * cfg.rout_ohm * cfg.cout_f)
    bw = result["metrics"]["bandwidth_hz"]
    assert bw == pytest.approx(f_expected, rel=0.5)


def test_extract_gm_metrics_from_sweep() -> None:
    """extract_gm_metrics reports configured gm and Rout."""
    cfg = GmConfig(gm_s=3.0e-3, rout_ohm=200.0e3)
    f = np.logspace(0, 6, 50)
    gain_db = np.full(50, 40.0)
    metrics = extract_gm_metrics(cfg, f, gain_db)
    assert metrics["gm_s"] == cfg.gm_s
    assert metrics["rout_ohm"] == cfg.rout_ohm
    assert metrics["gain_db"] == pytest.approx(40.0)


def test_transconductance_accepts_scalar_frequency() -> None:
    """transconductance_transfer accepts a scalar frequency."""
    cfg = GmConfig()
    h = transconductance_transfer(cfg, 1.0e3)
    assert h.shape == (1,)


def test_simulate_gm_ac_with_noise_config() -> None:
    """Noise config is accepted without changing small-signal gm."""
    cfg = GmConfig()
    noise = OpampNoiseConfig(en_white_v_per_sqrt_hz=1.0e-9)
    result = simulate_gm_ac(cfg, noise)
    assert result["gm_s"][0] == cfg.gm_s


def test_bode_to_gm_result_from_ac_columns() -> None:
    """Loaded AC columns with Vdiff=1 V map to Gm metrics."""
    cfg = GmConfig(gm_s=2.0e-3, rout_ohm=50.0e3)
    f = np.array([1.0, 1.0e6])
    gain_db = np.array([40.0, 30.0])
    phase_deg = np.array([0.0, -45.0])
    result = bode_to_gm_result(cfg, f, gain_db, phase_deg)
    assert result["metrics"]["gain_db"] == pytest.approx(40.0)
    assert result["metrics"]["gm_s"] == cfg.gm_s


def test_gm_metric_source_tags_engine() -> None:
    """Gm metrics record engine-specific provenance."""
    cfg = GmConfig()
    metrics = extract_gm_metrics(cfg, np.array([1.0]), np.array([40.0]))
    py = build_gm_metric_entries(metrics, engine="python")
    ng = build_gm_metric_entries(metrics, engine="ngspice")
    assert py["gm_s"]["source"] == "python_macromodel"
    assert ng["gm_s"]["source"] == "ngspice_wrdata"


def test_render_ngspice_gm_netlist_substitutes_params() -> None:
    """Rendered ngspice Gm netlist carries macromodel parameters."""
    repo = package_root()
    template = repo / "testbench" / "ngspice" / "run_gm_ac.cir"
    cfg = GmConfig(gm_s=2.5e-3, rout_ohm=200.0e3, cout_f=1.0e-12)
    text = render_ngspice_gm_netlist(template, cfg)
    assert ".param gm_s=0.0025" in text or ".param gm_s=2.5e-03" in text
    assert "wrdata gm_ac_bode.raw" in text


def test_render_spectre_gm_netlist_absolutizes_va() -> None:
    """Rendered Spectre Gm netlist uses an absolute configurable_gm.va path."""
    repo = package_root()
    template = repo / "testbench" / "spectre" / "run_gm_ac.scs"
    cfg = GmConfig()
    text = render_spectre_gm_netlist(template, cfg, repo_root=repo)
    va = (repo / "veriloga/configurable_gm.va").resolve()
    assert f'ahdl_include "{va}"' in text
    assert "configurable_gm" in text


@pytest.mark.skipif(not _NGSPICE_AVAILABLE, reason="ngspice not on PATH")
def test_ngspice_gm_matches_python(tmp_path: Path) -> None:
    """ngspice Gm AC aligns with Python loaded transfer at DC."""
    cfg = GmConfig(gm_s=1.0e-3, rout_ohm=1.0e6, cout_f=500.0e-15)
    py = simulate_gm_ac(cfg)
    try:
        ng = simulate_gm_ac_ngspice(cfg, tmp_path)
    except NgspiceNotFoundError:
        pytest.skip("ngspice not available")
    assert ng["metrics"]["gain_db"] == pytest.approx(py["metrics"]["gain_db"], rel=0.05)
    assert ng["metrics"]["gm_s"] == pytest.approx(cfg.gm_s)
