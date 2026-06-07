"""Multi-engine TRAN (slew / THD) tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.io import package_root
from opamp_model.ngspice_engine import (
    NgspiceNotFoundError,
    render_ngspice_slew_netlist,
    render_ngspice_thd_netlist,
    simulate_slew_ngspice,
    simulate_thd_ngspice,
)
from opamp_model.spectre_engine import (
    render_spectre_slew_netlist,
    render_spectre_thd_netlist,
    spectre_is_runnable,
)
from opamp_model.tran import measure_slew_rates, simulate_sine_response, compute_thd

_NGSPICE_AVAILABLE = (
    shutil.which("ngspice") is not None or shutil.which("ngspice-shared") is not None
)


def test_render_ngspice_slew_netlist_has_tran() -> None:
    """Rendered ngspice slew netlist runs a follower transient."""
    cfg = OpampConfig(gbw_hz=5.0e6, slew_pos_vps=8.0e6)
    template = package_root() / "testbench" / "ngspice" / "slew_follower.cir"
    text = render_ngspice_slew_netlist(
        template,
        cfg,
        step_v=0.5,
        duration_s=1.0e-6,
        dt_s=2.0e-9,
        wrdata_name="slew_pos.raw",
    )
    assert ".tran" in text
    assert "Bs 0 n I" in text
    assert "wrdata slew_pos.raw" in text


def test_render_ngspice_thd_netlist_polynomial() -> None:
    """Rendered ngspice THD netlist includes memoryless polynomial."""
    cfg = OpampConfig(a0_db=70.0, nl_a2=1.0e5, nl_a3=1.0e7)
    template = package_root() / "testbench" / "ngspice" / "thd_open_loop.cir"
    text = render_ngspice_thd_netlist(
        template,
        cfg,
        amplitude_v=2.0e-3,
        freq_hz=2.0e3,
        cycles=10.0,
        dt_s=1.0e-6,
        wrdata_name="thd_waveform.raw",
    )
    assert "Bout out 0 V" in text
    assert "VSIN inp 0 SIN" in text


def test_render_spectre_tran_netlists() -> None:
    """Rendered Spectre TRAN netlists carry macromodel parameters."""
    repo = package_root()
    cfg = OpampConfig(gbw_hz=8.0e6, nl_a2=2.0e4)
    slew = render_spectre_slew_netlist(
        repo / "testbench" / "spectre" / "slew_follower.scs",
        cfg,
        step_v=0.4,
        duration_s=1.5e-6,
        dt_s=1.0e-9,
    )
    thd = render_spectre_thd_netlist(
        repo / "testbench" / "spectre" / "thd_open_loop.scs",
        cfg,
        amplitude_v=1.0e-3,
        freq_hz=1.0e3,
        cycles=5.0,
        dt_s=2.0e-6,
    )
    assert "tran tran" in slew
    assert "Bout (out 0) bsource v=" in thd


@pytest.mark.skipif(not _NGSPICE_AVAILABLE, reason="ngspice not on PATH")
def test_ngspice_slew_matches_python(tmp_path: Path) -> None:
    """ngspice slew rates align with Python macromodel."""
    cfg = OpampConfig(gbw_hz=10.0e6, slew_pos_vps=10.0e6, slew_neg_vps=-10.0e6)
    py_metrics, _, _ = measure_slew_rates(cfg, OpampNoiseConfig(), step_v=0.8)
    try:
        ng_metrics, _, _ = simulate_slew_ngspice(cfg, tmp_path, step_v=0.8)
    except NgspiceNotFoundError:
        pytest.skip("ngspice not available")
    assert ng_metrics["slew_pos_vps"] == pytest.approx(py_metrics["slew_pos_vps"], rel=0.15)
    assert ng_metrics["slew_neg_vps"] == pytest.approx(py_metrics["slew_neg_vps"], rel=0.15)


@pytest.mark.skipif(not _NGSPICE_AVAILABLE, reason="ngspice not on PATH")
def test_ngspice_thd_matches_python(tmp_path: Path) -> None:
    """ngspice THD aligns with Python polynomial macromodel."""
    cfg = OpampConfig(a0_db=80.0, nl_a2=1.0e5, nl_a3=1.0e7)
    py = simulate_sine_response(cfg, None, amplitude_v=1.0e-3, freq_hz=1.0e3, cycles=20.0, dt_s=1.0e-6)
    py_thd = compute_thd(py["time_s"], py["vout_v"], 1.0e3)
    try:
        _, ng_thd = simulate_thd_ngspice(
            cfg,
            tmp_path,
            amplitude_v=1.0e-3,
            freq_hz=1.0e3,
            cycles=20.0,
            dt_s=1.0e-6,
        )
    except NgspiceNotFoundError:
        pytest.skip("ngspice not available")
    assert ng_thd is not None
    assert ng_thd["thd_db"] == pytest.approx(py_thd["thd_db"], abs=1.0)


@pytest.mark.skipif(not spectre_is_runnable(), reason="Spectre not runnable")
def test_spectre_thd_matches_python(tmp_path: Path) -> None:
    """Spectre THD aligns with Python polynomial macromodel."""
    from opamp_model.spectre_engine import simulate_thd_spectre

    cfg = OpampConfig(a0_db=80.0, nl_a2=1.0e5, nl_a3=1.0e7)
    py = simulate_sine_response(cfg, None, amplitude_v=1.0e-3, freq_hz=1.0e3, cycles=20.0, dt_s=1.0e-6)
    py_thd = compute_thd(py["time_s"], py["vout_v"], 1.0e3)
    _, sp_thd = simulate_thd_spectre(
        cfg,
        tmp_path,
        amplitude_v=1.0e-3,
        freq_hz=1.0e3,
        cycles=20.0,
        dt_s=1.0e-6,
    )
    assert sp_thd is not None
    assert sp_thd["thd_db"] == pytest.approx(py_thd["thd_db"], abs=1.0)


@pytest.mark.skipif(not spectre_is_runnable(), reason="Spectre not runnable")
def test_spectre_slew_matches_python(tmp_path: Path) -> None:
    """Spectre slew rates align with Python macromodel."""
    from opamp_model.spectre_engine import simulate_slew_spectre

    cfg = OpampConfig(gbw_hz=10.0e6, slew_pos_vps=10.0e6, slew_neg_vps=-10.0e6)
    py_metrics, _, _ = measure_slew_rates(cfg, OpampNoiseConfig(), step_v=0.8)
    sp_metrics, _, _ = simulate_slew_spectre(cfg, tmp_path, step_v=0.8)
    assert sp_metrics["slew_pos_vps"] == pytest.approx(py_metrics["slew_pos_vps"], rel=0.15)
    assert sp_metrics["slew_neg_vps"] == pytest.approx(py_metrics["slew_neg_vps"], rel=0.15)
