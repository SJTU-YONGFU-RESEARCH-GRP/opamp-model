"""Tests for cross-engine metrics comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opamp_model.compare import (
    TOLERANCE_A0_DB,
    TOLERANCE_GBW_REL,
    TOLERANCE_NOISE_REL,
    TOLERANCE_PM_DEG,
    compare_engines,
    compute_spread,
    format_compare_table,
    load_engine_metrics,
)
from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.io import package_root
from opamp_model.metrics import build_metrics_report
from opamp_model.model import simulate_ac, simulate_noise


def _write_metrics(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _minimal_report(
    *,
    engine: str,
    a0_db: float = 80.0,
    gbw_hz: float = 10e6,
    phase_margin_deg: float = 60.0,
    cmrr_db: float = 90.0,
    psrr_db: float = 80.0,
    integrated_noise_rms_v: float = 0.01,
    slew_pos_vps: float | None = 10e6,
) -> dict:
    cfg = OpampConfig(a0_db=a0_db, gbw_hz=gbw_hz, cmrr_db=cmrr_db, psrr_db=psrr_db)
    noise = OpampNoiseConfig()
    ac = simulate_ac(cfg)
    stb = simulate_ac(cfg)
    noise_result = simulate_noise(cfg, noise)
    report = build_metrics_report(
        cfg,
        noise,
        engine=engine,
        ac_result=ac,
        stb_result=stb,
        noise_result=noise_result,
    )
    report["ac"]["a0_db"]["value"] = a0_db
    report["ac"]["gbw_hz"]["value"] = gbw_hz
    report["stb"]["phase_margin_deg"]["value"] = phase_margin_deg
    report["cmrr_psrr"]["cmrr_db"]["value"] = cmrr_db
    report["cmrr_psrr"]["psrr_db"]["value"] = psrr_db
    report["noise"]["integrated_noise_rms_v"]["value"] = integrated_noise_rms_v
    if slew_pos_vps is not None:
        report["large_signal"]["slew_pos_vps"]["value"] = slew_pos_vps
        report["large_signal"]["slew_pos_vps"]["status"] = "reported"
    else:
        report["large_signal"]["slew_pos_vps"]["value"] = None
    return report


def test_compute_spread_relative() -> None:
    """Relative spread uses midpoint normalization."""
    assert compute_spread([9.5e6, 10.0e6], kind="relative") == pytest.approx(0.5e6 / 9.75e6)


def test_compare_passes_when_engines_agree(tmp_path: Path) -> None:
    """Identical metrics across engines stay within MODEL.md tolerances."""
    for engine in ("python", "ngspice", "spectre"):
        _write_metrics(
            tmp_path / engine / "opamp_metrics.json",
            _minimal_report(engine=engine),
        )
    result = compare_engines(tmp_path, golden_path=tmp_path / "missing.yaml")
    assert result.passed
    assert result.failures == []
    text = format_compare_table(result)
    assert "python" in text
    assert "ok" in text


def test_compare_fails_a0_spread(tmp_path: Path) -> None:
    """A0 spread above 0.1 dB fails the check."""
    _write_metrics(tmp_path / "python" / "opamp_metrics.json", _minimal_report(engine="python"))
    _write_metrics(
        tmp_path / "ngspice" / "opamp_metrics.json",
        _minimal_report(engine="ngspice", a0_db=80.0 + TOLERANCE_A0_DB + 0.05),
    )
    _write_metrics(tmp_path / "spectre" / "opamp_metrics.json", _minimal_report(engine="spectre"))
    result = compare_engines(tmp_path, engines=("python", "ngspice", "spectre"))
    assert not result.passed
    assert any("A0" in f for f in result.failures)


def test_compare_fails_gbw_spread(tmp_path: Path) -> None:
    """GBW relative spread above 5% fails."""
    base = 10e6
    high = base * (1.0 + TOLERANCE_GBW_REL + 0.01)
    _write_metrics(tmp_path / "python" / "opamp_metrics.json", _minimal_report(engine="python"))
    _write_metrics(
        tmp_path / "ngspice" / "opamp_metrics.json",
        _minimal_report(engine="ngspice", gbw_hz=high),
    )
    _write_metrics(tmp_path / "spectre" / "opamp_metrics.json", _minimal_report(engine="spectre"))
    result = compare_engines(tmp_path)
    assert not result.passed
    assert any("GBW" in f for f in result.failures)


def test_compare_fails_pm_spread(tmp_path: Path) -> None:
    """Phase-margin spread above 2° fails."""
    _write_metrics(tmp_path / "python" / "opamp_metrics.json", _minimal_report(engine="python"))
    _write_metrics(
        tmp_path / "ngspice" / "opamp_metrics.json",
        _minimal_report(engine="ngspice", phase_margin_deg=60.0 + TOLERANCE_PM_DEG + 1.0),
    )
    _write_metrics(tmp_path / "spectre" / "opamp_metrics.json", _minimal_report(engine="spectre"))
    result = compare_engines(tmp_path)
    assert not result.passed
    assert any("Phase margin" in f for f in result.failures)


def test_compare_fails_noise_spread(tmp_path: Path) -> None:
    """Integrated noise RMS spread above 5% fails."""
    base = 0.01
    high = base * (1.0 + TOLERANCE_NOISE_REL + 0.02)
    _write_metrics(tmp_path / "python" / "opamp_metrics.json", _minimal_report(engine="python"))
    _write_metrics(
        tmp_path / "ngspice" / "opamp_metrics.json",
        _minimal_report(engine="ngspice", integrated_noise_rms_v=high),
    )
    _write_metrics(tmp_path / "spectre" / "opamp_metrics.json", _minimal_report(engine="spectre"))
    result = compare_engines(tmp_path)
    assert not result.passed
    assert any("Integrated noise" in f for f in result.failures)


def test_golden_column_optional(tmp_path: Path) -> None:
    """Golden YAML adds a reference column without affecting pass/fail."""
    for engine in ("python", "ngspice"):
        _write_metrics(
            tmp_path / engine / "opamp_metrics.json",
            _minimal_report(engine=engine),
        )
    golden = package_root() / "docs" / "golden_metrics.yaml"
    result = compare_engines(tmp_path, engines=("python", "ngspice"), golden_path=golden)
    assert result.golden_path == golden
    text = format_compare_table(result)
    assert "ref" in text


def test_load_engine_metrics_missing(tmp_path: Path) -> None:
    """Missing metrics file returns None."""
    assert load_engine_metrics(tmp_path, "python") is None
