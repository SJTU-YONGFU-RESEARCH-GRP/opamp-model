"""Tests for Markdown report generation."""

from __future__ import annotations

from pathlib import Path

from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.impedance import plot_impedance
from opamp_model.metrics import build_metrics_report
from opamp_model.model import simulate_ac
from opamp_model.report import (
    preserve_metrics_sections,
    write_ac_report,
    write_engine_report,
    write_stb_report,
)


def test_ac_report_embeds_figures(tmp_path: Path) -> None:
    """AC_REPORT.md references SVG plots and metric tables."""
    cfg = OpampConfig(a0_db=60.0)
    noise = OpampNoiseConfig()
    ac = simulate_ac(cfg)
    report = build_metrics_report(cfg, noise, engine="python", ac_result=ac)
    bode = tmp_path / "ac_bode.svg"
    bode.write_text("<svg></svg>", encoding="utf-8")
    zplot = plot_impedance(cfg, tmp_path / "impedance.svg")
    md = write_ac_report(
        tmp_path,
        engine="python",
        cfg=cfg,
        noise=noise,
        report=report,
        bode_svg=bode,
        impedance_svg=zplot,
    )
    text = md.read_text(encoding="utf-8")
    assert "![Open-loop Bode" in text
    assert "ac_bode.svg" in text
    assert "impedance.svg" in text
    assert "`gbw_hz`" in text
    assert "`cmrr_db`" in text


def test_engine_report_links_benches(tmp_path: Path) -> None:
    """REPORT.md links AC/STB reports when both exist."""
    cfg = OpampConfig()
    noise = OpampNoiseConfig()
    ac = simulate_ac(cfg)
    report = build_metrics_report(cfg, noise, engine="python", ac_result=ac)
    bode = tmp_path / "ac_bode.svg"
    bode.write_text("<svg></svg>", encoding="utf-8")
    write_ac_report(
        tmp_path,
        engine="python",
        cfg=cfg,
        noise=noise,
        report=report,
        bode_svg=bode,
        impedance_svg=plot_impedance(cfg, tmp_path / "impedance.svg"),
    )
    stb_bode = tmp_path / "stb_bode.svg"
    stb_bode.write_text("<svg></svg>", encoding="utf-8")
    stb_report = build_metrics_report(cfg, noise, engine="python", stb_result=ac)
    write_stb_report(
        tmp_path,
        engine="python",
        cfg=cfg,
        report=stb_report,
        bode_svg=stb_bode,
    )
    engine_md = write_engine_report(tmp_path, engine="python")
    assert engine_md is not None
    text = engine_md.read_text(encoding="utf-8")
    assert "[AC open-loop](AC_REPORT.md)" in text
    assert "[STB loop gain](STB_REPORT.md)" in text


def test_preserve_metrics_sections_keeps_ac() -> None:
    """Re-running STB alone retains prior AC metrics in the merged report."""
    cfg = OpampConfig()
    noise = OpampNoiseConfig()
    ac = simulate_ac(cfg)
    ac_report = build_metrics_report(cfg, noise, engine="python", ac_result=ac)
    stb_only = build_metrics_report(cfg, noise, engine="python", stb_result=ac)
    merged = preserve_metrics_sections(stb_only, ac_report)
    assert merged["ac"]["gbw_hz"]["value"] is not None
    assert merged["stb"]["gbw_hz"]["value"] is not None
