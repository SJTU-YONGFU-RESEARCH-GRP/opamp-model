"""Aggregate op-amp characterization metrics for reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from opamp_model.ac import AcCompMetrics, AcMetrics
from opamp_model.cm_ps import CmrrSimulationResult, PsrrSimulationResult
from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.gm import GmAcSimulationResult, GmMetrics
from opamp_model.impedance import zin_diff
from opamp_model.impedance import zout as zout_model
from opamp_model.model import AcSimulationResult, NoiseSimulationResult
from opamp_model.noise import flicker_corner_frequency, flicker_voltage_density
from opamp_model.tia import TiaMetrics, TiaSimulationResult
from opamp_model.tran import ThdMetrics, is_thd_ideal

# Sources that do not represent independent SPICE/Spectre extraction for a metric.
PARITY_SKIP_SOURCES = frozenset(
    {
        "python_macromodel",
        "tran_scaffold",
        "hybrid_noise_merge",
        "ac_noise_merge",
        "hybrid_tia_merge",
    }
)


class MetricEntry(TypedDict):
    """One scalar metric with unit and provenance."""

    value: float | None
    unit: str
    status: str
    source: str


class OpampMetricsReport(TypedDict):
    """Full metrics bundle for JSON export."""

    engine: str
    config: dict[str, float | int]
    ac: dict[str, MetricEntry]
    stb: dict[str, MetricEntry]
    impedance: dict[str, MetricEntry]
    cmrr_psrr: dict[str, MetricEntry]
    noise: dict[str, MetricEntry]
    large_signal: dict[str, MetricEntry]
    tia: dict[str, MetricEntry]
    gm: dict[str, MetricEntry]


def _metric(
    value: float | None,
    *,
    unit: str,
    status: str,
    source: str,
) -> MetricEntry:
    """Build one metric entry."""
    out: float | None = None
    if value is not None and np.isfinite(value):
        out = float(value)
    return MetricEntry(value=out, unit=unit, status=status, source=source)


def _config_snapshot(cfg: OpampConfig) -> dict[str, float | int]:
    """Serialize key macromodel parameters for the report."""
    return {
        "a0_db": cfg.a0_db,
        "gbw_hz": cfg.gbw_hz,
        "cmrr_db": cfg.cmrr_db,
        "psrr_db": cfg.psrr_db,
        "rin_ohm": cfg.rin_ohm,
        "cin_f": cfg.cin_f,
        "rout_ohm": cfg.rout_ohm,
        "cout_f": cfg.cout_f,
        "loop_beta": cfg.loop_beta,
        "slew_pos_vps": cfg.slew_pos_vps,
        "slew_neg_vps": cfg.slew_neg_vps,
    }


def _ac_metric_source(engine: str) -> str:
    """Provenance for AC/STB scalars parsed or simulated per engine."""
    if engine == "python":
        return "python_macromodel"
    if engine == "ngspice":
        return "ngspice_wrdata"
    if engine == "spectre":
        return "spectre_psf"
    return "python_macromodel"


def _psrr_metric_source(engine: str) -> str:
    """Provenance for simulated PSRR scalars."""
    if engine == "python":
        return "psrr_macromodel"
    if engine == "ngspice":
        return "ngspice_psrr"
    if engine == "spectre":
        return "spectre_psrr"
    return "psrr_macromodel"


def _noise_metric_source(engine: str) -> str:
    """Provenance for simulated noise scalars."""
    if engine == "python":
        return "python_macromodel"
    if engine == "ngspice":
        return "ngspice_noise_merge"
    if engine == "spectre":
        return "spectre_noise_merge"
    return "python_macromodel"


def _tran_metric_source(engine: str) -> str:
    """Provenance for TRAN slew/THD scalars."""
    if engine == "python":
        return "python_macromodel"
    if engine == "ngspice":
        return "ngspice_tran_wrdata"
    if engine == "spectre":
        return "spectre_tran_psf"
    return "python_macromodel"


def build_ac_metric_entries(ac: AcMetrics, *, engine: str = "python") -> dict[str, MetricEntry]:
    """Map AC/STB extraction results to report entries."""
    source = _ac_metric_source(engine)
    return {
        "a0_db": _metric(ac["a0_db"], unit="dB", status="reported", source=source),
        "gbw_hz": _metric(ac["gbw_hz"], unit="Hz", status="reported", source=source),
        "phase_margin_deg": _metric(
            ac["phase_margin_deg"],
            unit="deg",
            status="reported",
            source=source,
        ),
        "gain_margin_db": _metric(
            ac["gain_margin_db"],
            unit="dB",
            status="reported",
            source=source,
        ),
    }


def merge_ac_comp_entries(
    ac_entries: dict[str, MetricEntry],
    comp: AcCompMetrics,
    *,
    engine: str = "python",
) -> dict[str, MetricEntry]:
    """Add gain-peaking scalars from ``run_ac_comp.py`` into the AC metrics group."""
    source = _ac_metric_source(engine)
    merged = dict(ac_entries)
    merged["peak_db"] = _metric(
        comp["peak_db"],
        unit="dB",
        status="reported",
        source=source,
    )
    merged["peak_freq_hz"] = _metric(
        comp["peak_freq_hz"],
        unit="Hz",
        status="reported",
        source=source,
    )
    return merged


def build_impedance_entries(cfg: OpampConfig, *, freq_hz: float = 1.0) -> dict[str, MetricEntry]:
    """Report |Zin| and |Zout| at a low frequency from macromodel passives."""
    f = np.array([freq_hz])
    zin = float(np.abs(zin_diff(cfg, f)[0]))
    zout_abs = float(np.abs(zout_model(cfg, f)[0]))
    return {
        "zin_ohm": _metric(zin, unit="ohm", status="reported", source="python_macromodel"),
        "zout_ohm": _metric(zout_abs, unit="ohm", status="reported", source="python_macromodel"),
        "rin_ohm": _metric(cfg.rin_ohm, unit="ohm", status="param", source="OpampConfig"),
        "cin_f": _metric(cfg.cin_f, unit="F", status="param", source="OpampConfig"),
        "rout_ohm": _metric(cfg.rout_ohm, unit="ohm", status="param", source="OpampConfig"),
        "cout_f": _metric(cfg.cout_f, unit="F", status="param", source="OpampConfig"),
    }


def build_cmrr_psrr_entries(
    cfg: OpampConfig,
    *,
    engine: str = "python",
    cmrr_result: CmrrSimulationResult | None = None,
    psrr_result: PsrrSimulationResult | None = None,
) -> dict[str, MetricEntry]:
    """CMRR/PSRR scalars from simulation benches when available."""
    cmrr_dc = cmrr_result["cmrr_dc_db"] if cmrr_result else None
    acm_dc = float(cmrr_result["acm_db"][0]) if cmrr_result and len(cmrr_result["acm_db"]) else None
    psrr_dc = psrr_result["psrr_dc_db"] if psrr_result else None

    ac_source = _ac_metric_source(engine)
    cmrr_status = "reported" if cmrr_result is not None else "param"
    if cmrr_result is not None:
        cmrr_source = cmrr_result.get("source", ac_source)
    else:
        cmrr_source = "OpampConfig"
    cmrr_value = cmrr_dc if cmrr_result is not None else cfg.cmrr_db

    psrr_status = "reported" if psrr_result is not None else "param"
    psrr_source = _psrr_metric_source(engine) if psrr_result is not None else "OpampConfig"
    psrr_value = psrr_dc if psrr_result is not None else cfg.psrr_db

    return {
        "cmrr_db": _metric(cmrr_value, unit="dB", status=cmrr_status, source=cmrr_source),
        "psrr_db": _metric(psrr_value, unit="dB", status=psrr_status, source=psrr_source),
        "acm_db": _metric(
            acm_dc,
            unit="dB",
            status="reported" if cmrr_result is not None else "planned",
            source=ac_source,
        ),
        "psrr_vs_f": _metric(
            psrr_dc,
            unit="dB",
            status="reported" if psrr_result is not None else "planned",
            source=_psrr_metric_source(engine) if psrr_result is not None else "run_psrr.py",
        ),
    }


def build_noise_entries(
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
    *,
    engine: str = "python",
    noise_result: NoiseSimulationResult | None = None,
) -> dict[str, MetricEntry]:
    """Noise scalars from config and optional ``run_noise.py`` simulation."""
    _ = cfg
    sim_source = _noise_metric_source(engine)
    corner_hz = flicker_corner_frequency(noise)
    en_flicker_1hz = float(
        flicker_voltage_density(np.array([1.0]), noise)[0]
    )
    entries: dict[str, MetricEntry] = {
        "en_white_v_per_sqrt_hz": _metric(
            noise.en_white_v_per_sqrt_hz,
            unit="V/sqrt(Hz)",
            status="param" if noise.enabled else "ideal",
            source="OpampNoiseConfig",
        ),
        "en_flicker_1hz_v_per_sqrt_hz": _metric(
            en_flicker_1hz if en_flicker_1hz > 0.0 else None,
            unit="V/sqrt(Hz)",
            status="param" if noise.enabled else "ideal",
            source="OpampNoiseConfig",
        ),
        "en_flicker_corner_hz": _metric(
            corner_hz,
            unit="Hz",
            status="param" if noise.enabled and np.isfinite(corner_hz) else "ideal",
            source="OpampNoiseConfig",
        ),
    }
    if noise_result is None:
        entries["integrated_noise_rms_v"] = _metric(
            None,
            unit="V",
            status="planned",
            source=sim_source,
        )
        return entries
    metrics = noise_result["metrics"]
    entries["en_in_spot_1khz_v_per_sqrt_hz"] = _metric(
        metrics["en_in_spot_1khz_v_per_sqrt_hz"],
        unit="V/sqrt(Hz)",
        status="reported",
        source=sim_source,
    )
    entries["en_out_spot_1khz_v_per_sqrt_hz"] = _metric(
        metrics["en_out_spot_1khz_v_per_sqrt_hz"],
        unit="V/sqrt(Hz)",
        status="reported",
        source=sim_source,
    )
    entries["integrated_noise_rms_v"] = _metric(
        metrics["integrated_noise_rms_v"],
        unit="V",
        status="reported",
        source=sim_source,
    )
    return entries


def _tia_metric_source(engine: str) -> str:
    """Provenance for TIA AC scalars parsed or simulated per engine."""
    if engine == "python":
        return "python_macromodel"
    if engine == "ngspice":
        return "ngspice_wrdata"
    if engine == "spectre":
        return "spectre_psf"
    return "python_macromodel"


def _gm_metric_source(engine: str) -> str:
    """Provenance for Gm AC scalars parsed or simulated per engine."""
    return _tia_metric_source(engine)


def build_tia_metric_entries(
    metrics: TiaMetrics,
    *,
    engine: str = "python",
    unit: str = "ohm",
) -> dict[str, MetricEntry]:
    """Map TIA AC extraction results to report entries."""
    source = _tia_metric_source(engine)
    return {
        "zt_dc_ohm": _metric(
            metrics["zt_dc_ohm"],
            unit=unit,
            status="reported",
            source=source,
        ),
        "zt_dc_db": _metric(
            metrics["zt_dc_db"],
            unit="dB",
            status="reported",
            source=source,
        ),
        "bandwidth_hz": _metric(
            metrics["bandwidth_hz"],
            unit="Hz",
            status="reported",
            source=source,
        ),
    }


def build_gm_metric_entries(
    metrics: GmMetrics,
    *,
    engine: str = "python",
) -> dict[str, MetricEntry]:
    """Map Gm AC extraction results to report entries."""
    source = _gm_metric_source(engine)
    return {
        "gm_s": _metric(metrics["gm_s"], unit="S", status="reported", source=source),
        "rout_ohm": _metric(metrics["rout_ohm"], unit="ohm", status="reported", source=source),
        "gain_db": _metric(metrics["gain_db"], unit="dB", status="reported", source=source),
        "bandwidth_hz": _metric(
            metrics["bandwidth_hz"],
            unit="Hz",
            status="reported",
            source=source,
        ),
    }


def build_large_signal_entries(
    cfg: OpampConfig,
    *,
    engine: str = "python",
    thd: ThdMetrics | None = None,
    ideal_flag: bool = False,
    slew_pos_measured: float | None = None,
    slew_neg_measured: float | None = None,
) -> dict[str, MetricEntry]:
    """Large-signal parameters; measured by TRAN benches when available."""
    tran_source = _tran_metric_source(engine)
    if slew_pos_measured is not None and np.isfinite(slew_pos_measured):
        slew_pos = _metric(
            slew_pos_measured,
            unit="V/s",
            status="reported",
            source=tran_source,
        )
    else:
        slew_pos = _metric(
            cfg.slew_pos_vps,
            unit="V/s",
            status="param",
            source="OpampConfig",
        )
    if slew_neg_measured is not None and np.isfinite(slew_neg_measured):
        slew_neg = _metric(
            slew_neg_measured,
            unit="V/s",
            status="reported",
            source=tran_source,
        )
    else:
        slew_neg = _metric(
            cfg.slew_neg_vps,
            unit="V/s",
            status="param",
            source="OpampConfig",
        )

    if is_thd_ideal(cfg, ideal_flag=ideal_flag):
        thd_status = "ideal"
        thd_value = None
        hd2_value = None
        hd3_value = None
    elif thd is not None:
        thd_status = "reported"
        thd_value = thd["thd_db"]
        hd2_value = thd["hd2_db"]
        hd3_value = thd["hd3_db"]
    else:
        thd_status = "planned"
        thd_value = None
        hd2_value = None
        hd3_value = None

    return {
        "slew_pos_vps": slew_pos,
        "slew_neg_vps": slew_neg,
        "thd_db": _metric(thd_value, unit="dB", status=thd_status, source=tran_source),
        "hd2_db": _metric(hd2_value, unit="dB", status=thd_status, source=tran_source),
        "hd3_db": _metric(hd3_value, unit="dB", status=thd_status, source=tran_source),
    }


def build_metrics_report(
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
    *,
    engine: str,
    ac_result: AcSimulationResult | None = None,
    stb_result: AcSimulationResult | None = None,
    noise_result: NoiseSimulationResult | None = None,
    cmrr_result: CmrrSimulationResult | None = None,
    psrr_result: PsrrSimulationResult | None = None,
    thd: ThdMetrics | None = None,
    ideal_flag: bool = False,
    slew_pos_measured: float | None = None,
    slew_neg_measured: float | None = None,
    tia_result: TiaSimulationResult | None = None,
    gm_result: GmAcSimulationResult | None = None,
) -> OpampMetricsReport:
    """Assemble a full metrics report from available simulations and parameters."""
    ac_metrics = ac_result["metrics"] if ac_result else None
    stb_metrics = stb_result["metrics"] if stb_result else ac_metrics
    ac_entries = build_ac_metric_entries(ac_metrics, engine=engine) if ac_metrics else {}
    stb_entries = build_ac_metric_entries(stb_metrics, engine=engine) if stb_metrics else {}
    tia_entries = (
        build_tia_metric_entries(tia_result["metrics"], engine=engine, unit="ohm")
        if tia_result is not None
        else {}
    )
    gm_entries = (
        build_gm_metric_entries(gm_result["metrics"], engine=engine)
        if gm_result is not None
        else {}
    )
    return OpampMetricsReport(
        engine=engine,
        config=_config_snapshot(cfg),
        ac=ac_entries,
        stb=stb_entries,
        impedance=build_impedance_entries(cfg),
        cmrr_psrr=build_cmrr_psrr_entries(
            cfg,
            engine=engine,
            cmrr_result=cmrr_result,
            psrr_result=psrr_result,
        ),
        noise=build_noise_entries(
            cfg,
            noise,
            engine=engine,
            noise_result=noise_result,
        ),
        large_signal=build_large_signal_entries(
            cfg,
            engine=engine,
            thd=thd,
            ideal_flag=ideal_flag,
            slew_pos_measured=slew_pos_measured,
            slew_neg_measured=slew_neg_measured,
        ),
        tia=tia_entries,
        gm=gm_entries,
    )


def write_metrics_json(path: Path, report: OpampMetricsReport) -> Path:
    """Write ``opamp_metrics.json`` with all metric groups."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def format_metrics_table(report: OpampMetricsReport) -> str:
    """Return a plain-text summary for CLI printing."""
    lines = [f"Engine: {report['engine']}", ""]

    def section(title: str, entries: dict[str, MetricEntry]) -> None:
        lines.append(title)
        for name, entry in entries.items():
            val = entry["value"]
            val_str = f"{val:.6g}" if val is not None else "—"
            lines.append(
                f"  {name}: {val_str} {entry['unit']}  "
                f"[{entry['status']}] ({entry['source']})"
            )
        lines.append("")

    section("AC / open-loop", report["ac"])
    section("STB / loop gain", report["stb"])
    section("Impedance", report["impedance"])
    section("CMRR / PSRR", report["cmrr_psrr"])
    section("Noise", report["noise"])
    section("Large-signal", report["large_signal"])
    if report.get("tia"):
        section("TIA", report["tia"])
    if report.get("gm"):
        section("Gm", report["gm"])
    return "\n".join(lines)
