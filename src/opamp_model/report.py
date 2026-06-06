"""Markdown reports with embedded figures for each bench and engine."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from opamp_model.cli_helpers import resolve_engine_label
from opamp_model.compare import TOLERANCE_MODULE_REL
from opamp_model.config import OpampConfig, OpampNoiseConfig, TiaConfig
from opamp_model.metrics import MetricEntry, OpampMetricsReport, format_metrics_table
from opamp_model.noise_analysis import NoiseBreakdown
from opamp_model.tran import ThdMetrics


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for report headers."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_value(entry: MetricEntry) -> str:
    """Format one metric value for Markdown tables."""
    val = entry["value"]
    if val is None:
        return "—"
    if entry["unit"] == "Hz" and abs(val) >= 1.0e6:
        return f"{val / 1.0e6:.6g} M"
    if entry["unit"] == "Hz" and abs(val) >= 1.0e3:
        return f"{val / 1.0e3:.6g} k"
    return f"{val:.6g}"


def format_metrics_markdown(
    entries: dict[str, MetricEntry],
    *,
    heading: str,
) -> str:
    """Render one metrics group as a Markdown table."""
    lines = [f"### {heading}", "", "| Metric | Value | Unit | Status |", "| --- | --- | --- | --- |"]
    for name, entry in entries.items():
        lines.append(
            f"| `{name}` | {_format_value(entry)} | {entry['unit']} | {entry['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _figure_block(rel_path: str, caption: str) -> str:
    """Markdown image block using a path relative to the report file."""
    return f"![{caption}]({rel_path})\n\n*{caption}*\n"


def _config_table(cfg: OpampConfig) -> str:
    """Summarize key macromodel parameters."""
    rows = [
        ("A0", f"{cfg.a0_db:.3g}", "dB"),
        ("GBW (param)", f"{cfg.gbw_hz:.6g}", "Hz"),
        ("Rin", f"{cfg.rin_ohm:.6g}", "Ω"),
        ("Cin", f"{cfg.cin_f:.6g}", "F"),
        ("Rout", f"{cfg.rout_ohm:.6g}", "Ω"),
        ("Cout", f"{cfg.cout_f:.6g}", "F"),
        ("CMRR (param)", f"{cfg.cmrr_db:.3g}", "dB"),
        ("PSRR (param)", f"{cfg.psrr_db:.3g}", "dB"),
        ("loop β (STB)", f"{cfg.loop_beta:.6g}", "—"),
    ]
    lines = [
        "### Macromodel parameters",
        "",
        "| Parameter | Value | Unit |",
        "| --- | --- | --- |",
    ]
    for name, val, unit in rows:
        lines.append(f"| {name} | {val} | {unit} |")
    lines.append("")
    return "\n".join(lines)


def write_ac_report(
    output_dir: Path,
    *,
    engine: str,
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
    report: OpampMetricsReport,
    bode_svg: Path,
    impedance_svg: Path,
    cmrr_svg: Path | None = None,
) -> Path:
    """Write ``AC_REPORT.md`` with Bode plot, impedance plot, and metrics."""
    path = output_dir / "AC_REPORT.md"
    engine_label = resolve_engine_label(engine)
    lines = [
        "# AC open-loop bench",
        "",
        f"- **Engine:** {engine_label} (`{engine}`)",
        f"- **Generated:** {_utc_timestamp()}",
        f"- **Noise model:** {'disabled (--ideal)' if not noise.enabled else 'enabled'}",
        "",
        _config_table(cfg),
        "## Figures",
        "",
        _figure_block(bode_svg.name, "Open-loop Bode (gain and phase)"),
        _figure_block(impedance_svg.name, "Input and output impedance vs frequency"),
    ]
    if cmrr_svg is not None and cmrr_svg.is_file():
        lines.append(_figure_block(cmrr_svg.name, "Common-mode gain and CMRR vs frequency"))
    lines.extend(
        [
        "",
        "## Metrics",
        "",
        format_metrics_markdown(report["ac"], heading="AC / open-loop (simulated)"),
        format_metrics_markdown(report["impedance"], heading="Impedance"),
        format_metrics_markdown(report["cmrr_psrr"], heading="CMRR / PSRR"),
        format_metrics_markdown(report["noise"], heading="Noise"),
        format_metrics_markdown(report["large_signal"], heading="Large-signal"),
        "",
        "## Artifacts",
        "",
        "| File | Description |",
        "| --- | --- |",
        f"| `{bode_svg.name}` | Bode plot (SVG) |",
        f"| `{impedance_svg.name}` | Impedance plot (SVG) |",
        ]
    )
    if cmrr_svg is not None:
        lines.append(f"| `{cmrr_svg.name}` | ACM / CMRR plot (SVG) |")
        lines.append("| `cmrr.csv` | Frequency, ACM (dB), CMRR (dB) |")
    lines.extend(
        [
        "| `ac_bode.csv` | Frequency, gain (dB), phase (deg) |",
        "| `opamp_metrics.json` | Scalar metrics bundle |",
        "",
        "> CMRR/PSRR/noise/large-signal entries marked **param** or **planned** "
        "are documented in `docs/metrics_catalog.md`.",
        "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _tia_config_table(tia: TiaConfig) -> str:
    """Summarize TIA feedback parameters."""
    rows = [
        ("Rf", f"{tia.rf_ohm:.6g}", "Ω"),
        ("Cf", f"{tia.cf_f:.6g}", "F"),
        ("Cs", f"{tia.cs_f:.6g}", "F"),
        ("Input", "current" if tia.current_input else "voltage", "—"),
    ]
    lines = [
        "### TIA feedback",
        "",
        "| Parameter | Value | Unit |",
        "| --- | --- | --- |",
    ]
    for name, val, unit in rows:
        lines.append(f"| {name} | {val} | {unit} |")
    lines.append("")
    return "\n".join(lines)


def write_tia_report(
    output_dir: Path,
    *,
    engine: str,
    cfg: OpampConfig,
    tia: TiaConfig,
    report: OpampMetricsReport,
    zt_svg: Path,
) -> Path:
    """Write ``TIA_REPORT.md`` with transimpedance plot and metrics."""
    path = output_dir / "TIA_REPORT.md"
    engine_label = resolve_engine_label(engine)
    unit = "Ω" if tia.current_input else "V/V"
    lines = [
        "# TIA AC bench",
        "",
        f"- **Engine:** {engine_label} (`{engine}`)",
        f"- **Generated:** {_utc_timestamp()}",
        f"- **Transfer:** Vout / Iin ({unit})",
        "",
        _config_table(cfg),
        _tia_config_table(tia),
        "## Figures",
        "",
        _figure_block(zt_svg.name, f"Closed-loop transimpedance |Zt| ({unit})"),
        "",
        "## Metrics",
        "",
        format_metrics_markdown(report.get("tia", {}), heading="TIA (simulated)"),
        "",
        "## Artifacts",
        "",
        "| File | Description |",
        "| --- | --- |",
        f"| `{zt_svg.name}` | Transimpedance plot (SVG) |",
        "| `tia_zt.csv` | Frequency, |Zt|, Zt (dB), phase |",
        "| `opamp_metrics.json` | Scalar metrics bundle |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_psrr_report(
    output_dir: Path,
    *,
    engine: str,
    cfg: OpampConfig,
    report: OpampMetricsReport,
    psrr_svg: Path,
) -> Path:
    """Write ``PSRR_REPORT.md`` with PSRR plot and metrics."""
    path = output_dir / "PSRR_REPORT.md"
    engine_label = resolve_engine_label(engine)
    lines = [
        "# PSRR bench",
        "",
        f"- **Engine:** {engine_label} (`{engine}`)",
        f"- **Generated:** {_utc_timestamp()}",
        f"- **PSRR pole:** {cfg.psrr_pole_hz:.6g} Hz",
        "",
        _config_table(cfg),
        "## Figures",
        "",
        _figure_block(psrr_svg.name, "PSRR vs frequency"),
        "",
        "## Metrics",
        "",
        format_metrics_markdown(report["cmrr_psrr"], heading="CMRR / PSRR"),
        "",
        "## Artifacts",
        "",
        "| File | Description |",
        "| --- | --- |",
        f"| `{psrr_svg.name}` | PSRR plot (SVG) |",
        "| `psrr.csv` | Frequency, PSRR (dB) |",
        "| `opamp_metrics.json` | Scalar metrics bundle |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _noise_config_summary(noise: OpampNoiseConfig) -> str:
    """Summarize enabled noise parameters for Markdown reports."""
    if not noise.enabled:
        return "Noise disabled (`--ideal` or zero density parameters)."
    corner = noise.en_flicker_corner_hz
    corner_str = f"{corner:.4g} Hz" if np.isfinite(corner) and corner > 0.0 else "—"
    return (
        f"White input density: **{noise.en_white_v_per_sqrt_hz * 1.0e9:.4g} nV/√Hz**; "
        f"flicker @ 1 Hz: **{noise.en_flicker_1hz_v_per_sqrt_hz * 1.0e9:.4g} nV/√Hz** "
        f"(ef={noise.en_flicker_ef:g}); flicker corner: **{corner_str}**; "
        f"seed={noise.noise_seed}."
    )


def write_thd_report(
    output_dir: Path,
    *,
    engine: str,
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
    report: OpampMetricsReport,
    waveform_svg: Path,
    spectrum_svg: Path,
    thd: ThdMetrics | None,
    freq_hz: float,
    amplitude_v: float,
    noise_rms_v: float | None = None,
) -> Path:
    """Write ``THD_REPORT.md`` with waveform, spectrum, and distortion metrics."""
    path = output_dir / "THD_REPORT.md"
    engine_label = resolve_engine_label(engine)
    scaffold = _tran_scaffold_note(engine)
    lines = [
        "# THD / large-signal sine bench",
        "",
        f"- **Engine:** {engine_label} (`{engine}`)",
        f"- **Generated:** {_utc_timestamp()}",
        f"- **Noise model:** {'disabled (--ideal)' if not noise.enabled else 'enabled'}",
        f"- **Sine:** {amplitude_v:.6g} V peak @ {freq_hz:.6g} Hz",
        f"- **Nonlinearity:** nl_a2={cfg.nl_a2:.6g}, nl_a3={cfg.nl_a3:.6g}",
        "",
    ]
    if scaffold:
        lines.append(scaffold)
    lines.extend(
        [
        _config_table(cfg),
        "## Transient noise",
        "",
        _noise_config_summary(noise),
        ]
    )
    if noise_rms_v is not None and np.isfinite(noise_rms_v) and noise_rms_v > 0.0:
        lines.append(
            f"Input-referred transient noise RMS over the simulated window: "
            f"**{noise_rms_v:.4g} V** (dashed traces show the noise-free response)."
        )
    lines.extend(
        [
        "",
        "## Figures",
        "",
        _figure_block(waveform_svg.name, "Output voltage vs time (sine steady state)"),
        _figure_block(spectrum_svg.name, "Output magnitude spectrum"),
        "",
        "## Metrics",
        "",
        format_metrics_markdown(report["large_signal"], heading="Large-signal / distortion"),
        ]
    )
    if thd is not None and np.isfinite(thd["thd_db"]):
        lines.extend(
            [
                "",
                "### Distortion summary",
                "",
                f"- THD: **{thd['thd_db']:.2f} dB**",
                f"- HD2: **{thd['hd2_db']:.2f} dB**",
                f"- HD3: **{thd['hd3_db']:.2f} dB**",
                f"- Fundamental magnitude: {thd['fundamental_v']:.6g} V",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifacts",
            "",
            "| File | Description |",
            "| --- | --- |",
            f"| `{waveform_svg.name}` | Sine response waveform (SVG) |",
            f"| `{spectrum_svg.name}` | Harmonic spectrum (SVG) |",
            "| `thd_waveform.csv` | Time, Vout |",
            "| `opamp_metrics.json` | Scalar metrics bundle |",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_slew_report(
    output_dir: Path,
    *,
    engine: str,
    cfg: OpampConfig,
    report: OpampMetricsReport,
    slew_svg: Path,
    noise: OpampNoiseConfig | None = None,
    noise_rms_v: float | None = None,
    noise_trace_svg: Path | None = None,
) -> Path:
    """Write ``SLEW_REPORT.md`` with step-response plot and slew metrics."""
    path = output_dir / "SLEW_REPORT.md"
    engine_label = resolve_engine_label(engine)
    ls = report["large_signal"]
    scaffold = _tran_scaffold_note(engine)
    lines = [
        "# TRAN slew-rate bench",
        "",
        f"- **Engine:** {engine_label} (`{engine}`)",
        f"- **Generated:** {_utc_timestamp()}",
        f"- **Extraction:** 10–90 % output swing (see `tran.extract_slew_rate`)",
        "",
    ]
    if scaffold:
        lines.append(scaffold)
    lines.extend(
        [
        _config_table(cfg),
        ]
    )
    if noise is not None:
        lines.extend(
            [
                "## Transient noise",
                "",
                _noise_config_summary(noise),
            ]
        )
        if noise_rms_v is not None and np.isfinite(noise_rms_v) and noise_rms_v > 0.0:
            lines.append(
                f"Input-referred noise RMS on the step window: **{noise_rms_v:.4g} V**. "
                "Dashed curves show the noise-free step; solid curves include noise "
                "(clipped to the output swing)."
            )
        lines.append("")
    lines.extend(
        [
        "## Figures",
        "",
        _figure_block(slew_svg.name, "Unity-gain step response (SR+ / SR−)"),
        ]
    )
    if noise_trace_svg is not None and noise_trace_svg.is_file():
        lines.append(_figure_block(noise_trace_svg.name, "Input-referred noise samples"))
    lines.extend(
        [
        "",
        "## Metrics",
        "",
        format_metrics_markdown(ls, heading="Large-signal / slew (measured)"),
        "",
        "## Artifacts",
        "",
        "| File | Description |",
        "| --- | --- |",
        f"| `{slew_svg.name}` | Step response plot (SVG) |",
        ]
    )
    if noise_trace_svg is not None and noise_trace_svg.is_file():
        lines.append(f"| `{noise_trace_svg.name}` | Transient noise trace (SVG) |")
    lines.extend(
        [
        "| `slew_step.csv` | Time, Vout (positive step), Vout (negative step) |",
        "| `opamp_metrics.json` | Scalar metrics bundle |",
        "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_ac_comp_report(
    output_dir: Path,
    *,
    engine: str,
    cfg: OpampConfig,
    report: OpampMetricsReport,
    comp_svg: Path,
    peak_db: float,
    peak_freq_hz: float,
    gbw_hz: float,
) -> Path:
    """Write ``AC_COMP_REPORT.md`` with peaking plot and metrics."""
    path = output_dir / "AC_COMP_REPORT.md"
    engine_label = resolve_engine_label(engine)
    lines = [
        "# AC compensation / gain peaking",
        "",
        f"- **Engine:** {engine_label} (`{engine}`)",
        f"- **Generated:** {_utc_timestamp()}",
        "- **Metric:** `peak_db` — max excess gain over ideal single-pole rolloff "
        f"in [{0.1:.0g}×GBW, {10:.0g}×GBW]",
        "",
        _config_table(cfg),
        "## Summary",
        "",
        f"| Quantity | Value |",
        "| --- | --- |",
        f"| GBW | {gbw_hz:.6g} Hz |",
        f"| Peak excess | {peak_db:.4g} dB |",
        f"| Peak frequency | {peak_freq_hz:.6g} Hz |",
        "",
        "## Figures",
        "",
        _figure_block(comp_svg.name, "Open-loop Bode with GBW and peaking markers"),
        "",
        "## Metrics",
        "",
        format_metrics_markdown(report["ac"], heading="AC / open-loop (incl. peaking)"),
        "",
        "## Artifacts",
        "",
        "| File | Description |",
        "| --- | --- |",
        f"| `{comp_svg.name}` | Peaking Bode plot (SVG) |",
        "| `ac_bode.csv` | Open-loop Bode (from AC sim) |",
        "| `opamp_metrics.json` | Scalar metrics bundle |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _tran_scaffold_note(engine: str) -> str:
    """Explain when TRAN curves still come from the Python macromodel."""
    if engine == "python":
        return ""
    return (
        f"- **TRAN scaffold:** `{engine}` runs a minimal netlist (`.op` only); "
        "waveforms and extracted metrics use the **Python behavioral TRAN** model "
        "until full SPICE macromodel TRAN netlists exist. See `docs/MODEL.md`.\n"
    )


def write_stb_report(
    output_dir: Path,
    *,
    engine: str,
    cfg: OpampConfig,
    report: OpampMetricsReport,
    bode_svg: Path,
) -> Path:
    """Write ``STB_REPORT.md`` with loop-gain Bode and metrics."""
    path = output_dir / "STB_REPORT.md"
    engine_label = resolve_engine_label(engine)
    lines = [
        "# STB / loop-gain bench",
        "",
        f"- **Engine:** {engine_label} (`{engine}`)",
        f"- **Generated:** {_utc_timestamp()}",
        f"- **Feedback factor β:** {cfg.loop_beta:.6g}",
        "",
        _config_table(cfg),
        "## Figures",
        "",
        _figure_block(bode_svg.name, "Loop gain Bode (gain and phase)"),
        "",
        "## Metrics",
        "",
        format_metrics_markdown(report["stb"], heading="STB / loop gain (simulated)"),
    ]
    if report["ac"]:
        lines.append(format_metrics_markdown(report["ac"], heading="AC / open-loop (from prior run)"))
    lines.extend(
        [
            format_metrics_markdown(report["impedance"], heading="Impedance"),
            format_metrics_markdown(report["cmrr_psrr"], heading="CMRR / PSRR"),
            format_metrics_markdown(report["noise"], heading="Noise"),
            format_metrics_markdown(report["large_signal"], heading="Large-signal"),
            "",
            "## Artifacts",
            "",
            "| File | Description |",
            "| --- | --- |",
            f"| `{bode_svg.name}` | Loop Bode plot (SVG) |",
            "| `stb_bode.csv` | Frequency, gain (dB), phase (deg) |",
            "| `opamp_metrics.json` | Scalar metrics bundle |",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_noise_report(
    output_dir: Path,
    *,
    engine: str,
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
    report: OpampMetricsReport,
    spectrum_svg: Path,
    breakdown_svg: Path | None = None,
    breakdown: NoiseBreakdown | None = None,
) -> Path:
    """Write ``NOISE_REPORT.md`` with noise spectrum figure and metrics."""
    path = output_dir / "NOISE_REPORT.md"
    engine_label = resolve_engine_label(engine)
    lines = [
        "# Noise bench",
        "",
        f"- **Engine:** {engine_label} (`{engine}`)",
        f"- **Generated:** {_utc_timestamp()}",
        f"- **Noise model:** {'disabled (--ideal)' if not noise.enabled else 'enabled'}",
        f"- **Analysis band:** {cfg.sweep.f_start_hz:.6g} Hz – {cfg.sweep.f_stop_hz:.6g} Hz",
        "",
        _config_table(cfg),
        "## Noise model",
        "",
        _noise_config_summary(noise),
        "",
    ]
    if breakdown is not None and noise.enabled:
        corner = breakdown["flicker_corner_hz"]
        corner_str = f"{corner:.4g} Hz" if np.isfinite(corner) and corner > 0.0 else "—"
        lines.extend(
            [
                "The breakdown plot separates **white** and **flicker** contributions at the "
                "input and output. Total density follows RSS combination; the vertical marker "
                f"shows the flicker corner (**{corner_str}**) where white and flicker intersect.",
                "",
            ]
        )
    lines.extend(
        [
        "## Figures",
        "",
        _figure_block(spectrum_svg.name, "Output-referred noise spectrum"),
        ]
    )
    if breakdown_svg is not None and breakdown_svg.is_file():
        lines.append(
            _figure_block(
                breakdown_svg.name,
                "White / flicker / total noise density (input and output)",
            )
        )
    lines.extend(
        [
        "",
        "## Metrics",
        "",
        format_metrics_markdown(report["noise"], heading="Noise (simulated)"),
        "",
        "## Artifacts",
        "",
        "| File | Description |",
        "| --- | --- |",
        f"| `{spectrum_svg.name}` | Noise spectrum plot (SVG) |",
        ]
    )
    if breakdown_svg is not None:
        lines.append(f"| `{breakdown_svg.name}` | White/flicker breakdown plot (SVG) |")
        lines.append("| `noise_breakdown.csv` | Per-frequency white/flicker/total densities |")
    lines.extend(
        [
        "| `noise_spectrum.csv` | Frequency, output noise density (V/√Hz) |",
        "| `opamp_metrics.json` | Scalar metrics bundle |",
        "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def read_metrics_json(path: Path) -> OpampMetricsReport | None:
    """Load ``opamp_metrics.json`` if present."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return OpampMetricsReport(**data)


def _merge_reported_metrics(
    current: dict[str, MetricEntry],
    previous: dict[str, MetricEntry],
) -> dict[str, MetricEntry]:
    """Keep prior **reported** entries when the current bench did not simulate them."""
    merged = dict(current)
    for name, entry in previous.items():
        if entry["status"] == "reported" and merged.get(name, {}).get("status") != "reported":
            merged[name] = entry
    return merged


def preserve_metrics_sections(
    report: OpampMetricsReport,
    previous: OpampMetricsReport | None,
) -> OpampMetricsReport:
    """Keep AC/STB/noise sections from a prior bench when re-running one script."""
    if previous is None:
        return report
    merged = dict(report)
    if previous.get("ac"):
        merged["ac"] = _merge_reported_metrics(merged.get("ac", {}), previous["ac"])
    if not merged["stb"] and previous.get("stb"):
        merged["stb"] = previous["stb"]
    if previous.get("noise"):
        merged["noise"] = _merge_reported_metrics(merged.get("noise", {}), previous["noise"])
    if previous.get("cmrr_psrr"):
        merged["cmrr_psrr"] = _merge_reported_metrics(
            merged.get("cmrr_psrr", {}),
            previous["cmrr_psrr"],
        )
    if previous.get("large_signal"):
        merged["large_signal"] = _merge_reported_metrics(
            merged.get("large_signal", {}),
            previous["large_signal"],
        )
    if not merged.get("tia") and previous.get("tia"):
        merged["tia"] = previous["tia"]
    if not merged.get("gm") and previous.get("gm"):
        merged["gm"] = previous["gm"]
    return OpampMetricsReport(**merged)


def write_engine_report(output_dir: Path, *, engine: str) -> Path | None:
    """Write top-level ``REPORT.md`` linking AC/STB reports and figures."""
    ac_report = output_dir / "AC_REPORT.md"
    stb_report = output_dir / "STB_REPORT.md"
    psrr_report = output_dir / "PSRR_REPORT.md"
    noise_report = output_dir / "NOISE_REPORT.md"
    slew_report = output_dir / "SLEW_REPORT.md"
    thd_report = output_dir / "THD_REPORT.md"
    tia_report = output_dir / "TIA_REPORT.md"
    ac_comp_report = output_dir / "AC_COMP_REPORT.md"
    if (
        not ac_report.is_file()
        and not stb_report.is_file()
        and not psrr_report.is_file()
        and not noise_report.is_file()
        and not slew_report.is_file()
        and not thd_report.is_file()
        and not tia_report.is_file()
        and not ac_comp_report.is_file()
    ):
        return None

    engine_label = resolve_engine_label(engine)
    scaffold = _tran_scaffold_note(engine)
    lines = [
        f"# opamp-model — {engine_label}",
        "",
        f"- **Engine:** `{engine}`",
        f"- **Generated:** {_utc_timestamp()}",
        f"- **Output directory:** `{output_dir.name}/`",
        "",
    ]
    if scaffold:
        lines.append(scaffold.rstrip())
        lines.append("")
    lines.extend(
        [
        "## Bench reports",
        "",
        ]
    )
    if ac_report.is_file():
        lines.append("- [AC open-loop](AC_REPORT.md)")
    if ac_comp_report.is_file():
        lines.append("- [AC gain peaking](AC_COMP_REPORT.md)")
    if stb_report.is_file():
        lines.append("- [STB loop gain](STB_REPORT.md)")
    if psrr_report.is_file():
        lines.append("- [PSRR](PSRR_REPORT.md)")
    if noise_report.is_file():
        lines.append("- [Noise](NOISE_REPORT.md)")
    if slew_report.is_file():
        lines.append("- [TRAN slew rate](SLEW_REPORT.md)")
    if thd_report.is_file():
        lines.append("- [THD](THD_REPORT.md)")
    if tia_report.is_file():
        lines.append("- [TIA AC](TIA_REPORT.md)")
    lines.extend(["", "## Figures", ""])
    for name, caption in (
        ("ac_bode.svg", "Open-loop Bode"),
        ("ac_comp.svg", "Gain peaking near GBW"),
        ("impedance.svg", "Impedance"),
        ("cmrr.svg", "ACM / CMRR"),
        ("stb_bode.svg", "Loop-gain Bode"),
        ("psrr.svg", "PSRR"),
        ("noise_spectrum.svg", "Output noise spectrum"),
        ("noise_breakdown.svg", "Noise white/flicker breakdown"),
        ("slew.svg", "Step response / slew"),
        ("slew_noise.svg", "Transient noise on step bench"),
        ("thd_waveform.svg", "THD sine waveform"),
        ("thd_spectrum.svg", "THD harmonic spectrum"),
        ("tia_zt.svg", "TIA transimpedance"),
    ):
        fig = output_dir / name
        if fig.is_file():
            lines.append(_figure_block(name, caption))

    metrics_path = output_dir / "opamp_metrics.json"
    compare_report = output_dir.parent / "COMPARE_REPORT.md"
    if compare_report.is_file():
        lines.extend(
            [
                "## Cross-engine comparison",
                "",
                "Peer spread across simulation modules is summarized in "
                "[COMPARE_REPORT.md](../COMPARE_REPORT.md) "
                f"(default **{TOLERANCE_MODULE_REL * 100:.0g}%** relative tolerance between modules where applicable).",
                "",
            ]
        )
    if metrics_path.is_file():
        report = read_metrics_json(metrics_path)
        if report is not None:
            lines.extend(
                [
                    "## Combined metrics (CLI summary)",
                    "",
                    "```text",
                    format_metrics_table(report).rstrip(),
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "## Data files",
            "",
            "| File | Description |",
            "| --- | --- |",
            "| `opamp_metrics.json` | All scalar metrics |",
            "| `ac_bode.csv` | AC Bode data |",
            "| `stb_bode.csv` | STB Bode data |",
            "| `cmrr.csv` | ACM / CMRR data |",
            "| `psrr.csv` | PSRR data |",
            "| `noise_spectrum.csv` | Noise spectrum data |",
            "| `noise_breakdown.csv` | White/flicker breakdown data |",
            "| `slew_step.csv` | TRAN step data |",
            "| `thd_waveform.csv` | THD sine waveform |",
            "| `tia_zt.csv` | TIA transimpedance data |",
            "| `logs/` | Simulation logs |",
            "",
        ]
    )
    path = output_dir / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_report_stub(output_dir: Path, *, title: str = "opamp-model report") -> Path:
    """Backward-compatible stub; prefer :func:`write_engine_report`."""
    path = output_dir / "REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {title}\n\n"
        "Run `run_ac.py` and `run_stb.py` to generate AC_REPORT.md, STB_REPORT.md, "
        "and figures.\n",
        encoding="utf-8",
    )
    return path
