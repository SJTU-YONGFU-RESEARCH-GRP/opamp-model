"""Cross-engine comparison of ``opamp_metrics.json`` bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from opamp_model.io import package_root
from opamp_model.metrics import OpampMetricsReport

DEFAULT_ENGINES = ("python", "ngspice", "spectre")
METRICS_JSON_NAME = "opamp_metrics.json"

# Peer-engine spread limits (cross-module / cross-engine check in COMPARE_REPORT.md).
TOLERANCE_A0_DB = 0.1
TOLERANCE_GBW_REL = 0.02
TOLERANCE_PM_DEG = 2.0
TOLERANCE_NOISE_REL = 0.02
TOLERANCE_MODULE_REL = 0.02  # default 2% between python / ngspice / spectre

SpreadKind = Literal["absolute", "relative", "none"]


@dataclass(frozen=True)
class MetricSpec:
    """One scalar compared across engines."""

    key: str
    label: str
    section: str
    field: str
    unit: str
    spread_kind: SpreadKind
    tolerance: float | None = None  # absolute or relative per spread_kind


COMPARE_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("a0_db", "A0", "ac", "a0_db", "dB", "absolute", TOLERANCE_A0_DB),
    MetricSpec("gbw_hz", "GBW", "ac", "gbw_hz", "Hz", "relative", TOLERANCE_GBW_REL),
    MetricSpec(
        "phase_margin_deg",
        "Phase margin",
        "stb",
        "phase_margin_deg",
        "deg",
        "absolute",
        TOLERANCE_PM_DEG,
    ),
    MetricSpec("cmrr_db", "CMRR", "cmrr_psrr", "cmrr_db", "dB", "absolute", None),
    MetricSpec("psrr_db", "PSRR", "cmrr_psrr", "psrr_db", "dB", "absolute", None),
    MetricSpec(
        "integrated_noise_rms_v",
        "Integrated noise RMS",
        "noise",
        "integrated_noise_rms_v",
        "V",
        "relative",
        TOLERANCE_NOISE_REL,
    ),
    MetricSpec(
        "slew_pos_vps",
        "Slew (+)",
        "large_signal",
        "slew_pos_vps",
        "V/s",
        "relative",
        TOLERANCE_MODULE_REL,
    ),
    MetricSpec(
        "slew_neg_vps",
        "Slew (−)",
        "large_signal",
        "slew_neg_vps",
        "V/s",
        "relative",
        TOLERANCE_MODULE_REL,
    ),
)

GOLDEN_KEY_MAP: dict[str, str] = {
    "a0_db": "a0_db",
    "gbw_hz": "gbw_hz",
    "phase_margin_deg": "phase_margin_deg",
    "cmrr_db": "cmrr_db",
    "psrr_db": "psrr_db",
}


@dataclass
class MetricRow:
    """Per-metric values and spread across engines."""

    spec: MetricSpec
    engine_values: dict[str, float | None]
    golden_value: float | None
    spread: float | None
    spread_display: str
    within_tolerance: bool | None  # None when not checked or insufficient data


@dataclass
class CompareResult:
    """Outcome of a cross-engine comparison."""

    rows: list[MetricRow]
    engines: tuple[str, ...]
    golden_path: Path | None
    passed: bool
    failures: list[str]


def _metric_value(report: OpampMetricsReport, section: str, field: str) -> float | None:
    """Extract a finite scalar from a metrics report section."""
    block = report.get(section)  # type: ignore[call-overload]
    if not isinstance(block, dict):
        return None
    entry = block.get(field)
    if not isinstance(entry, dict):
        return None
    raw = entry.get("value")
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not (val == val):  # NaN
        return None
    return val


def load_engine_metrics(output_root: Path, engine: str) -> OpampMetricsReport | None:
    """Load ``outputs/<engine>/opamp_metrics.json`` if it exists."""
    path = output_root / engine / METRICS_JSON_NAME
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return OpampMetricsReport(**data)


def load_golden_metrics(path: Path | None) -> dict[str, Any] | None:
    """Load optional reference targets from YAML."""
    if path is None or not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else None


def _golden_scalar(golden: dict[str, Any] | None, spec: MetricSpec) -> float | None:
    """Map golden YAML keys to compare-metric units."""
    if golden is None:
        return None
    if spec.key == "slew_pos_vps":
        raw = golden.get("slew_v_per_us")
        return float(raw) * 1.0e6 if raw is not None else None
    if spec.key == "slew_neg_vps":
        raw = golden.get("slew_v_per_us")
        return -float(raw) * 1.0e6 if raw is not None else None
    yaml_key = GOLDEN_KEY_MAP.get(spec.key)
    if yaml_key is None:
        return None
    raw = golden.get(yaml_key)
    if raw is None:
        return None
    return float(raw)


def compute_spread(values: list[float], *, kind: SpreadKind) -> float | None:
    """Return spread across finite engine values (needs at least two)."""
    if len(values) < 2:
        return None
    lo = min(values)
    hi = max(values)
    if kind == "absolute":
        return hi - lo
    if kind == "relative":
        mid = 0.5 * (hi + lo)
        if mid == 0.0:
            return None
        return (hi - lo) / abs(mid)
    return None


def _format_spread(spread: float | None, spec: MetricSpec) -> str:
    if spread is None:
        return "—"
    if spec.spread_kind == "relative":
        return f"{spread * 100:.4g}%"
    if spec.unit == "dB":
        return f"{spread:.4g} dB"
    if spec.unit == "deg":
        return f"{spread:.4g}°"
    return f"{spread:.6g} {spec.unit}"


def _format_value(val: float | None, spec: MetricSpec) -> str:
    if val is None:
        return "—"
    if spec.unit == "Hz" and abs(val) >= 1.0e6:
        return f"{val / 1.0e6:.6g} M"
    if spec.unit == "Hz" and abs(val) >= 1.0e3:
        return f"{val / 1.0e3:.6g} k"
    return f"{val:.6g}"


def _format_tolerance_limit(spec: MetricSpec) -> str:
    """Human-readable peer spread limit for one metric."""
    if spec.tolerance is None:
        return "—"
    if spec.spread_kind == "relative":
        return f"{spec.tolerance * 100:.0g}%"
    if spec.unit == "dB":
        return f"{spec.tolerance:g} dB"
    if spec.unit == "deg":
        return f"{spec.tolerance:g}°"
    return f"{spec.tolerance:g} {spec.unit}"


def _check_tolerance(spread: float | None, spec: MetricSpec) -> bool | None:
    if spec.tolerance is None or spread is None:
        return None
    return spread <= spec.tolerance


def compare_engines(
    output_root: Path,
    *,
    engines: tuple[str, ...] = DEFAULT_ENGINES,
    golden_path: Path | None = None,
) -> CompareResult:
    """Compare metrics across peer engines under ``output_root``."""
    if golden_path is None:
        golden_path = package_root() / "docs" / "golden_metrics.yaml"
    golden = load_golden_metrics(golden_path if golden_path.is_file() else None)

    reports: dict[str, OpampMetricsReport] = {}
    for engine in engines:
        report = load_engine_metrics(output_root, engine)
        if report is not None:
            reports[engine] = report

    rows: list[MetricRow] = []
    failures: list[str] = []

    for spec in COMPARE_METRICS:
        engine_values: dict[str, float | None] = {}
        for engine in engines:
            report = reports.get(engine)
            engine_values[engine] = (
                _metric_value(report, spec.section, spec.field) if report else None
            )

        finite = [v for v in engine_values.values() if v is not None]
        spread = compute_spread(finite, kind=spec.spread_kind)
        within = _check_tolerance(spread, spec)
        if within is False and spec.tolerance is not None:
            limit = (
                f"{spec.tolerance * 100:.0g}%"
                if spec.spread_kind == "relative"
                else f"{spec.tolerance:g} {spec.unit}"
            )
            failures.append(
                f"{spec.label}: spread {_format_spread(spread, spec)} exceeds limit {limit}"
            )

        rows.append(
            MetricRow(
                spec=spec,
                engine_values=engine_values,
                golden_value=_golden_scalar(golden, spec),
                spread=spread,
                spread_display=_format_spread(spread, spec),
                within_tolerance=within,
            )
        )

    return CompareResult(
        rows=rows,
        engines=engines,
        golden_path=golden_path if golden is not None else None,
        passed=len(failures) == 0,
        failures=failures,
    )


def _status_label(within_tolerance: bool | None) -> str:
    if within_tolerance is None:
        return "—"
    if within_tolerance:
        return "ok"
    return "FAIL"


def format_compare_table(result: CompareResult) -> str:
    """Render a plain-text comparison table."""
    engines = result.engines
    has_golden = any(row.golden_value is not None for row in result.rows)
    header = ["Metric", *engines]
    if has_golden:
        header.append("ref")
    header.extend(["Spread", "Limit", "Status"])
    col_w = [max(len(h), 12) for h in header]

    def pad(cols: list[str]) -> str:
        return "  ".join(s.ljust(col_w[i]) for i, s in enumerate(cols))

    lines = [pad(header), pad(["-" * w for w in col_w])]

    for row in result.rows:
        cols = [row.spec.label]
        for engine in engines:
            cols.append(_format_value(row.engine_values.get(engine), row.spec))
        if has_golden:
            cols.append(_format_value(row.golden_value, row.spec))
        cols.append(row.spread_display)
        cols.append(_format_tolerance_limit(row.spec))
        cols.append(_status_label(row.within_tolerance))
        lines.append(pad(cols))

    lines.append("")
    if result.failures:
        lines.append("Tolerance failures:")
        for msg in result.failures:
            lines.append(f"  - {msg}")
    else:
        lines.append(
            "All checked spreads within peer-engine limits "
            f"(default {TOLERANCE_MODULE_REL * 100:.0g}% between modules where applicable)."
        )
    return "\n".join(lines)


def format_compare_markdown(result: CompareResult) -> str:
    """Render a Markdown cross-engine comparison report."""
    engines = result.engines
    has_golden = any(row.golden_value is not None for row in result.rows)
    header = ["Metric", *engines]
    if has_golden:
        header.append("ref")
    header.extend(["Spread", "Limit", "Status"])

    lines = [
        "# Cross-engine comparison",
        "",
        "Peer spread across simulation modules "
        f"(`{'`, `'.join(engines)}`). Relative metrics use a **{TOLERANCE_MODULE_REL * 100:.0g}%** "
        "tolerance between modules unless noted otherwise.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]

    for row in result.rows:
        cols = [row.spec.label]
        for engine in engines:
            cols.append(_format_value(row.engine_values.get(engine), row.spec))
        if has_golden:
            cols.append(_format_value(row.golden_value, row.spec))
        cols.extend(
            [
                row.spread_display,
                _format_tolerance_limit(row.spec),
                _status_label(row.within_tolerance),
            ]
        )
        lines.append("| " + " | ".join(cols) + " |")

    lines.extend(["", "## Tolerance limits", ""])
    for spec in COMPARE_METRICS:
        limit = _format_tolerance_limit(spec)
        if limit == "—":
            continue
        lines.append(f"- **{spec.label}:** {limit}")

    lines.append("")
    if result.failures:
        lines.extend(["## Failures", ""])
        for msg in result.failures:
            lines.append(f"- {msg}")
    else:
        lines.append(
            "All checked spreads are within peer-engine limits "
            f"(default {TOLERANCE_MODULE_REL * 100:.0g}% between modules where applicable)."
        )
    lines.append("")
    return "\n".join(lines)


def write_compare_report(output_root: Path, result: CompareResult) -> Path:
    """Write ``COMPARE_REPORT.md`` under ``output_root``."""
    path = output_root / "COMPARE_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_compare_markdown(result), encoding="utf-8")
    return path
