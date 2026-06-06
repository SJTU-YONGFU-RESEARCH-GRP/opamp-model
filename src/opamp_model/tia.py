"""Transimpedance amplifier closed-loop small-signal model."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from opamp_model.ac import _interp_crossing_frequency
from opamp_model.config import OpampConfig, OpampNoiseConfig, TiaConfig
from opamp_model.core import bode_from_transfer, open_loop_transfer
from opamp_model.io import log_frequency_sweep
from opamp_model.plot_style import (
    FIGSIZE,
    LINE_COLORS,
    LINEWIDTH_MAIN,
    apply_rcparams,
    apply_style,
)


class TiaMetrics(TypedDict):
    """Scalar TIA AC summary metrics."""

    zt_dc_ohm: float
    zt_dc_db: float
    bandwidth_hz: float


class TiaSimulationResult(TypedDict):
    """TIA AC simulation result bundle."""

    frequency_hz: NDArray[np.float64]
    zt_ohm: NDArray[np.float64]
    zt_db: NDArray[np.float64]
    zt_phase_deg: NDArray[np.float64]
    metrics: TiaMetrics


def feedback_impedance(
    tia: TiaConfig,
    frequency_hz: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Return feedback impedance ``Zf = Rf || Cf``."""
    omega = 2.0 * np.pi * frequency_hz.astype(np.float64)
    s = 1j * omega
    zf = tia.rf_ohm / (1.0 + s * tia.rf_ohm * tia.cf_f)
    return zf.astype(np.complex128)


def input_shunt_impedance(
    opamp: OpampConfig,
    tia: TiaConfig,
    frequency_hz: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Return shunt impedance at the summing node (``Rin || (Cin + Cs)``)."""
    omega = 2.0 * np.pi * frequency_hz.astype(np.float64)
    s = 1j * omega
    c_total = opamp.cin_f + tia.cs_f
    y = 1.0 / opamp.rin_ohm + s * c_total
    return (1.0 / y).astype(np.complex128)


def closed_loop_transimpedance(
    opamp: OpampConfig,
    tia: TiaConfig,
    frequency_hz: NDArray[np.float64],
    noise: OpampNoiseConfig | None = None,
) -> NDArray[np.complex128]:
    """Return closed-loop transimpedance ``Zt(s) = Vout / Iin`` (Ω).

    Inverting current-input TIA with feedback ``Zf`` and input shunt ``Zin``:

    ``Zt = -A(s) * Zf / (1 + A(s) + Zf / Zin)``

    When ``current_input`` is False, returns inverting voltage gain ``Vout / Vin``
    in V/V: ``Acl = -A * Zf / ((1 + A) * Zin + Zf)``.
    """
    _ = noise
    aol = open_loop_transfer(opamp, frequency_hz, noise)
    zf = feedback_impedance(tia, frequency_hz)
    zin = input_shunt_impedance(opamp, tia, frequency_hz)

    if tia.current_input:
        zt = -aol * zf / (1.0 + aol + zf / zin)
    else:
        zt = -aol * zf / ((1.0 + aol) * zin + zf)
    return zt.astype(np.complex128)


def extract_tia_metrics(
    frequency_hz: NDArray[np.float64],
    zt_db: NDArray[np.float64],
) -> TiaMetrics:
    """Extract DC transimpedance magnitude and −3 dB bandwidth of ``|Zt|``."""
    zt_db_arr = zt_db.astype(np.float64)
    zt_dc_db = float(zt_db_arr[0]) if len(zt_db_arr) else float("nan")
    zt_dc_ohm = float(10.0 ** (zt_dc_db / 20.0)) if np.isfinite(zt_dc_db) else float("nan")
    target_db = zt_dc_db - 3.0
    bandwidth_hz = _interp_crossing_frequency(frequency_hz, zt_db_arr, target_db)
    return TiaMetrics(
        zt_dc_ohm=zt_dc_ohm,
        zt_dc_db=zt_dc_db,
        bandwidth_hz=bandwidth_hz,
    )


def bode_to_tia_result(
    frequency_hz: NDArray[np.float64],
    gain_db: NDArray[np.float64],
    phase_deg: NDArray[np.float64],
) -> TiaSimulationResult:
    """Build a TIA result from AC columns (Iin AC = 1 A → |Vout| in Ω)."""
    zt_db = gain_db.astype(np.float64)
    zt_ohm = (10.0 ** (zt_db / 20.0)).astype(np.float64)
    zt_phase_deg = phase_deg.astype(np.float64)
    metrics = extract_tia_metrics(frequency_hz, zt_db)
    return TiaSimulationResult(
        frequency_hz=frequency_hz.astype(np.float64),
        zt_ohm=zt_ohm,
        zt_db=zt_db,
        zt_phase_deg=zt_phase_deg,
        metrics=metrics,
    )


def simulate_tia_ac(
    opamp: OpampConfig,
    tia: TiaConfig,
    noise: OpampNoiseConfig | None = None,
) -> TiaSimulationResult:
    """Run TIA AC simulation (Python macromodel)."""
    f = log_frequency_sweep(
        opamp.sweep.f_start_hz,
        opamp.sweep.f_stop_hz,
        opamp.sweep.points_per_decade,
    )
    zt = closed_loop_transimpedance(opamp, tia, f, noise)
    zt_ohm = np.abs(zt).astype(np.float64)
    zt_db, zt_phase_deg = bode_from_transfer(zt)
    metrics = extract_tia_metrics(f, zt_db)
    return TiaSimulationResult(
        frequency_hz=f,
        zt_ohm=zt_ohm,
        zt_db=zt_db,
        zt_phase_deg=zt_phase_deg,
        metrics=metrics,
    )


def plot_zt(
    result: TiaSimulationResult,
    output_path: Path,
    *,
    title: str = "TIA transimpedance",
    unit: str = "Ω",
) -> Path:
    """Plot ``|Zt|`` and phase vs frequency and write an SVG."""
    apply_rcparams()
    f = result["frequency_hz"]
    metrics = result["metrics"]
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    ax_mag.semilogx(
        f,
        result["zt_db"],
        color=LINE_COLORS["gain"],
        linewidth=LINEWIDTH_MAIN,
    )
    ylabel = f"|Zt| (dB re 1 {unit})" if unit == "Ω" else f"|Zt| (dB re 1 {unit})"
    ax_mag.set_ylabel(ylabel, fontsize=13)
    subtitle = (
        f"|Zt|(DC)={metrics['zt_dc_ohm']:.3g} {unit}  "
        f"BW−3dB={metrics['bandwidth_hz']:.3g} Hz"
    )
    ax_mag.set_title(f"{title}\n{subtitle}", fontsize=14)
    if np.isfinite(metrics["bandwidth_hz"]):
        ax_mag.axvline(metrics["bandwidth_hz"], color="#888888", linewidth=1.0, linestyle=":")
        ax_mag.axhline(metrics["zt_dc_db"] - 3.0, color="#888888", linewidth=1.0, linestyle="--")

    ax_phase.semilogx(
        f,
        result["zt_phase_deg"],
        color=LINE_COLORS["phase"],
        linewidth=LINEWIDTH_MAIN,
    )
    ax_phase.set_ylabel("Phase (deg)", fontsize=13)
    ax_phase.set_xlabel("Frequency (Hz)", fontsize=13)

    apply_style(ax_mag)
    apply_style(ax_phase)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path
