"""ngspice AC testbench generation and execution."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from opamp_model.ac import extract_gbw_phase_margin
from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.core import dominant_pole_rad_s
from opamp_model.io import (
    log_frequency_sweep,
    package_root,
    read_ngspice_ac_wrdata,
    read_ngspice_noise_wrdata,
    write_bode_csv,
)
from opamp_model.model import AcSimulationResult, NoiseSimulationResult
from opamp_model.noise import flicker_voltage_density
from opamp_model.noise_analysis import extract_noise_metrics


class NgspiceNotFoundError(RuntimeError):
    """Raised when the ``ngspice`` executable is not on PATH."""


@dataclass(frozen=True)
class NgspiceAcResult:
    """Artifacts from an ngspice AC run."""

    netlist_path: Path
    wrdata_path: Path
    log_path: Path
    csv_path: Path


def find_ngspice_executable() -> str:
    """Return the ngspice binary path or raise ``NgspiceNotFoundError``."""
    import shutil

    for name in ("ngspice", "ngspice-shared"):
        found = shutil.which(name)
        if found:
            return found
    raise NgspiceNotFoundError(
        "ngspice not found on PATH. Install ngspice or use --simulator python."
    )


def _spectre_value(value: float) -> str:
    """Format a numeric value for SPICE netlists."""
    if value == 0.0:
        return "0"
    if abs(value) >= 1.0e9:
        return f"{value:.12g}"
    return f"{value:.12g}"


# Reference capacitor for the dominant-pole RC (R*C = 1/wp); value only sets R scale.
_CLP_REF_F = 1.0e-12


def render_ngspice_ac_netlist(
    template_path: Path,
    cfg: OpampConfig,
) -> str:
    """Render an ngspice AC netlist with one-pole macromodel parameters.

    Pole placement matches ``dominant_pole_rad_s`` / Spectre ``laplace_nd``:
    ``wp = 2*pi*GBW/A0``, ``Rlp*Clp = 1/wp`` with ``Clp = _CLP_REF_F``.
    """
    text = template_path.read_text(encoding="utf-8")
    a0 = max(cfg.a0_linear, 1.0)
    wp = dominant_pole_rad_s(cfg)
    clp_f = _CLP_REF_F
    rlp_ohm = 1.0 / (wp * clp_f)
    replacements = {
        "a0_db": str(cfg.a0_db),
        "gbw_hz": _spectre_value(cfg.gbw_hz),
        "rin_ohm": _spectre_value(cfg.rin_ohm),
        "rout_ohm": _spectre_value(cfg.rout_ohm),
        "a0_linear": _spectre_value(a0),
        "wp_rad_s": _spectre_value(wp),
        "rlp_ohm": _spectre_value(rlp_ohm),
        "clp_f": _spectre_value(clp_f),
    }
    for key, val in replacements.items():
        text = re.sub(
            rf"^\.param\s+{key}=.*$",
            f".param {key}={val}",
            text,
            flags=re.MULTILINE,
        )
    return (
        text.replace("PLACEHOLDER_A0", _spectre_value(a0))
        .replace("PLACEHOLDER_WP", _spectre_value(wp))
        .replace("PLACEHOLDER_RLP", _spectre_value(rlp_ohm))
        .replace("PLACEHOLDER_CLP", _spectre_value(clp_f))
    )


def run_ngspice_ac(
    cfg: OpampConfig,
    output_dir: Path,
    *,
    template_name: str = "ac_open_loop.cir",
) -> NgspiceAcResult:
    """Run ngspice AC and export Bode data to CSV."""
    repo = package_root()
    template = repo / "testbench" / "ngspice" / template_name
    ng_dir = output_dir / "ngspice"
    logs_dir = output_dir / "logs"
    ng_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    netlist_path = ng_dir / template_name
    netlist_path.write_text(render_ngspice_ac_netlist(template, cfg), encoding="utf-8")
    log_path = logs_dir / f"ngspice_{template_name.replace('.cir', '')}.log"
    wrdata_name = "ac_bode.raw" if "ac_open" in template_name else "stb_bode.raw"
    wrdata_path = ng_dir / wrdata_name
    csv_path = output_dir / "ac_bode.csv"

    executable = find_ngspice_executable()
    completed = subprocess.run(
        [executable, "-b", netlist_path.name],
        cwd=ng_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        msg = f"ngspice failed with code {completed.returncode}; see {log_path}"
        raise RuntimeError(msg)

    if not wrdata_path.is_file():
        msg = f"ngspice did not produce {wrdata_path}"
        raise RuntimeError(msg)

    bode = read_ngspice_ac_wrdata(wrdata_path)
    write_bode_csv(
        csv_path,
        bode["frequency_hz"],
        bode["gain_db"],
        bode["phase_deg"],
    )
    return NgspiceAcResult(
        netlist_path=netlist_path,
        wrdata_path=wrdata_path,
        log_path=log_path,
        csv_path=csv_path,
    )


def ngspice_ac_to_simulation_result(
    frequency_hz: NDArray[np.float64],
    gain_db: NDArray[np.float64],
    phase_deg: NDArray[np.float64],
) -> AcSimulationResult:
    """Build ``AcSimulationResult`` from ngspice AC columns."""
    metrics = extract_gbw_phase_margin(frequency_hz, gain_db, phase_deg)
    return AcSimulationResult(
        frequency_hz=frequency_hz,
        gain_db=gain_db,
        phase_deg=phase_deg,
        metrics=metrics,
    )


def simulate_ac_ngspice(
    cfg: OpampConfig,
    output_dir: Path,
    noise: OpampNoiseConfig | None = None,
) -> AcSimulationResult:
    """Run open-loop AC in ngspice and return aligned metrics."""
    _ = noise
    artifacts = run_ngspice_ac(cfg, output_dir)
    bode = read_ngspice_ac_wrdata(artifacts.wrdata_path)
    return ngspice_ac_to_simulation_result(
        bode["frequency_hz"],
        bode["gain_db"],
        bode["phase_deg"],
    )


@dataclass(frozen=True)
class NgspiceNoiseResult:
    """Artifacts from an ngspice noise + AC run."""

    netlist_path: Path
    noise_wrdata_path: Path
    ac_wrdata_path: Path
    log_path: Path


def _boltzmann_kT_v2_per_hz() -> float:
    """Return ``4*k*T`` at 300 K in V²/Hz (for thermal resistor noise)."""
    return 4.0 * 1.380649e-23 * 300.0


def render_ngspice_noise_netlist(
    template_path: Path,
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
) -> str:
    """Render ngspice open-loop noise netlist (``.noise`` + AC)."""
    text = template_path.read_text(encoding="utf-8")
    a0 = max(cfg.a0_linear, 1.0)
    wp = dominant_pole_rad_s(cfg)
    clp_f = _CLP_REF_F
    rlp_ohm = 1.0 / (wp * clp_f)
    en_white = noise.en_white_v_per_sqrt_hz if noise.enabled else 0.0
    rn_white = (en_white * en_white) / _boltzmann_kT_v2_per_hz() if en_white > 0.0 else 1.0e-6
    replacements = {
        "a0_db": str(cfg.a0_db),
        "gbw_hz": _spectre_value(cfg.gbw_hz),
        "rin_ohm": _spectre_value(cfg.rin_ohm),
        "rout_ohm": _spectre_value(cfg.rout_ohm),
        "en_white": _spectre_value(en_white),
        "a0_linear": _spectre_value(a0),
        "wp_rad_s": _spectre_value(wp),
        "rlp_ohm": _spectre_value(rlp_ohm),
        "clp_f": _spectre_value(clp_f),
        "rn_white": _spectre_value(rn_white),
    }
    for key, val in replacements.items():
        text = re.sub(
            rf"^\.param\s+{key}=.*$",
            f".param {key}={val}",
            text,
            flags=re.MULTILINE,
        )
    return (
        text.replace("PLACEHOLDER_A0", _spectre_value(a0))
        .replace("PLACEHOLDER_WP", _spectre_value(wp))
        .replace("PLACEHOLDER_RLP", _spectre_value(rlp_ohm))
        .replace("PLACEHOLDER_CLP", _spectre_value(clp_f))
        .replace("PLACEHOLDER_RN", _spectre_value(rn_white))
        .replace("PLACEHOLDER_DEC", str(cfg.sweep.points_per_decade))
        .replace("PLACEHOLDER_FSTART", _spectre_value(cfg.sweep.f_start_hz))
        .replace("PLACEHOLDER_FSTOP", _spectre_value(cfg.sweep.f_stop_hz))
    )


def _interp_gain_linear(
    frequency_hz: NDArray[np.float64],
    ac_frequency_hz: NDArray[np.float64],
    gain_db: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate open-loop |A(f)| (linear) onto ``frequency_hz``."""
    log_f = np.log10(np.maximum(frequency_hz, 1.0e-30))
    log_ac = np.log10(np.maximum(ac_frequency_hz, 1.0e-30))
    gain_db_i = np.interp(log_f, log_ac, gain_db)
    return (10.0 ** (gain_db_i / 20.0)).astype(np.float64)


def ngspice_output_noise_spectrum(
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
    *,
    frequency_hz: NDArray[np.float64],
    gain_db: NDArray[np.float64],
    ac_frequency_hz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build output-referred noise density from ngspice AC gain and noise parameters.

    ngspice ``.noise`` on this bench does not amplify input thermal noise through
    ideal VCVS elements; open-loop gain is taken from the companion AC analysis in
    the same netlist. Input-referred density follows ``OpampNoiseConfig`` (white +
    flicker), matching the Python macromodel in ``noise.py``.
    """
    if not noise.enabled:
        return np.zeros_like(frequency_hz, dtype=np.float64)
    a_mag = _interp_gain_linear(frequency_hz, ac_frequency_hz, gain_db)
    white = np.full_like(frequency_hz, noise.en_white_v_per_sqrt_hz, dtype=np.float64)
    flicker = flicker_voltage_density(frequency_hz, noise)
    en_in = np.sqrt(white**2 + flicker**2)
    return (en_in * a_mag).astype(np.float64)


def run_ngspice_noise(
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
    output_dir: Path,
    *,
    template_name: str = "noise_open_loop.cir",
) -> NgspiceNoiseResult:
    """Run ngspice ``.noise`` and companion AC; export ``wrdata`` spectra."""
    repo = package_root()
    template = repo / "testbench" / "ngspice" / template_name
    ng_dir = output_dir / "ngspice"
    logs_dir = output_dir / "logs"
    ng_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    netlist_path = ng_dir / template_name
    netlist_path.write_text(
        render_ngspice_noise_netlist(template, cfg, noise),
        encoding="utf-8",
    )
    log_path = logs_dir / "ngspice_noise.log"
    noise_wrdata_path = ng_dir / "noise_spectrum.raw"
    ac_wrdata_path = ng_dir / "noise_ac.raw"

    executable = find_ngspice_executable()
    completed = subprocess.run(
        [executable, "-b", netlist_path.name],
        cwd=ng_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        msg = f"ngspice noise failed with code {completed.returncode}; see {log_path}"
        raise RuntimeError(msg)
    if not noise_wrdata_path.is_file():
        msg = f"ngspice did not produce {noise_wrdata_path}"
        raise RuntimeError(msg)
    if not ac_wrdata_path.is_file():
        msg = f"ngspice did not produce {ac_wrdata_path}"
        raise RuntimeError(msg)
    return NgspiceNoiseResult(
        netlist_path=netlist_path,
        noise_wrdata_path=noise_wrdata_path,
        ac_wrdata_path=ac_wrdata_path,
        log_path=log_path,
    )


def simulate_noise_ngspice(
    cfg: OpampConfig,
    output_dir: Path,
    noise: OpampNoiseConfig,
) -> NoiseSimulationResult:
    """Run ngspice noise bench; spectrum from AC gain × input-referred noise model."""
    artifacts = run_ngspice_noise(cfg, noise, output_dir)
    _ = read_ngspice_noise_wrdata(artifacts.noise_wrdata_path)
    ac = read_ngspice_ac_wrdata(artifacts.ac_wrdata_path)
    frequency_hz = log_frequency_sweep(
        cfg.sweep.f_start_hz,
        cfg.sweep.f_stop_hz,
        cfg.sweep.points_per_decade,
    )
    spectrum = ngspice_output_noise_spectrum(
        cfg,
        noise,
        frequency_hz=frequency_hz,
        gain_db=ac["gain_db"],
        ac_frequency_hz=ac["frequency_hz"],
    )
    metrics = extract_noise_metrics(cfg, noise, frequency_hz, spectrum)
    return NoiseSimulationResult(
        frequency_hz=frequency_hz,
        noise_v_per_sqrt_hz=spectrum,
        metrics=metrics,
    )


def run_ngspice_tran_stub(
    cfg: OpampConfig,
    output_dir: Path,
    *,
    template_name: str,
    log_name: str,
) -> Path:
    """Run a minimal ngspice TRAN stub (``.op`` only) to validate the toolchain."""
    repo = package_root()
    template = repo / "testbench" / "ngspice" / template_name
    ng_dir = output_dir / "ngspice"
    logs_dir = output_dir / "logs"
    ng_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = ng_dir / template_name
    netlist_path.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    log_path = logs_dir / log_name
    executable = find_ngspice_executable()
    completed = subprocess.run(
        [executable, "-b", netlist_path.name],
        cwd=ng_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        msg = f"ngspice TRAN stub failed with code {completed.returncode}; see {log_path}"
        raise RuntimeError(msg)
    _ = cfg
    return log_path
