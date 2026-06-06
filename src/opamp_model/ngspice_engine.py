"""ngspice AC testbench generation and execution."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from opamp_model.ac import cmrr_from_aol_and_acm, extract_gbw_phase_margin
from opamp_model.cm_ps import PsrrSimulationResult, psrr_db_from_transfer
from opamp_model.config import BenchSweepConfig, GmConfig, OpampConfig, OpampNoiseConfig, TiaConfig
from opamp_model.core import dominant_pole_rad_s
from opamp_model.io import (
    log_frequency_sweep,
    package_root,
    read_ngspice_ac_wrdata,
    read_ngspice_noise_wrdata,
    read_ngspice_psrr_wrdata,
    write_bode_csv,
    write_psrr_csv,
)
from opamp_model.cm_ps import CmrrSimulationResult
from opamp_model.gm import GmAcSimulationResult, bode_to_gm_result
from opamp_model.model import AcSimulationResult, NoiseSimulationResult
from opamp_model.tia import TiaSimulationResult, bode_to_tia_result
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


def _ngspice_fp2_fz_blocks(cfg: OpampConfig) -> tuple[str, str, str]:
    """Return (fp2_block, fz_block, out_node) for optional pole/zero stages.

    Equations (match ``core.py`` / VA):
      second pole: ``1 / (1 + s/wp2)`` via ``Rfp2*Cfp2 = 1/wp2``
      zero: ``(1 + s/wz)`` via B-source ``laplace(V, {wz, 1}, {wz, 0})``
    """
    if cfg.fp2_hz > 0.0:
        wp2 = 2.0 * np.pi * cfg.fp2_hz
        cfp2_f = _CLP_REF_F
        rfp2_ohm = 1.0 / (wp2 * cfp2_f)
        fp2_block = (
            f"Rfp2 n2 n3 {_spectre_value(rfp2_ohm)}\n"
            f"Cfp2 n3 0 {_spectre_value(cfp2_f)}"
        )
        pole_out = "n3"
    else:
        fp2_block = "Efp2byp n3 0 n2 0 1"
        pole_out = "n3"

    if cfg.fz_hz > 0.0:
        wz = 2.0 * np.pi * cfg.fz_hz
        fz_block = f"Bfz n4 0 V=laplace(V({pole_out}), {{{_spectre_value(wz)}, 1}}, {{{_spectre_value(wz)}, 0}})"
        out_node = "n4"
    else:
        fz_block = f"Efzbyp n4 0 {pole_out} 0 1"
        out_node = "n4"
    return fp2_block, fz_block, out_node


def _apply_ngspice_param_replacements(text: str, replacements: dict[str, str]) -> str:
    for key, val in replacements.items():
        text = re.sub(
            rf"^\.param\s+{key}=.*$",
            f".param {key}={val}",
            text,
            flags=re.MULTILINE,
        )
    return text


def render_ngspice_ac_netlist(
    template_path: Path,
    cfg: OpampConfig,
) -> str:
    """Render an ngspice AC netlist with macromodel parameters.

    Pole placement matches ``dominant_pole_rad_s`` / Spectre ``laplace_nd``:
    ``wp = 2*pi*GBW/A0``, ``Rlp*Clp = 1/wp`` with ``Clp = _CLP_REF_F``.
    Optional ``fp2_hz`` / ``fz_hz`` add a second RC pole and B-source zero.
    """
    text = template_path.read_text(encoding="utf-8")
    a0 = max(cfg.a0_linear, 1.0)
    wp = dominant_pole_rad_s(cfg)
    clp_f = _CLP_REF_F
    rlp_ohm = 1.0 / (wp * clp_f)
    fp2_block, fz_block, out_node = _ngspice_fp2_fz_blocks(cfg)
    replacements = {
        "a0_db": str(cfg.a0_db),
        "gbw_hz": _spectre_value(cfg.gbw_hz),
        "fp2_hz": _spectre_value(cfg.fp2_hz),
        "fz_hz": _spectre_value(cfg.fz_hz),
        "rin_ohm": _spectre_value(cfg.rin_ohm),
        "rout_ohm": _spectre_value(cfg.rout_ohm),
        "a0_linear": _spectre_value(a0),
        "wp_rad_s": _spectre_value(wp),
        "rlp_ohm": _spectre_value(rlp_ohm),
        "clp_f": _spectre_value(clp_f),
    }
    text = _apply_ngspice_param_replacements(text, replacements)
    return (
        text.replace("PLACEHOLDER_A0", _spectre_value(a0))
        .replace("PLACEHOLDER_WP", _spectre_value(wp))
        .replace("PLACEHOLDER_RLP", _spectre_value(rlp_ohm))
        .replace("PLACEHOLDER_CLP", _spectre_value(clp_f))
        .replace("PLACEHOLDER_FP2_BLOCK", fp2_block)
        .replace("PLACEHOLDER_FZ_BLOCK", fz_block)
        .replace("PLACEHOLDER_OUT_NODE", out_node)
    )


def render_ngspice_cm_netlist(
    template_path: Path,
    cfg: OpampConfig,
) -> str:
    """Render ngspice common-mode AC netlist (``ACM = Aol / CMRR_linear``)."""
    text = template_path.read_text(encoding="utf-8")
    a0 = max(cfg.a0_linear, 1.0)
    cmrr_linear = max(cfg.cmrr_linear, 1.0)
    acm_gain = a0 / cmrr_linear
    wp = dominant_pole_rad_s(cfg)
    clp_f = _CLP_REF_F
    rlp_ohm = 1.0 / (wp * clp_f)
    fp2_block, fz_block, out_node = _ngspice_fp2_fz_blocks(cfg)
    replacements = {
        "a0_db": str(cfg.a0_db),
        "gbw_hz": _spectre_value(cfg.gbw_hz),
        "fp2_hz": _spectre_value(cfg.fp2_hz),
        "fz_hz": _spectre_value(cfg.fz_hz),
        "cmrr_db": str(cfg.cmrr_db),
        "rin_ohm": _spectre_value(cfg.rin_ohm),
        "rout_ohm": _spectre_value(cfg.rout_ohm),
        "a0_linear": _spectre_value(a0),
        "cmrr_linear": _spectre_value(cmrr_linear),
        "acm_gain": _spectre_value(acm_gain),
        "wp_rad_s": _spectre_value(wp),
        "rlp_ohm": _spectre_value(rlp_ohm),
        "clp_f": _spectre_value(clp_f),
    }
    text = _apply_ngspice_param_replacements(text, replacements)
    return (
        text.replace("PLACEHOLDER_A0", _spectre_value(a0))
        .replace("PLACEHOLDER_CMRR", _spectre_value(cmrr_linear))
        .replace("PLACEHOLDER_ACM", _spectre_value(acm_gain))
        .replace("PLACEHOLDER_WP", _spectre_value(wp))
        .replace("PLACEHOLDER_RLP", _spectre_value(rlp_ohm))
        .replace("PLACEHOLDER_CLP", _spectre_value(clp_f))
        .replace("PLACEHOLDER_FP2_BLOCK", fp2_block)
        .replace("PLACEHOLDER_FZ_BLOCK", fz_block)
        .replace("PLACEHOLDER_OUT_NODE", out_node)
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


def run_ngspice_ac_cm(
    cfg: OpampConfig,
    output_dir: Path,
    *,
    template_name: str = "ac_cm.cir",
) -> NgspiceAcResult:
    """Run ngspice common-mode AC and export ACM Bode data."""
    repo = package_root()
    template = repo / "testbench" / "ngspice" / template_name
    ng_dir = output_dir / "ngspice"
    logs_dir = output_dir / "logs"
    ng_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    netlist_path = ng_dir / template_name
    netlist_path.write_text(render_ngspice_cm_netlist(template, cfg), encoding="utf-8")
    log_path = logs_dir / f"ngspice_{template_name.replace('.cir', '')}.log"
    wrdata_path = ng_dir / "ac_cm.raw"

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
        msg = f"ngspice CM AC failed with code {completed.returncode}; see {log_path}"
        raise RuntimeError(msg)
    if not wrdata_path.is_file():
        msg = f"ngspice did not produce {wrdata_path}"
        raise RuntimeError(msg)

    bode = read_ngspice_ac_wrdata(wrdata_path)
    csv_path = output_dir / "ac_cm_bode.csv"
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


def try_cmrr_from_ngspice(
    cfg: OpampConfig,
    output_dir: Path,
    ac_result: AcSimulationResult,
) -> CmrrSimulationResult | None:
    """Derive CMRR from ngspice diff + CM AC benches when ngspice is available."""
    try:
        cm_artifacts = run_ngspice_ac_cm(cfg, output_dir)
        acm = read_ngspice_ac_wrdata(cm_artifacts.wrdata_path)
    except (NgspiceNotFoundError, RuntimeError):
        return None
    return cmrr_from_aol_and_acm(
        ac_result["frequency_hz"],
        ac_result["gain_db"],
        acm["frequency_hz"],
        acm["gain_db"],
        source="cmrr_bench",
    )


_CPS_REF_F = 1.0e-12


def render_ngspice_psrr_netlist(
    template_path: Path,
    cfg: OpampConfig,
) -> str:
    """Render an ngspice PSRR netlist with single-pole supply feedthrough."""
    text = template_path.read_text(encoding="utf-8")
    psrr_linear = max(cfg.psrr_linear, 1.0)
    wp_psrr = 2.0 * np.pi * max(cfg.psrr_pole_hz, 1.0e-30)
    cps_f = _CPS_REF_F
    rps_ohm = 1.0 / (wp_psrr * cps_f)
    psrr_inv = 1.0 / psrr_linear
    replacements = {
        "psrr_db": str(cfg.psrr_db),
        "psrr_pole_hz": _spectre_value(cfg.psrr_pole_hz),
        "rout_ohm": _spectre_value(cfg.rout_ohm),
        "psrr_linear": _spectre_value(psrr_linear),
        "psrr_inv": _spectre_value(psrr_inv),
        "wp_psrr_rad_s": _spectre_value(wp_psrr),
        "rps_ohm": _spectre_value(rps_ohm),
        "cps_f": _spectre_value(cps_f),
    }
    for key, val in replacements.items():
        text = re.sub(
            rf"^\.param\s+{key}=.*$",
            f".param {key}={val}",
            text,
            flags=re.MULTILINE,
        )
    return (
        text.replace("PLACEHOLDER_PSRR_LINEAR", _spectre_value(psrr_linear))
        .replace("PLACEHOLDER_PSRR_INV", _spectre_value(psrr_inv))
        .replace("PLACEHOLDER_WP_PSRR", _spectre_value(wp_psrr))
        .replace("PLACEHOLDER_RPS", _spectre_value(rps_ohm))
        .replace("PLACEHOLDER_CPS", _spectre_value(cps_f))
        .replace("PLACEHOLDER_DEC", str(cfg.sweep.points_per_decade))
        .replace("PLACEHOLDER_FSTART", _spectre_value(cfg.sweep.f_start_hz))
        .replace("PLACEHOLDER_FSTOP", _spectre_value(cfg.sweep.f_stop_hz))
    )


@dataclass(frozen=True)
class NgspicePsrrResult:
    """Artifacts from an ngspice PSRR run."""

    netlist_path: Path
    wrdata_path: Path
    log_path: Path
    csv_path: Path


def run_ngspice_psrr(
    cfg: OpampConfig,
    output_dir: Path,
    *,
    template_name: str = "psrr.cir",
) -> NgspicePsrrResult:
    """Run ngspice PSRR AC and export supply-transfer data to CSV."""
    repo = package_root()
    template = repo / "testbench" / "ngspice" / template_name
    ng_dir = output_dir / "ngspice"
    logs_dir = output_dir / "logs"
    ng_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    netlist_path = ng_dir / template_name
    netlist_path.write_text(render_ngspice_psrr_netlist(template, cfg), encoding="utf-8")
    log_path = logs_dir / "ngspice_psrr.log"
    wrdata_path = ng_dir / "psrr.raw"
    csv_path = output_dir / "psrr.csv"

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
        msg = f"ngspice PSRR failed with code {completed.returncode}; see {log_path}"
        raise RuntimeError(msg)
    if not wrdata_path.is_file():
        msg = f"ngspice did not produce {wrdata_path}"
        raise RuntimeError(msg)

    psrr_data = read_ngspice_psrr_wrdata(wrdata_path)
    transfer = psrr_data["magnitude"].astype(np.complex128)
    psrr_db = psrr_db_from_transfer(transfer)
    write_psrr_csv(csv_path, psrr_data["frequency_hz"], psrr_db)
    return NgspicePsrrResult(
        netlist_path=netlist_path,
        wrdata_path=wrdata_path,
        log_path=log_path,
        csv_path=csv_path,
    )


def ngspice_psrr_to_simulation_result(
    frequency_hz: NDArray[np.float64],
    magnitude: NDArray[np.float64],
) -> PsrrSimulationResult:
    """Build ``PsrrSimulationResult`` from ngspice ``vm(out)`` columns."""
    transfer = magnitude.astype(np.complex128)
    psrr_db = psrr_db_from_transfer(transfer)
    return PsrrSimulationResult(
        frequency_hz=frequency_hz,
        psrr_db=psrr_db,
        psrr_dc_db=float(psrr_db[0]),
    )


def simulate_psrr_ngspice(
    cfg: OpampConfig,
    output_dir: Path,
) -> PsrrSimulationResult:
    """Run PSRR bench in ngspice and return supply-transfer data from wrdata."""
    artifacts = run_ngspice_psrr(cfg, output_dir)
    psrr_data = read_ngspice_psrr_wrdata(artifacts.wrdata_path)
    return ngspice_psrr_to_simulation_result(
        psrr_data["frequency_hz"],
        psrr_data["magnitude"],
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


def render_ngspice_tia_netlist(
    template_path: Path,
    cfg: OpampConfig,
    tia: TiaConfig,
) -> str:
    """Render ngspice TIA AC netlist with one-pole op-amp and feedback network."""
    text = template_path.read_text(encoding="utf-8")
    a0 = max(cfg.a0_linear, 1.0)
    wp = dominant_pole_rad_s(cfg)
    clp_f = _CLP_REF_F
    rlp_ohm = 1.0 / (wp * clp_f)
    cshunt_f = cfg.cin_f + tia.cs_f
    replacements = {
        "rf_ohm": _spectre_value(tia.rf_ohm),
        "cf_f": _spectre_value(tia.cf_f),
        "cs_f": _spectre_value(tia.cs_f),
        "a0_db": str(cfg.a0_db),
        "gbw_hz": _spectre_value(cfg.gbw_hz),
        "rin_ohm": _spectre_value(cfg.rin_ohm),
        "cin_f": _spectre_value(cfg.cin_f),
        "cshunt_f": _spectre_value(cshunt_f),
        "rout_ohm": _spectre_value(cfg.rout_ohm),
        "cout_f": _spectre_value(cfg.cout_f),
        "a0_linear": _spectre_value(a0),
        "wp_rad_s": _spectre_value(wp),
        "rlp_ohm": _spectre_value(rlp_ohm),
        "clp_f": _spectre_value(clp_f),
        "f_start": _spectre_value(cfg.sweep.f_start_hz),
        "f_stop": _spectre_value(cfg.sweep.f_stop_hz),
        "dec": str(cfg.sweep.points_per_decade),
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


def run_ngspice_tia_ac(
    cfg: OpampConfig,
    tia: TiaConfig,
    output_dir: Path,
    *,
    template_name: str = "run_tia_ac.cir",
) -> Path:
    """Run ngspice TIA AC and export ``tia_zt.raw``."""
    if not tia.current_input:
        msg = "ngspice TIA bench supports current-input mode only; use --simulator python."
        raise ValueError(msg)
    repo = package_root()
    template = repo / "testbench" / "ngspice" / template_name
    ng_dir = output_dir / "ngspice"
    logs_dir = output_dir / "logs"
    ng_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    netlist_path = ng_dir / template_name
    netlist_path.write_text(render_ngspice_tia_netlist(template, cfg, tia), encoding="utf-8")
    log_path = logs_dir / "ngspice_tia_ac.log"
    wrdata_path = ng_dir / "tia_zt.raw"

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
        msg = f"ngspice TIA AC failed with code {completed.returncode}; see {log_path}"
        raise RuntimeError(msg)
    if not wrdata_path.is_file():
        msg = f"ngspice did not produce {wrdata_path}"
        raise RuntimeError(msg)
    return wrdata_path


def simulate_tia_ac_ngspice(
    cfg: OpampConfig,
    tia: TiaConfig,
    output_dir: Path,
    noise: OpampNoiseConfig | None = None,
) -> TiaSimulationResult:
    """Run TIA AC in ngspice and return transimpedance curves."""
    _ = noise
    wrdata_path = run_ngspice_tia_ac(cfg, tia, output_dir)
    bode = read_ngspice_ac_wrdata(wrdata_path)
    return bode_to_tia_result(
        bode["frequency_hz"],
        bode["gain_db"],
        bode["phase_deg"],
    )


def render_ngspice_gm_netlist(
    template_path: Path,
    gm_cfg: GmConfig,
    *,
    sweep: BenchSweepConfig | None = None,
) -> str:
    """Render ngspice Gm loaded AC netlist."""
    text = template_path.read_text(encoding="utf-8")
    bench = sweep or BenchSweepConfig()
    replacements = {
        "gm_s": _spectre_value(gm_cfg.gm_s),
        "rout_ohm": _spectre_value(gm_cfg.rout_ohm),
        "cout_f": _spectre_value(gm_cfg.cout_f),
        "f_start": _spectre_value(bench.f_start_hz),
        "f_stop": _spectre_value(bench.f_stop_hz),
        "dec": str(bench.points_per_decade),
    }
    for key, val in replacements.items():
        text = re.sub(
            rf"^\.param\s+{key}=.*$",
            f".param {key}={val}",
            text,
            flags=re.MULTILINE,
        )
    return text


def run_ngspice_gm_ac(
    gm_cfg: GmConfig,
    output_dir: Path,
    *,
    sweep: BenchSweepConfig | None = None,
    template_name: str = "run_gm_ac.cir",
) -> Path:
    """Run ngspice Gm AC and export ``gm_ac_bode.raw``."""
    repo = package_root()
    template = repo / "testbench" / "ngspice" / template_name
    ng_dir = output_dir / "ngspice"
    logs_dir = output_dir / "logs"
    ng_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    netlist_path = ng_dir / template_name
    netlist_path.write_text(
        render_ngspice_gm_netlist(template, gm_cfg, sweep=sweep),
        encoding="utf-8",
    )
    log_path = logs_dir / "ngspice_gm_ac.log"
    wrdata_path = ng_dir / "gm_ac_bode.raw"

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
        msg = f"ngspice Gm AC failed with code {completed.returncode}; see {log_path}"
        raise RuntimeError(msg)
    if not wrdata_path.is_file():
        msg = f"ngspice did not produce {wrdata_path}"
        raise RuntimeError(msg)
    return wrdata_path


def simulate_gm_ac_ngspice(
    gm_cfg: GmConfig,
    output_dir: Path,
    noise: OpampNoiseConfig | None = None,
    *,
    sweep: BenchSweepConfig | None = None,
) -> GmAcSimulationResult:
    """Run Gm loaded AC in ngspice and return Bode data."""
    _ = noise
    wrdata_path = run_ngspice_gm_ac(gm_cfg, output_dir, sweep=sweep)
    bode = read_ngspice_ac_wrdata(wrdata_path)
    return bode_to_gm_result(
        gm_cfg,
        bode["frequency_hz"],
        bode["gain_db"],
        bode["phase_deg"],
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
