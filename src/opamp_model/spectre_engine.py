"""Cadence Spectre AC testbench runner."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from opamp_model.ac import cmrr_from_aol_and_acm, extract_gbw_phase_margin
from opamp_model.core import dominant_pole_rad_s
from opamp_model.cm_ps import CmrrSimulationResult, PsrrSimulationResult, psrr_db_from_transfer
from opamp_model.config import BenchSweepConfig, GmConfig, OpampConfig, OpampNoiseConfig, TiaConfig
from opamp_model.gm import GmAcSimulationResult, bode_to_gm_result
from opamp_model.io import log_frequency_sweep, package_root, write_bode_csv, write_psrr_csv
from opamp_model.model import AcSimulationResult, NoiseSimulationResult
from opamp_model.ngspice_engine import merge_engine_noise_spectrum
from opamp_model.noise_analysis import extract_noise_metrics
from opamp_model.spectre_psf import (
    SpectrePsfError,
    locate_ac_psf,
    locate_noise_psf,
    read_spectre_ac_from_netlists,
    read_spectre_noise_from_netlists,
    read_spectre_psrr_from_netlists,
    read_spectre_tran_from_netlists,
)
from opamp_model.tia import TiaSimulationResult, bode_to_tia_result
from opamp_model.tran import (
    SlewMetrics,
    ThdMetrics,
    TranSineResult,
    TranStepResult,
    compute_thd,
    extract_slew_rate,
    is_thd_ideal,
    overlay_transient_noise_on_sine,
    overlay_transient_noise_on_step,
)


class SpectreNotFoundError(RuntimeError):
    """Raised when the ``spectre`` executable is not on PATH."""


class SpectreLicenseError(RuntimeError):
    """Raised when Spectre is installed but the license file/server is unavailable."""


_DEFAULT_CDS_LIC_FILE = Path("/eda/cadence/license.dat")
_LICENSE_FAILURE_MARKERS = (
    "LMC-01902",
    "Failed to initialize license API",
    "Can't find license file",
)


_SCS_INCLUDE = re.compile(
    r'include\s+"\./testbench/spectre/opamp_include\.scs"\s*',
    re.IGNORECASE,
)
_SCS_AHDL_INCLUDE = re.compile(
    r'ahdl_include\s+"\./veriloga/([^"]+)"\s*',
    re.IGNORECASE,
)


def find_spectre_executable() -> str:
    """Return the Spectre binary path or raise ``SpectreNotFoundError``."""
    found = shutil.which("spectre")
    if found:
        return found
    for candidate in (
        "/eda/cadence/SPECTRE241/tools/bin/spectre",
        "/eda/cadence/SPECTRE231/tools/bin/spectre",
    ):
        if Path(candidate).is_file():
            return candidate
    raise SpectreNotFoundError(
        "Cadence Spectre not found on PATH. Install Spectre or use --simulator python."
    )


def spectre_is_runnable() -> bool:
    """Return True when Spectre exists and a license file/env is configured."""
    try:
        find_spectre_executable()
    except SpectreNotFoundError:
        return False
    if os.environ.get("CDS_LIC_FILE") or os.environ.get("LM_LICENSE_FILE"):
        return True
    return _DEFAULT_CDS_LIC_FILE.is_file()


def _spectre_subprocess_env() -> dict[str, str]:
    """Build subprocess env with Spectre bin on PATH and a default license file."""
    env = os.environ.copy()
    executable = find_spectre_executable()
    spectre_bin = str(Path(executable).parent)
    env["PATH"] = f"{spectre_bin}:{env.get('PATH', '')}"
    if not env.get("CDS_LIC_FILE") and _DEFAULT_CDS_LIC_FILE.is_file():
        env["CDS_LIC_FILE"] = str(_DEFAULT_CDS_LIC_FILE)
    return env


def _raise_if_spectre_license_failure(
    completed: subprocess.CompletedProcess[str],
    *,
    log_path: Path,
) -> None:
    """Raise ``SpectreLicenseError`` when Spectre output indicates license failure."""
    output = completed.stdout + completed.stderr
    if any(marker in output for marker in _LICENSE_FAILURE_MARKERS):
        raise SpectreLicenseError(f"Cadence Spectre license unavailable; see {log_path}")


def _raise_if_spectre_failed(
    completed: subprocess.CompletedProcess[str],
    *,
    log_path: Path,
    message: str,
) -> None:
    """Raise on Spectre failure, distinguishing license errors from other faults."""
    if completed.returncode == 0:
        return
    _raise_if_spectre_license_failure(completed, log_path=log_path)
    raise RuntimeError(f"{message} with code {completed.returncode}; see {log_path}")


def _format_param(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.12g}"


def _absolutize_spectre_includes(text: str, repo_root: Path) -> str:
    """Replace repo-relative includes with absolute Verilog-A paths.

    Rendered netlists run from ``<output>/logs/netlists/``; relative paths break.
    """
    opamp_va = (repo_root / "veriloga/configurable_opamp.va").resolve()
    text = _SCS_INCLUDE.sub(f'ahdl_include "{opamp_va}"\n', text)

    def _va_repl(match: re.Match[str]) -> str:
        va_path = (repo_root / "veriloga" / match.group(1)).resolve()
        return f'ahdl_include "{va_path}"\n'

    return _SCS_AHDL_INCLUDE.sub(_va_repl, text)


def render_spectre_ac_netlist(
    template_path: Path,
    cfg: OpampConfig,
    *,
    repo_root: Path | None = None,
) -> str:
    """Render a Spectre AC netlist with CLI macromodel settings."""
    root = repo_root or package_root()
    text = template_path.read_text(encoding="utf-8")
    overrides = {
        "a0_db": _format_param(cfg.a0_db),
        "gbw_hz": _format_param(cfg.gbw_hz),
        "fp2_hz": _format_param(cfg.fp2_hz),
        "fz_hz": _format_param(cfg.fz_hz),
        "cmrr_db": _format_param(cfg.cmrr_db),
        "rin_ohm": _format_param(cfg.rin_ohm),
        "cin_f": _format_param(cfg.cin_f),
        "rout_ohm": _format_param(cfg.rout_ohm),
        "cout_f": _format_param(cfg.cout_f),
        "f_start": _format_param(cfg.sweep.f_start_hz),
        "f_stop": _format_param(cfg.sweep.f_stop_hz),
        "dec": _format_param(cfg.sweep.points_per_decade),
    }
    for name, value in overrides.items():
        text = re.sub(
            rf"^parameters\s+{name}=.*$",
            f"parameters {name}={value}",
            text,
            flags=re.MULTILINE,
        )
    return _absolutize_spectre_includes(text, root)


def render_spectre_psrr_netlist(
    template_path: Path,
    cfg: OpampConfig,
    *,
    repo_root: Path | None = None,
) -> str:
    """Render a Spectre PSRR netlist with supply-AC stimulus parameters."""
    root = repo_root or package_root()
    text = template_path.read_text(encoding="utf-8")
    overrides = {
        "a0_db": _format_param(cfg.a0_db),
        "gbw_hz": _format_param(cfg.gbw_hz),
        "cmrr_db": _format_param(cfg.cmrr_db),
        "psrr_db": _format_param(cfg.psrr_db),
        "psrr_pole_hz": _format_param(cfg.psrr_pole_hz),
        "rin_ohm": _format_param(cfg.rin_ohm),
        "cin_f": _format_param(cfg.cin_f),
        "rout_ohm": _format_param(cfg.rout_ohm),
        "cout_f": _format_param(cfg.cout_f),
        "f_start": _format_param(cfg.sweep.f_start_hz),
        "f_stop": _format_param(cfg.sweep.f_stop_hz),
        "dec": _format_param(cfg.sweep.points_per_decade),
    }
    for name, value in overrides.items():
        text = re.sub(
            rf"^parameters\s+{name}=.*$",
            f"parameters {name}={value}",
            text,
            flags=re.MULTILINE,
        )
    return _absolutize_spectre_includes(text, root)


def _noise_en_flicker_1hz(noise: OpampNoiseConfig) -> float:
    return float(
        getattr(
            noise,
            "en_flicker_1hz_v_per_sqrt_hz",
            getattr(noise, "en_flicker_at_1hz_v_per_sqrt_hz", 0.0),
        )
    )


def _noise_en_flicker_ef(noise: OpampNoiseConfig) -> float:
    return float(getattr(noise, "en_flicker_ef", 1.0))


def _noise_kf(noise: OpampNoiseConfig) -> float:
    return float(getattr(noise, "kf", 0.0))


def _noise_af(noise: OpampNoiseConfig) -> float:
    return float(getattr(noise, "af", 1.0))


def _noise_bias_current_a(noise: OpampNoiseConfig) -> float:
    return float(getattr(noise, "bias_current_a", 0.0))


def _va_intrinsic_white(noise: OpampNoiseConfig) -> bool:
    return noise.enabled and noise.en_white_v_per_sqrt_hz > 0.0


def _va_intrinsic_flicker(noise: OpampNoiseConfig) -> bool:
    if not noise.enabled:
        return False
    if _noise_en_flicker_1hz(noise) > 0.0:
        return True
    return _noise_kf(noise) > 0.0 and abs(_noise_bias_current_a(noise)) > 0.0


def render_spectre_noise_netlist(
    template_path: Path,
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
    *,
    repo_root: Path | None = None,
) -> str:
    """Render a Spectre open-loop noise netlist (``noise`` + ``ac`` analyses)."""
    root = repo_root or package_root()
    text = template_path.read_text(encoding="utf-8")
    en_white = noise.en_white_v_per_sqrt_hz if noise.enabled else 0.0
    en_flicker_1hz = _noise_en_flicker_1hz(noise) if noise.enabled else 0.0
    kT = 4.0 * 1.380649e-23 * 300.0
    rn_fallback = (en_white * en_white) / kT if en_white > 0.0 else 1.0e-6
    # VA white_noise replaces RN when intrinsic noise is enabled.
    rn_white = 1.0e12 if _va_intrinsic_white(noise) else rn_fallback
    enable_noise = 1 if noise.enabled else 0
    overrides = {
        "a0_db": _format_param(cfg.a0_db),
        "gbw_hz": _format_param(cfg.gbw_hz),
        "rin_ohm": _format_param(cfg.rin_ohm),
        "cin_f": _format_param(cfg.cin_f),
        "rout_ohm": _format_param(cfg.rout_ohm),
        "cout_f": _format_param(cfg.cout_f),
        "en_white_v_per_sqrt_hz": _format_param(en_white),
        "rn_white_ohm": _format_param(rn_white),
        "en_flicker_1hz_v_per_sqrt_hz": _format_param(en_flicker_1hz),
        "en_flicker_ef": _format_param(_noise_en_flicker_ef(noise)),
        "kf": _format_param(_noise_kf(noise)),
        "af": _format_param(_noise_af(noise)),
        "bias_current_a": _format_param(_noise_bias_current_a(noise)),
        "enable_noise": _format_param(enable_noise),
        "en_flicker_corner_hz": _format_param(
            getattr(noise, "en_flicker_corner_hz", float("nan"))
        ),
        "en_flicker_at_1hz_v_per_sqrt_hz": _format_param(en_flicker_1hz),
        "f_start": _format_param(cfg.sweep.f_start_hz),
        "f_stop": _format_param(cfg.sweep.f_stop_hz),
        "dec": _format_param(cfg.sweep.points_per_decade),
    }
    for name, value in overrides.items():
        text = re.sub(
            rf"^parameters\s+{name}=.*$",
            f"parameters {name}={value}",
            text,
            flags=re.MULTILINE,
        )
    text = text.replace("PLACEHOLDER_RN", _format_param(rn_white))
    return _absolutize_spectre_includes(text, root)


def run_spectre_netlist(
    netlist: Path,
    *,
    cwd: Path | None = None,
    timeout_s: float | None = None,
    psf_format: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run Spectre on a rendered ``.scs`` netlist."""
    executable = find_spectre_executable()
    work_dir = (cwd or netlist.parent).resolve()
    netlist_arg = netlist.name if netlist.resolve().parent == work_dir else str(netlist.resolve())
    args = [executable, netlist_arg, "+log", "status"]
    if psf_format is not None:
        args.extend(["-format", psf_format])
    return subprocess.run(
        args,
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
        env=_spectre_subprocess_env(),
    )


@dataclass(frozen=True)
class SpectreAcResult:
    """Artifacts from a Spectre AC run."""

    netlist_path: Path
    log_path: Path
    raw_dir: Path
    csv_path: Path


def run_spectre_ac(
    cfg: OpampConfig,
    output_dir: Path,
    *,
    template_name: str = "ac_open_loop.scs",
) -> SpectreAcResult:
    """Render and execute a Spectre AC netlist; export Bode CSV from PSF."""
    repo = package_root()
    template = repo / "testbench" / "spectre" / template_name
    logs_dir = output_dir / "logs" / "netlists"
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = logs_dir / template_name
    netlist_path.write_text(
        render_spectre_ac_netlist(template, cfg, repo_root=repo),
        encoding="utf-8",
    )
    stem = template_name.replace(".scs", "")
    log_path = output_dir / "logs" / f"spectre_{stem}.log"
    csv_path = output_dir / ("stb_bode.csv" if stem == "stb_loop" else "ac_bode.csv")
    completed = run_spectre_netlist(netlist_path, cwd=logs_dir)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    _raise_if_spectre_failed(
        completed,
        log_path=log_path,
        message="Spectre failed",
    )

    try:
        bode = read_spectre_ac_from_netlists(logs_dir, stem=stem, signal="out")
    except SpectrePsfError as exc:
        msg = f"Spectre AC PSF read failed: {exc}"
        raise RuntimeError(msg) from exc

    write_bode_csv(
        csv_path,
        bode["frequency_hz"],
        bode["gain_db"],
        bode["phase_deg"],
    )
    raw_dir = locate_ac_psf(logs_dir, stem=stem).raw_dir
    return SpectreAcResult(
        netlist_path=netlist_path,
        log_path=log_path,
        raw_dir=raw_dir,
        csv_path=csv_path,
    )


def spectre_ac_to_simulation_result(
    frequency_hz: NDArray[np.float64],
    gain_db: NDArray[np.float64],
    phase_deg: NDArray[np.float64],
) -> AcSimulationResult:
    """Build ``AcSimulationResult`` from Spectre AC PSF columns."""
    metrics = extract_gbw_phase_margin(frequency_hz, gain_db, phase_deg)
    return AcSimulationResult(
        frequency_hz=frequency_hz,
        gain_db=gain_db,
        phase_deg=phase_deg,
        metrics=metrics,
    )


def simulate_ac_spectre(
    cfg: OpampConfig,
    output_dir: Path,
    noise: OpampNoiseConfig | None = None,
) -> AcSimulationResult:
    """Run open-loop AC in Spectre and return Bode data parsed from PSF."""
    _ = noise
    artifacts = run_spectre_ac(cfg, output_dir)
    bode = read_spectre_ac_from_netlists(
        artifacts.netlist_path.parent,
        stem=artifacts.netlist_path.stem,
        signal="out",
    )
    return spectre_ac_to_simulation_result(
        bode["frequency_hz"],
        bode["gain_db"],
        bode["phase_deg"],
    )


def run_spectre_ac_cm(
    cfg: OpampConfig,
    output_dir: Path,
    *,
    template_name: str = "ac_cm.scs",
) -> SpectreAcResult:
    """Render and execute a Spectre common-mode AC netlist; export ACM from PSF."""
    repo = package_root()
    template = repo / "testbench" / "spectre" / template_name
    logs_dir = output_dir / "logs" / "netlists"
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = logs_dir / template_name
    netlist_path.write_text(
        render_spectre_ac_netlist(template, cfg, repo_root=repo),
        encoding="utf-8",
    )
    stem = template_name.replace(".scs", "")
    log_path = output_dir / "logs" / f"spectre_{stem}.log"
    csv_path = output_dir / "ac_cm_bode.csv"
    completed = run_spectre_netlist(netlist_path, cwd=logs_dir)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    _raise_if_spectre_failed(
        completed,
        log_path=log_path,
        message="Spectre CM AC failed",
    )

    try:
        bode = read_spectre_ac_from_netlists(logs_dir, stem=stem, signal="out")
    except SpectrePsfError as exc:
        msg = f"Spectre CM AC PSF read failed: {exc}"
        raise RuntimeError(msg) from exc

    write_bode_csv(
        csv_path,
        bode["frequency_hz"],
        bode["gain_db"],
        bode["phase_deg"],
    )
    raw_dir = locate_ac_psf(logs_dir, stem=stem).raw_dir
    return SpectreAcResult(
        netlist_path=netlist_path,
        log_path=log_path,
        raw_dir=raw_dir,
        csv_path=csv_path,
    )


def try_cmrr_from_spectre(
    cfg: OpampConfig,
    output_dir: Path,
    ac_result: AcSimulationResult,
) -> CmrrSimulationResult | None:
    """Derive CMRR from Spectre diff + CM AC PSF when Spectre/PSF are available."""
    try:
        cm_artifacts = run_spectre_ac_cm(cfg, output_dir)
        acm = read_spectre_ac_from_netlists(
            cm_artifacts.netlist_path.parent,
            stem=cm_artifacts.netlist_path.stem,
            signal="out",
        )
    except (SpectreNotFoundError, SpectreLicenseError, RuntimeError, SpectrePsfError):
        return None
    return cmrr_from_aol_and_acm(
        ac_result["frequency_hz"],
        ac_result["gain_db"],
        acm["frequency_hz"],
        acm["gain_db"],
        source="cmrr_bench",
    )


@dataclass(frozen=True)
class SpectrePsrrResult:
    """Artifacts from a Spectre PSRR run."""

    netlist_path: Path
    log_path: Path
    raw_dir: Path
    csv_path: Path


def run_spectre_psrr(
    cfg: OpampConfig,
    output_dir: Path,
    *,
    template_name: str = "psrr.scs",
) -> SpectrePsrrResult:
    """Render and execute a Spectre PSRR netlist; export PSRR CSV from PSF."""
    repo = package_root()
    template = repo / "testbench" / "spectre" / template_name
    logs_dir = output_dir / "logs" / "netlists"
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = logs_dir / template_name
    netlist_path.write_text(
        render_spectre_psrr_netlist(template, cfg, repo_root=repo),
        encoding="utf-8",
    )
    stem = template_name.replace(".scs", "")
    log_path = output_dir / "logs" / f"spectre_{stem}.log"
    csv_path = output_dir / "psrr.csv"
    completed = run_spectre_netlist(netlist_path, cwd=logs_dir)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    _raise_if_spectre_failed(
        completed,
        log_path=log_path,
        message="Spectre PSRR failed",
    )

    try:
        psrr_data = read_spectre_psrr_from_netlists(logs_dir, stem=stem)
    except SpectrePsfError as exc:
        msg = f"Spectre PSRR PSF read failed: {exc}"
        raise RuntimeError(msg) from exc

    psrr_db = psrr_db_from_transfer(psrr_data["transfer"])
    write_psrr_csv(csv_path, psrr_data["frequency_hz"], psrr_db)
    raw_dir = locate_ac_psf(logs_dir, stem=stem).raw_dir
    return SpectrePsrrResult(
        netlist_path=netlist_path,
        log_path=log_path,
        raw_dir=raw_dir,
        csv_path=csv_path,
    )


def spectre_psrr_to_simulation_result(
    frequency_hz: NDArray[np.float64],
    transfer: NDArray[np.complex128],
) -> PsrrSimulationResult:
    """Build ``PsrrSimulationResult`` from Spectre supply-transfer PSF columns."""
    psrr_db = psrr_db_from_transfer(transfer)
    return PsrrSimulationResult(
        frequency_hz=frequency_hz,
        psrr_db=psrr_db,
        psrr_dc_db=float(psrr_db[0]),
    )


def simulate_psrr_spectre(
    cfg: OpampConfig,
    output_dir: Path,
) -> PsrrSimulationResult:
    """Run PSRR bench in Spectre and return supply-transfer data parsed from PSF."""
    artifacts = run_spectre_psrr(cfg, output_dir)
    psrr_data = read_spectre_psrr_from_netlists(
        artifacts.netlist_path.parent,
        stem=artifacts.netlist_path.stem,
    )
    return spectre_psrr_to_simulation_result(
        psrr_data["frequency_hz"],
        psrr_data["transfer"],
    )


@dataclass(frozen=True)
class SpectreNoiseResult:
    """Artifacts from a Spectre noise analysis run."""

    netlist_path: Path
    log_path: Path
    raw_dir: Path


def run_spectre_noise(
    cfg: OpampConfig,
    noise: OpampNoiseConfig,
    output_dir: Path,
    *,
    template_name: str = "noise_open_loop.scs",
) -> SpectreNoiseResult:
    """Render and execute a Spectre open-loop noise netlist."""
    repo = package_root()
    template = repo / "testbench" / "spectre" / template_name
    logs_dir = output_dir / "logs" / "netlists"
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = logs_dir / template_name
    netlist_path.write_text(
        render_spectre_noise_netlist(template, cfg, noise, repo_root=repo),
        encoding="utf-8",
    )
    stem = template_name.replace(".scs", "")
    log_path = output_dir / "logs" / f"spectre_{stem}.log"
    completed = run_spectre_netlist(netlist_path, cwd=logs_dir)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    _raise_if_spectre_failed(
        completed,
        log_path=log_path,
        message="Spectre noise failed",
    )
    raw_dir = locate_noise_psf(logs_dir, stem=stem).raw_dir
    return SpectreNoiseResult(
        netlist_path=netlist_path,
        log_path=log_path,
        raw_dir=raw_dir,
    )


def simulate_noise_spectre(
    cfg: OpampConfig,
    output_dir: Path,
    noise: OpampNoiseConfig,
) -> NoiseSimulationResult:
    """Run Spectre noise bench; merge native PSF with AC-gain spectrum."""
    artifacts = run_spectre_noise(cfg, noise, output_dir)
    logs_dir = output_dir / "logs" / "netlists"
    try:
        ac = read_spectre_ac_from_netlists(logs_dir, stem="noise_open_loop", signal="out")
        native = read_spectre_noise_from_netlists(logs_dir, stem="noise_open_loop")
    except SpectrePsfError as exc:
        msg = f"Spectre noise PSF read failed: {exc}"
        raise RuntimeError(msg) from exc
    frequency_hz = log_frequency_sweep(
        cfg.sweep.f_start_hz,
        cfg.sweep.f_stop_hz,
        cfg.sweep.points_per_decade,
    )
    spectrum, _ = merge_engine_noise_spectrum(
        cfg,
        noise,
        frequency_hz=frequency_hz,
        gain_db=ac["gain_db"],
        ac_frequency_hz=ac["frequency_hz"],
        native_onoise=native["noise_v_per_sqrt_hz"],
        native_frequency_hz=native["frequency_hz"],
    )
    _ = artifacts
    metrics = extract_noise_metrics(cfg, noise, frequency_hz, spectrum)
    return NoiseSimulationResult(
        frequency_hz=frequency_hz,
        noise_v_per_sqrt_hz=spectrum,
        metrics=metrics,
    )


_TIA_CLP_REF_F = 1.0e-12


def render_spectre_tia_netlist(
    template_path: Path,
    cfg: OpampConfig,
    tia: TiaConfig,
    *,
    repo_root: Path | None = None,
) -> str:
    """Render a Spectre TIA AC netlist with CLI macromodel settings."""
    root = repo_root or package_root()
    text = template_path.read_text(encoding="utf-8")
    a0 = max(cfg.a0_linear, 1.0)
    wp = dominant_pole_rad_s(cfg)
    clp_f = _TIA_CLP_REF_F
    rlp_ohm = 1.0 / (wp * clp_f)
    cshunt_f = cfg.cin_f + tia.cs_f
    # Norton injection: Z_in ≈ Rf/A0 at low frequency → scale V_inj for 1 A AC.
    iinj_mag = 1.0 + tia.rf_ohm / a0
    overrides = {
        "rf_ohm": _format_param(tia.rf_ohm),
        "cf_f": _format_param(tia.cf_f),
        "cs_f": _format_param(tia.cs_f),
        "a0_db": _format_param(cfg.a0_db),
        "gbw_hz": _format_param(cfg.gbw_hz),
        "a0_linear": _format_param(a0),
        "rlp_ohm": _format_param(rlp_ohm),
        "clp_f": _format_param(clp_f),
        "rin_ohm": _format_param(cfg.rin_ohm),
        "cin_f": _format_param(cfg.cin_f),
        "cshunt_f": _format_param(cshunt_f),
        "rout_ohm": _format_param(cfg.rout_ohm),
        "cout_f": _format_param(cfg.cout_f),
        "iinj_mag": _format_param(iinj_mag),
        "f_start": _format_param(cfg.sweep.f_start_hz),
        "f_stop": _format_param(cfg.sweep.f_stop_hz),
        "dec": _format_param(cfg.sweep.points_per_decade),
    }
    for name, value in overrides.items():
        text = re.sub(
            rf"^parameters\s+{name}=.*$",
            f"parameters {name}={value}",
            text,
            flags=re.MULTILINE,
        )
    return (
        text.replace("PLACEHOLDER_A0", _format_param(a0))
        .replace("PLACEHOLDER_RLP", _format_param(rlp_ohm))
        .replace("PLACEHOLDER_CLP", _format_param(clp_f))
        .replace("PLACEHOLDER_IINJ", _format_param(iinj_mag))
    )


def run_spectre_tia_ac(
    cfg: OpampConfig,
    tia: TiaConfig,
    output_dir: Path,
    *,
    template_name: str = "run_tia_ac.scs",
) -> SpectreAcResult:
    """Render and execute a Spectre TIA AC netlist; export Zt CSV from PSF."""
    if not tia.current_input:
        msg = "Spectre TIA bench supports current-input mode only; use --simulator python."
        raise ValueError(msg)
    repo = package_root()
    template = repo / "testbench" / "spectre" / template_name
    logs_dir = output_dir / "logs" / "netlists"
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = logs_dir / template_name
    netlist_path.write_text(
        render_spectre_tia_netlist(template, cfg, tia, repo_root=repo),
        encoding="utf-8",
    )
    stem = template_name.replace(".scs", "")
    log_path = output_dir / "logs" / "spectre_tia_ac.log"
    csv_path = output_dir / "tia_zt.csv"
    completed = run_spectre_netlist(netlist_path, cwd=logs_dir)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    _raise_if_spectre_failed(
        completed,
        log_path=log_path,
        message="Spectre TIA AC failed",
    )

    try:
        bode = read_spectre_ac_from_netlists(logs_dir, stem=stem, signal="out")
    except SpectrePsfError as exc:
        msg = f"Spectre TIA AC PSF read failed: {exc}"
        raise RuntimeError(msg) from exc

    result = bode_to_tia_result(
        bode["frequency_hz"],
        bode["gain_db"],
        bode["phase_deg"],
    )
    from opamp_model.io import write_tia_csv

    write_tia_csv(
        csv_path,
        result["frequency_hz"],
        result["zt_ohm"],
        result["zt_db"],
        result["zt_phase_deg"],
    )
    raw_dir = locate_ac_psf(logs_dir, stem=stem).raw_dir
    return SpectreAcResult(
        netlist_path=netlist_path,
        log_path=log_path,
        raw_dir=raw_dir,
        csv_path=csv_path,
    )


def simulate_tia_ac_spectre(
    cfg: OpampConfig,
    tia: TiaConfig,
    output_dir: Path,
    noise: OpampNoiseConfig | None = None,
) -> TiaSimulationResult:
    """Run TIA AC in Spectre and return transimpedance curves from PSF."""
    _ = noise
    artifacts = run_spectre_tia_ac(cfg, tia, output_dir)
    bode = read_spectre_ac_from_netlists(
        artifacts.netlist_path.parent,
        stem=artifacts.netlist_path.stem,
        signal="out",
    )
    return bode_to_tia_result(
        bode["frequency_hz"],
        bode["gain_db"],
        bode["phase_deg"],
    )


def render_spectre_gm_netlist(
    template_path: Path,
    gm_cfg: GmConfig,
    *,
    sweep: BenchSweepConfig | None = None,
    repo_root: Path | None = None,
) -> str:
    """Render a Spectre Gm loaded AC netlist."""
    root = repo_root or package_root()
    bench = sweep or BenchSweepConfig()
    text = template_path.read_text(encoding="utf-8")
    overrides = {
        "gm_s": _format_param(gm_cfg.gm_s),
        "rout_ohm": _format_param(gm_cfg.rout_ohm),
        "cout_f": _format_param(gm_cfg.cout_f),
        "f_start": _format_param(bench.f_start_hz),
        "f_stop": _format_param(bench.f_stop_hz),
        "dec": _format_param(bench.points_per_decade),
    }
    for name, value in overrides.items():
        text = re.sub(
            rf"^parameters\s+{name}=.*$",
            f"parameters {name}={value}",
            text,
            flags=re.MULTILINE,
        )
    return _absolutize_spectre_includes(text, root)


def run_spectre_gm_ac(
    gm_cfg: GmConfig,
    output_dir: Path,
    *,
    sweep: BenchSweepConfig | None = None,
    template_name: str = "run_gm_ac.scs",
) -> SpectreAcResult:
    """Render and execute a Spectre Gm AC netlist; export Bode CSV from PSF."""
    repo = package_root()
    template = repo / "testbench" / "spectre" / template_name
    logs_dir = output_dir / "logs" / "netlists"
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = logs_dir / template_name
    netlist_path.write_text(
        render_spectre_gm_netlist(template, gm_cfg, sweep=sweep, repo_root=repo),
        encoding="utf-8",
    )
    stem = template_name.replace(".scs", "")
    log_path = output_dir / "logs" / "spectre_gm_ac.log"
    csv_path = output_dir / "gm_ac_bode.csv"
    completed = run_spectre_netlist(netlist_path, cwd=logs_dir)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    _raise_if_spectre_failed(
        completed,
        log_path=log_path,
        message="Spectre Gm AC failed",
    )

    try:
        bode = read_spectre_ac_from_netlists(logs_dir, stem=stem, signal="iout")
    except SpectrePsfError as exc:
        msg = f"Spectre Gm AC PSF read failed: {exc}"
        raise RuntimeError(msg) from exc

    write_bode_csv(
        csv_path,
        bode["frequency_hz"],
        bode["gain_db"],
        bode["phase_deg"],
    )
    raw_dir = locate_ac_psf(logs_dir, stem=stem).raw_dir
    return SpectreAcResult(
        netlist_path=netlist_path,
        log_path=log_path,
        raw_dir=raw_dir,
        csv_path=csv_path,
    )


def simulate_gm_ac_spectre(
    gm_cfg: GmConfig,
    output_dir: Path,
    noise: OpampNoiseConfig | None = None,
    *,
    sweep: BenchSweepConfig | None = None,
) -> GmAcSimulationResult:
    """Run Gm loaded AC in Spectre and return Bode data parsed from PSF."""
    _ = noise
    artifacts = run_spectre_gm_ac(gm_cfg, output_dir, sweep=sweep)
    bode = read_spectre_ac_from_netlists(
        artifacts.netlist_path.parent,
        stem=artifacts.netlist_path.stem,
        signal="iout",
    )
    return bode_to_gm_result(
        gm_cfg,
        bode["frequency_hz"],
        bode["gain_db"],
        bode["phase_deg"],
    )


def render_spectre_slew_netlist(
    template_path: Path,
    cfg: OpampConfig,
    *,
    step_v: float,
    duration_s: float,
    dt_s: float,
    repo_root: Path | None = None,
) -> str:
    """Render Spectre unity-gain follower slew transient netlist."""
    import math

    _ = repo_root
    gm = 2.0 * math.pi * cfg.gbw_hz
    slew_pos = abs(cfg.slew_pos_vps)
    slew_neg = cfg.slew_neg_vps if cfg.slew_neg_vps < 0.0 else -abs(cfg.slew_neg_vps)
    bs_expr = f"min(max({_format_param(gm)}*(v(vin)-v(n)), {_format_param(slew_neg)}), {_format_param(slew_pos)})"
    pulse_period = max(duration_s * 4.0, duration_s + 1.0e-9)
    return (
        template_path.read_text(encoding="utf-8")
        .replace("PLACEHOLDER_VCM", _format_param(cfg.vcm_v))
        .replace("PLACEHOLDER_STEP", _format_param(step_v))
        .replace("PLACEHOLDER_BS_EXPR", bs_expr)
        .replace("PLACEHOLDER_TSTOP", _format_param(duration_s))
        .replace("PLACEHOLDER_TSTEP", _format_param(dt_s))
        .replace("PLACEHOLDER_PULSE_PERIOD", _format_param(pulse_period))
    )


def run_spectre_slew_step(
    cfg: OpampConfig,
    output_dir: Path,
    *,
    step_v: float,
    duration_s: float,
    dt_s: float,
    stem: str,
    template_name: str = "slew_follower.scs",
) -> Path:
    """Run one Spectre slew step transient and return the netlist directory."""
    repo = package_root()
    template = repo / "testbench" / "spectre" / template_name
    logs_dir = output_dir / "logs" / "netlists"
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = logs_dir / f"{stem}.scs"
    netlist_path.write_text(
        render_spectre_slew_netlist(
            template,
            cfg,
            step_v=step_v,
            duration_s=duration_s,
            dt_s=dt_s,
            repo_root=repo,
        ),
        encoding="utf-8",
    )
    log_path = output_dir / "logs" / f"spectre_{stem}.log"
    completed = run_spectre_netlist(netlist_path, cwd=logs_dir, psf_format="psfascii")
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    _raise_if_spectre_failed(
        completed,
        log_path=log_path,
        message=f"Spectre slew TRAN failed ({stem})",
    )
    return logs_dir


def _spectre_tran_step_from_netlists(
    netlists_dir: Path,
    *,
    stem: str,
) -> TranStepResult:
    """Build a ``TranStepResult`` from Spectre slew PSF."""
    data = read_spectre_tran_from_netlists(netlists_dir, stem=stem, signal="out")
    zeros = np.zeros_like(data["signal_v"], dtype=np.float64)
    return TranStepResult(
        time_s=data["time_s"],
        vout_v=data["signal_v"],
        vout_clean_v=data["signal_v"],
        noise_v=zeros,
    )


def simulate_slew_spectre(
    cfg: OpampConfig,
    output_dir: Path,
    *,
    step_v: float = 0.8,
    duration_s: float = 2.0e-6,
    dt_s: float = 1.0e-9,
    noise: OpampNoiseConfig | None = None,
) -> tuple[SlewMetrics, TranStepResult, TranStepResult]:
    """Run positive/negative slew steps in Spectre and extract slew rates."""
    netlists_dir = run_spectre_slew_step(
        cfg,
        output_dir,
        step_v=step_v,
        duration_s=duration_s,
        dt_s=dt_s,
        stem="slew_pos",
    )
    pos = overlay_transient_noise_on_step(
        _spectre_tran_step_from_netlists(netlists_dir, stem="slew_pos"),
        cfg,
        noise,
    )
    run_spectre_slew_step(
        cfg,
        output_dir,
        step_v=-step_v,
        duration_s=duration_s,
        dt_s=dt_s,
        stem="slew_neg",
    )
    neg = overlay_transient_noise_on_step(
        _spectre_tran_step_from_netlists(netlists_dir, stem="slew_neg"),
        cfg,
        noise,
    )
    vfinal_pos = float(
        np.clip(cfg.vcm_v + step_v, cfg.vswing_low_v, cfg.vswing_high_v)
    )
    vfinal_neg = float(
        np.clip(cfg.vcm_v - step_v, cfg.vswing_low_v, cfg.vswing_high_v)
    )
    sr_pos, _ = extract_slew_rate(pos["time_s"], pos["vout_clean_v"], vfinal_pos)
    _, sr_neg = extract_slew_rate(neg["time_s"], neg["vout_clean_v"], vfinal_neg)
    metrics = SlewMetrics(
        slew_pos_vps=float(sr_pos),
        slew_neg_vps=float(sr_neg),
    )
    return metrics, pos, neg


def render_spectre_thd_netlist(
    template_path: Path,
    cfg: OpampConfig,
    *,
    amplitude_v: float,
    freq_hz: float,
    cycles: float,
    dt_s: float,
    repo_root: Path | None = None,
) -> str:
    """Render Spectre open-loop THD transient netlist."""
    _ = repo_root
    duration_s = cycles / freq_hz
    a0 = max(cfg.a0_linear, 1.0)
    poly_expr = (
        f"{_format_param(a0)}*(v(inp)-v(inn))"
        f"+{_format_param(cfg.nl_a2)}*(v(inp)-v(inn))*(v(inp)-v(inn))"
        f"+{_format_param(cfg.nl_a3)}*(v(inp)-v(inn))*(v(inp)-v(inn))*(v(inp)-v(inn))"
    )
    return (
        template_path.read_text(encoding="utf-8")
        .replace("PLACEHOLDER_VCM", _format_param(cfg.vcm_v))
        .replace("PLACEHOLDER_AMP", _format_param(amplitude_v))
        .replace("PLACEHOLDER_FREQ", _format_param(freq_hz))
        .replace("PLACEHOLDER_POLY_EXPR", poly_expr)
        .replace("PLACEHOLDER_TSTOP", _format_param(duration_s))
        .replace("PLACEHOLDER_TSTEP", _format_param(dt_s))
    )


def run_spectre_thd(
    cfg: OpampConfig,
    output_dir: Path,
    *,
    amplitude_v: float,
    freq_hz: float,
    cycles: float,
    dt_s: float,
    template_name: str = "thd_open_loop.scs",
) -> Path:
    """Run Spectre THD transient and return the netlist directory."""
    repo = package_root()
    template = repo / "testbench" / "spectre" / template_name
    logs_dir = output_dir / "logs" / "netlists"
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = logs_dir / template_name
    netlist_path.write_text(
        render_spectre_thd_netlist(
            template,
            cfg,
            amplitude_v=amplitude_v,
            freq_hz=freq_hz,
            cycles=cycles,
            dt_s=dt_s,
            repo_root=repo,
        ),
        encoding="utf-8",
    )
    log_path = output_dir / "logs" / "spectre_thd.log"
    completed = run_spectre_netlist(netlist_path, cwd=logs_dir, psf_format="psfascii")
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    _raise_if_spectre_failed(
        completed,
        log_path=log_path,
        message="Spectre THD TRAN failed",
    )
    return logs_dir


def simulate_thd_spectre(
    cfg: OpampConfig,
    output_dir: Path,
    *,
    amplitude_v: float,
    freq_hz: float,
    cycles: float,
    dt_s: float,
    ideal_flag: bool = False,
    noise: OpampNoiseConfig | None = None,
) -> tuple[TranSineResult, ThdMetrics | None]:
    """Run Spectre THD transient and compute distortion metrics from PSF."""
    netlists_dir = run_spectre_thd(
        cfg,
        output_dir,
        amplitude_v=amplitude_v,
        freq_hz=freq_hz,
        cycles=cycles,
        dt_s=dt_s,
    )
    data = read_spectre_tran_from_netlists(
        netlists_dir,
        stem="thd_open_loop",
        signal="out",
    )
    zeros = np.zeros_like(data["signal_v"], dtype=np.float64)
    sine = TranSineResult(
        time_s=data["time_s"],
        vout_v=data["signal_v"],
        vout_clean_v=data["signal_v"],
        noise_v=zeros,
    )
    sine = overlay_transient_noise_on_sine(
        sine,
        cfg,
        noise,
        amplitude_v=amplitude_v,
        freq_hz=freq_hz,
    )
    thd = None if is_thd_ideal(cfg, ideal_flag=ideal_flag) else compute_thd(
        sine["time_s"],
        sine["vout_v"],
        freq_hz,
    )
    return sine, thd


def run_spectre_tran_stub(
    cfg: OpampConfig,
    output_dir: Path,
    *,
    template_name: str,
    log_name: str,
) -> Path:
    """Run a minimal Spectre TRAN stub (``dc`` only) to validate the toolchain."""
    repo = package_root()
    template = repo / "testbench" / "spectre" / template_name
    logs_dir = output_dir / "logs" / "netlists"
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = logs_dir / template_name
    netlist_path.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    log_path = output_dir / "logs" / log_name
    completed = run_spectre_netlist(netlist_path, cwd=logs_dir)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    _raise_if_spectre_failed(
        completed,
        log_path=log_path,
        message="Spectre TRAN stub failed",
    )
    _ = cfg
    return log_path
