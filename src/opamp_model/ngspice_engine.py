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
from opamp_model.io import package_root, read_ngspice_ac_wrdata, write_bode_csv
from opamp_model.model import AcSimulationResult, NoiseSimulationResult, simulate_noise


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


def run_ngspice_noise_stub(cfg: OpampConfig, output_dir: Path) -> Path:
    """Run an ngspice noise stub netlist."""
    repo = package_root()
    template = repo / "testbench" / "ngspice" / "noise_stub.cir"
    ng_dir = output_dir / "ngspice"
    logs_dir = output_dir / "logs"
    ng_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = ng_dir / "noise_stub.cir"
    netlist_path.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    log_path = logs_dir / "ngspice_noise_stub.log"
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
        msg = f"ngspice noise stub failed with code {completed.returncode}; see {log_path}"
        raise RuntimeError(msg)
    _ = cfg
    return log_path


def simulate_noise_ngspice(
    cfg: OpampConfig,
    output_dir: Path,
    noise: OpampNoiseConfig,
) -> NoiseSimulationResult:
    """Run ngspice noise bench; spectrum must come from ngspice (stub today)."""
    run_ngspice_noise_stub(cfg, output_dir)
    return simulate_noise(cfg, noise)


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
