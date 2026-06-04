"""Cadence Spectre AC testbench runner."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.io import package_root
from opamp_model.model import AcSimulationResult, NoiseSimulationResult, simulate_ac, simulate_noise


class SpectreNotFoundError(RuntimeError):
    """Raised when the ``spectre`` executable is not on PATH."""


_SCS_INCLUDE = re.compile(
    r'include\s+"\./testbench/spectre/opamp_include\.scs"\s*',
    re.IGNORECASE,
)


def find_spectre_executable() -> str:
    """Return the Spectre binary path or raise ``SpectreNotFoundError``."""
    found = shutil.which("spectre")
    if not found:
        raise SpectreNotFoundError(
            "Cadence Spectre not found on PATH. Install Spectre or use --simulator python."
        )
    return found


def _format_param(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.12g}"


def _absolutize_spectre_includes(text: str, repo_root: Path) -> str:
    """Replace repo-relative includes with an absolute Verilog-A path.

    Rendered netlists run from ``<output>/logs/netlists/``; relative paths break.
    """
    va_path = (repo_root / "veriloga/configurable_opamp.va").resolve()
    return _SCS_INCLUDE.sub(f'ahdl_include "{va_path}"\n', text)


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


def run_spectre_netlist(
    netlist: Path,
    *,
    cwd: Path | None = None,
    timeout_s: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run Spectre on a rendered ``.scs`` netlist."""
    executable = find_spectre_executable()
    args = [executable, str(netlist), "+log", "status"]
    return subprocess.run(
        args,
        cwd=cwd or netlist.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
    )


def run_spectre_ac(
    cfg: OpampConfig,
    output_dir: Path,
    *,
    template_name: str = "ac_open_loop.scs",
) -> Path:
    """Render and execute a Spectre AC netlist; return log path."""
    repo = package_root()
    template = repo / "testbench" / "spectre" / template_name
    logs_dir = output_dir / "logs" / "netlists"
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = logs_dir / template_name
    netlist_path.write_text(
        render_spectre_ac_netlist(template, cfg, repo_root=repo),
        encoding="utf-8",
    )
    log_path = output_dir / "logs" / f"spectre_{template_name.replace('.scs', '')}.log"
    completed = run_spectre_netlist(netlist_path, cwd=logs_dir)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        msg = f"Spectre failed with code {completed.returncode}; see {log_path}"
        raise RuntimeError(msg)
    return log_path


def simulate_ac_spectre(
    cfg: OpampConfig,
    output_dir: Path,
    noise: OpampNoiseConfig | None = None,
) -> AcSimulationResult:
    """Run Spectre AC; metrics must come from Spectre results (PSF parser).

    Temporary: when no PSF parser is available, Python curves are substituted
    after the netlist run. This is scaffolding only — not peer-engine behavior.
    """
    _ = noise
    run_spectre_ac(cfg, output_dir)
    return simulate_ac(cfg, noise)


def run_spectre_noise_stub(cfg: OpampConfig, output_dir: Path) -> Path:
    """Render and execute a Spectre noise stub netlist (Phase 3)."""
    repo = package_root()
    template = repo / "testbench" / "spectre" / "noise_stub.scs"
    logs_dir = output_dir / "logs" / "netlists"
    logs_dir.mkdir(parents=True, exist_ok=True)
    netlist_path = logs_dir / "noise_stub.scs"
    netlist_path.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    log_path = output_dir / "logs" / "spectre_noise_stub.log"
    completed = run_spectre_netlist(netlist_path, cwd=logs_dir)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        msg = f"Spectre noise stub failed with code {completed.returncode}; see {log_path}"
        raise RuntimeError(msg)
    _ = cfg
    return log_path


def simulate_noise_spectre(
    cfg: OpampConfig,
    output_dir: Path,
    noise: OpampNoiseConfig,
) -> NoiseSimulationResult:
    """Run Spectre noise bench; spectrum must come from Spectre (stub today)."""
    run_spectre_noise_stub(cfg, output_dir)
    return simulate_noise(cfg, noise)
