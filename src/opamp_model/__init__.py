"""opamp-model: configurable op-amp, TIA, and Gm behavioral models."""

from opamp_model.ac import AcMetrics, extract_gbw_phase_margin, plot_bode
from opamp_model.config import (
    BenchSweepConfig,
    GmConfig,
    OpampConfig,
    OpampNoiseConfig,
    TiaConfig,
)
from opamp_model.cm_ps import simulate_cmrr, simulate_psrr
from opamp_model.model import simulate_ac, simulate_noise, simulate_stb
from opamp_model.tia import simulate_tia_ac

__all__ = [
    "AcMetrics",
    "BenchSweepConfig",
    "GmConfig",
    "OpampConfig",
    "OpampNoiseConfig",
    "TiaConfig",
    "extract_gbw_phase_margin",
    "plot_bode",
    "simulate_ac",
    "simulate_cmrr",
    "simulate_noise",
    "simulate_psrr",
    "simulate_stb",
    "simulate_tia_ac",
]

__version__ = "0.1.0"
