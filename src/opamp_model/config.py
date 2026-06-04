"""Configuration dataclasses for op-amp, TIA, and Gm macromodels."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchSweepConfig:
    """Log-spaced frequency sweep defaults (AC, STB, noise, PSRR)."""

    f_start_hz: float = 1.0
    f_stop_hz: float = 100.0e6
    points_per_decade: int = 10

    @property
    def num_decades(self) -> float:
        """Return number of frequency decades in the sweep."""
        if self.f_start_hz <= 0.0 or self.f_stop_hz <= 0.0:
            return 0.0
        return math.log10(self.f_stop_hz / self.f_start_hz)


@dataclass(frozen=True)
class OpampConfig:
    """Small- and large-signal parameters for the voltage op-amp core."""

    a0_db: float = 80.0
    gbw_hz: float = 10.0e6
    fp2_hz: float = 200.0e6
    fz_hz: float = 0.0
    cmrr_db: float = 90.0
    psrr_db: float = 80.0
    psrr_pole_hz: float = 100.0
    rin_ohm: float = 1.0e12
    cin_f: float = 1.0e-12
    rout_ohm: float = 100.0
    cout_f: float = 2.0e-12
    vswing_high_v: float = 1.0
    vswing_low_v: float = -1.0
    icl_max_a: float = 10.0e-3
    slew_pos_vps: float = 10.0e6
    slew_neg_vps: float = -10.0e6
    nl_a2: float = 0.0
    nl_a3: float = 0.0
    vdd_v: float = 1.8
    vss_v: float = 0.0
    vcm_v: float = 0.9
    loop_beta: float = 1.0
    sweep: BenchSweepConfig = field(default_factory=BenchSweepConfig)

    @property
    def a0_linear(self) -> float:
        """Return open-loop voltage gain (V/V)."""
        return 10.0 ** (self.a0_db / 20.0)

    @property
    def cmrr_linear(self) -> float:
        """Return common-mode rejection ratio (V/V)."""
        return 10.0 ** (self.cmrr_db / 20.0)

    @property
    def psrr_linear(self) -> float:
        """Return power-supply rejection ratio (V/V)."""
        return 10.0 ** (self.psrr_db / 20.0)


@dataclass(frozen=True)
class OpampNoiseConfig:
    """Input- and output-referred noise parameters."""

    en_white_v_per_sqrt_hz: float = 5.0e-9
    en_flicker_corner_hz: float = 100.0
    en_flicker_at_1hz_v_per_sqrt_hz: float = 50.0e-9
    in_white_a_per_sqrt_hz: float = 0.0
    in_flicker_corner_hz: float = 100.0
    in_flicker_at_1hz_a_per_sqrt_hz: float = 0.0
    noise_seed: int = 1

    @property
    def enabled(self) -> bool:
        """Return True when any noise mechanism is active."""
        return (
            self.en_white_v_per_sqrt_hz > 0.0
            or self.en_flicker_at_1hz_v_per_sqrt_hz > 0.0
            or self.in_white_a_per_sqrt_hz > 0.0
            or self.in_flicker_at_1hz_a_per_sqrt_hz > 0.0
        )


@dataclass(frozen=True)
class TiaConfig:
    """Transimpedance amplifier feedback network."""

    rf_ohm: float = 100.0e3
    cf_f: float = 100.0e-15
    cs_f: float = 0.0
    current_input: bool = True


@dataclass(frozen=True)
class GmConfig:
    """Transconductance stage parameters."""

    gm_s: float = 1.0e-3
    rout_ohm: float = 1.0e6
    cout_f: float = 500.0e-15
