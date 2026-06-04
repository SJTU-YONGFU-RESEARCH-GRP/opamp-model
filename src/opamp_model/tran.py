"""Transient analysis: slew rate and THD (Phase 2+)."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from opamp_model.config import OpampConfig, OpampNoiseConfig
from opamp_model.plot_style import (
    FIGSIZE,
    LINE_COLORS,
    LINEWIDTH_MAIN,
    LINEWIDTH_SECONDARY,
    apply_rcparams,
    apply_style,
    downsample_stride,
)


class TranStepResult(TypedDict):
    """Large-signal unity-gain step response."""

    time_s: NDArray[np.float64]
    vout_v: NDArray[np.float64]


class SlewMetrics(TypedDict):
    """Measured slew rates from transient step tests."""

    slew_pos_vps: float
    slew_neg_vps: float


class ThdMetrics(TypedDict):
    """Harmonic distortion scalars (``run_thd.py``)."""

    thd_db: float
    hd2_db: float
    hd3_db: float
    fundamental_hz: float
    fundamental_v: float


def is_thd_ideal(cfg: OpampConfig, *, ideal_flag: bool = False) -> bool:
    """Return True when weak nonlinearity is disabled (``--ideal`` or zero NL coeffs)."""
    if ideal_flag:
        return True
    return cfg.nl_a2 == 0.0 and cfg.nl_a3 == 0.0


def simulate_step_response(
    cfg: OpampConfig,
    noise: OpampNoiseConfig | None = None,
    *,
    step_v: float,
    duration_s: float,
    dt_s: float,
) -> TranStepResult:
    """Simulate unity-gain voltage-follower step response with slew and clipping.

    The output tracks ``vcm + step_v`` via a first-order loop ``dV/dt = 2π·GBW·(V_in − V_out)``
    with magnitude limited to ``slew_pos_vps`` / ``slew_neg_vps``, then clamped to
    ``vswing_low_v`` … ``vswing_high_v``. Integration uses explicit Euler steps.

    Args:
        cfg: Op-amp macromodel parameters.
        noise: Reserved for Phase 3 (ignored).
        step_v: Input step amplitude (V) applied at ``t = 0`` relative to ``vcm_v``.
        duration_s: Simulation stop time (s).
        dt_s: Time step (s).

    Returns:
        Time vector and output voltage samples.
    """
    _ = noise
    if duration_s <= 0.0 or dt_s <= 0.0:
        msg = f"Invalid transient setup: duration={duration_s}, dt={dt_s}"
        raise ValueError(msg)

    n = max(int(np.ceil(duration_s / dt_s)) + 1, 2)
    time_s = (np.arange(n, dtype=np.float64) * dt_s).astype(np.float64)
    vout = np.empty(n, dtype=np.float64)

    vcm = cfg.vcm_v
    vin_target = vcm + step_v
    vout[0] = vcm
    omega_ug = 2.0 * np.pi * cfg.gbw_hz
    slew_pos = abs(cfg.slew_pos_vps)
    slew_neg = cfg.slew_neg_vps if cfg.slew_neg_vps < 0.0 else -abs(cfg.slew_neg_vps)

    for idx in range(n - 1):
        vin = vin_target if time_s[idx] >= 0.0 else vcm
        err = vin - vout[idx]
        dv_ideal = omega_ug * err * dt_s
        dv = np.clip(dv_ideal, slew_neg * dt_s, slew_pos * dt_s)
        vout[idx + 1] = np.clip(
            vout[idx] + dv,
            cfg.vswing_low_v,
            cfg.vswing_high_v,
        )

    return TranStepResult(time_s=time_s, vout_v=vout)


def _interp_crossing_time(
    time_s: NDArray[np.float64],
    y_v: NDArray[np.float64],
    level_v: float,
    *,
    rising: bool,
) -> float:
    """Linearly interpolate the first time ``y_v`` crosses ``level_v``."""
    t = time_s.astype(np.float64)
    y = y_v.astype(np.float64)
    for idx in range(len(y) - 1):
        y0, y1 = y[idx], y[idx + 1]
        if rising:
            crossed = y0 < level_v <= y1 or (y0 <= level_v < y1 and y1 != y0)
        else:
            crossed = y0 > level_v >= y1 or (y0 >= level_v > y1 and y1 != y0)
        if crossed and y1 != y0:
            frac = (level_v - y0) / (y1 - y0)
            return float(t[idx] + frac * (t[idx + 1] - t[idx]))
    return float("nan")


def extract_slew_rate(
    time_s: NDArray[np.float64],
    vout_v: NDArray[np.float64],
    vfinal_v: float,
) -> tuple[float, float]:
    """Extract SR+ and SR− from a step waveform using 10–90 % crossings.

    Slew rate is ``(V90 − V10) / (t90 − t10)`` where ``V10`` and ``V90`` are 10 %
    and 90 % of the excursion from the initial output to ``vfinal_v``. For a rising
    edge, ``sr_pos`` is returned and ``sr_neg`` is NaN; for a falling edge the
    opposite. This matches typical SPICE ``slewRate(… 10 90 …)`` definitions.

    Args:
        time_s: Time samples (s).
        vout_v: Output voltage (V).
        vfinal_v: Expected steady-state output after the step (V).

    Returns:
        ``(sr_pos_vps, sr_neg_vps)`` in V/s; unused direction is NaN.
    """
    v0 = float(vout_v[0])
    vf = float(vfinal_v)
    delta = vf - v0
    if abs(delta) < 1.0e-12:
        return float("nan"), float("nan")

    v10 = v0 + 0.1 * delta
    v90 = v0 + 0.9 * delta
    rising = delta > 0.0
    t10 = _interp_crossing_time(time_s, vout_v, v10, rising=rising)
    t90 = _interp_crossing_time(time_s, vout_v, v90, rising=rising)
    if not np.isfinite(t10) or not np.isfinite(t90) or t90 <= t10:
        return float("nan"), float("nan")

    sr = (v90 - v10) / (t90 - t10)
    if rising:
        return float(sr), float("nan")
    return float("nan"), float(sr)


def measure_slew_rates(
    cfg: OpampConfig,
    noise: OpampNoiseConfig | None = None,
    *,
    step_v: float = 0.8,
    duration_s: float = 2.0e-6,
    dt_s: float = 1.0e-9,
) -> tuple[SlewMetrics, TranStepResult, TranStepResult]:
    """Run positive and negative follower steps and return measured slew rates."""
    pos = simulate_step_response(
        cfg,
        noise,
        step_v=step_v,
        duration_s=duration_s,
        dt_s=dt_s,
    )
    vfinal_pos = float(
        np.clip(cfg.vcm_v + step_v, cfg.vswing_low_v, cfg.vswing_high_v)
    )
    sr_pos, _ = extract_slew_rate(pos["time_s"], pos["vout_v"], vfinal_pos)

    neg = simulate_step_response(
        cfg,
        noise,
        step_v=-step_v,
        duration_s=duration_s,
        dt_s=dt_s,
    )
    vfinal_neg = float(
        np.clip(cfg.vcm_v - step_v, cfg.vswing_low_v, cfg.vswing_high_v)
    )
    _, sr_neg = extract_slew_rate(neg["time_s"], neg["vout_v"], vfinal_neg)

    metrics = SlewMetrics(
        slew_pos_vps=float(sr_pos),
        slew_neg_vps=float(sr_neg),
    )
    return metrics, pos, neg


def plot_slew_step(
    pos: TranStepResult,
    neg: TranStepResult,
    output_path: Path,
    *,
    title: str = "Unity-gain step response",
    metrics: SlewMetrics | None = None,
) -> Path:
    """Plot positive and negative step responses and write an SVG."""
    apply_rcparams()
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    t_pos = pos["time_s"]
    t_neg = neg["time_s"]
    stride_pos = max(len(t_pos) // 512, 1)
    stride_neg = max(len(t_neg) // 512, 1)
    ax.plot(
        t_pos[::stride_pos] * 1.0e6,
        pos["vout_v"][::stride_pos],
        color=LINE_COLORS["gain"],
        linewidth=LINEWIDTH_MAIN,
        label="SR+ step",
    )
    ax.plot(
        t_neg[::stride_neg] * 1.0e6,
        neg["vout_v"][::stride_neg],
        color=LINE_COLORS["phase"],
        linewidth=LINEWIDTH_SECONDARY,
        label="SR− step",
    )
    ax.set_xlabel("Time (µs)", fontsize=13)
    ax.set_ylabel("Vout (V)", fontsize=13)
    if metrics is not None:
        subtitle = (
            f"SR+={metrics['slew_pos_vps']:.3g} V/s  "
            f"SR−={metrics['slew_neg_vps']:.3g} V/s"
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=14)
    else:
        ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9)
    apply_style(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _apply_memoryless_nonlinearity(
    vin_v: NDArray[np.float64],
    *,
    a1: float,
    a2: float,
    a3: float,
) -> NDArray[np.float64]:
    """Evaluate Vout = a1*Vin + a2*Vin^2 + a3*Vin^3."""
    return (a1 * vin_v + a2 * vin_v**2 + a3 * vin_v**3).astype(np.float64)


def simulate_sine_response(
    cfg: OpampConfig,
    noise: OpampNoiseConfig | None = None,
    *,
    amplitude_v: float,
    freq_hz: float,
    cycles: float,
    dt_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate open-loop large-signal sine steady state with weak nonlinearity.

    Applies a memoryless polynomial: ``Vout = a1*Vin + a2*Vin^2 + a3*Vin^3``
    with ``a1 = cfg.a0_linear``, ``a2 = cfg.nl_a2``, ``a3 = cfg.nl_a3``.
    """
    _ = noise
    if freq_hz <= 0.0:
        raise ValueError(f"freq_hz must be positive, got {freq_hz}")
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    if cycles <= 0.0:
        raise ValueError(f"cycles must be positive, got {cycles}")

    duration_s = cycles / freq_hz
    n_samples = max(int(np.ceil(duration_s / dt_s)) + 1, 2)
    time_s = np.linspace(0.0, duration_s, n_samples, dtype=np.float64)
    vin_v = amplitude_v * np.sin(2.0 * np.pi * freq_hz * time_s)
    vout_v = _apply_memoryless_nonlinearity(
        vin_v,
        a1=cfg.a0_linear,
        a2=cfg.nl_a2,
        a3=cfg.nl_a3,
    )
    return time_s, vout_v


def _harmonic_amplitude(
    spectrum_v: NDArray[np.float64],
    freqs_hz: NDArray[np.float64],
    harmonic_hz: float,
) -> float:
    """Return peak magnitude (V) near ``harmonic_hz`` in a one-sided spectrum."""
    if harmonic_hz <= 0.0 or len(spectrum_v) == 0:
        return 0.0
    idx = int(np.argmin(np.abs(freqs_hz - harmonic_hz)))
    return float(spectrum_v[idx])


def compute_thd(
    time_s: NDArray[np.float64] | object,
    vout_v: NDArray[np.float64] | object,
    fundamental_hz: float,
    *,
    max_harmonic: int = 10,
    settle_cycles: float = 1.0,
) -> ThdMetrics:
    """Compute THD and HD2/HD3 from a transient waveform via FFT.

    THD (dB) = ``20*log10(sqrt(sum(harmonics^2)) / fundamental)`` using
    one-sided spectral peaks at integer multiples of ``fundamental_hz``.
    """
    t = np.asarray(time_s, dtype=np.float64).ravel()
    y = np.asarray(vout_v, dtype=np.float64).ravel()
    if t.size != y.size or t.size < 4:
        raise ValueError("time_s and vout_v must be same length with >= 4 samples")
    if fundamental_hz <= 0.0:
        raise ValueError(f"fundamental_hz must be positive, got {fundamental_hz}")

    dt = float(np.median(np.diff(t)))
    if dt <= 0.0:
        raise ValueError("time_s must be strictly increasing")

    period_s = 1.0 / fundamental_hz
    settle_s = settle_cycles * period_s
    mask = t >= (t[-1] - (t[-1] - settle_s))
    y_seg = y[mask] - np.mean(y[mask])

    samples_per_period = max(int(round(period_s / dt)), 1)
    n_periods = max(y_seg.size // samples_per_period, 1)
    n_use = n_periods * samples_per_period
    y_win = y_seg[-n_use:]
    window = np.hanning(y_win.size)
    y_windowed = y_win * window
    spectrum = np.abs(np.fft.rfft(y_windowed)) / (0.5 * window.sum())
    freqs_hz = np.fft.rfftfreq(y_win.size, dt)

    v1 = _harmonic_amplitude(spectrum, freqs_hz, fundamental_hz)
    if v1 <= 0.0:
        return ThdMetrics(
            thd_db=float("nan"),
            hd2_db=float("nan"),
            hd3_db=float("nan"),
            fundamental_hz=fundamental_hz,
            fundamental_v=0.0,
        )

    harmonic_v = [
        _harmonic_amplitude(spectrum, freqs_hz, order * fundamental_hz) ** 2
        for order in range(2, max_harmonic + 1)
    ]
    harm_rms = float(np.sqrt(np.sum(harmonic_v)))
    thd_db = 20.0 * np.log10(max(harm_rms / v1, 1.0e-30))

    v2 = _harmonic_amplitude(spectrum, freqs_hz, 2.0 * fundamental_hz)
    v3 = _harmonic_amplitude(spectrum, freqs_hz, 3.0 * fundamental_hz)
    hd2_db = 20.0 * np.log10(max(v2 / v1, 1.0e-30))
    hd3_db = 20.0 * np.log10(max(v3 / v1, 1.0e-30))

    return ThdMetrics(
        thd_db=float(thd_db),
        hd2_db=float(hd2_db),
        hd3_db=float(hd3_db),
        fundamental_hz=fundamental_hz,
        fundamental_v=v1,
    )


def plot_thd_waveform(
    time_s: NDArray[np.float64],
    vout_v: NDArray[np.float64],
    output_path: Path,
    *,
    freq_hz: float,
    title: str = "THD sine response",
) -> Path:
    """Plot output voltage vs time and write an SVG."""
    apply_rcparams()
    stride = downsample_stride(len(time_s))
    t_ms = time_s[::stride] * 1.0e3
    y_plot = vout_v[::stride]
    period_ms = 1.0e3 / freq_hz if freq_hz > 0.0 else 1.0

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(t_ms, y_plot, color=LINE_COLORS["gain"], linewidth=LINEWIDTH_MAIN)
    ax.set_xlabel("Time (ms)", fontsize=13)
    ax.set_ylabel("Vout (V)", fontsize=13)
    ax.set_title(f"{title}\n$f$ = {freq_hz:.6g} Hz, period = {period_ms:.4g} ms", fontsize=14)
    apply_style(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def plot_thd_spectrum(
    time_s: NDArray[np.float64],
    vout_v: NDArray[np.float64],
    fundamental_hz: float,
    output_path: Path,
    *,
    thd: ThdMetrics | None = None,
    title: str = "THD spectrum",
) -> Path:
    """Plot one-sided magnitude spectrum and write an SVG."""
    apply_rcparams()
    t = np.asarray(time_s, dtype=np.float64).ravel()
    y = np.asarray(vout_v, dtype=np.float64).ravel()
    dt = float(np.median(np.diff(t)))
    y_ac = y - np.mean(y)
    window = np.hanning(y_ac.size)
    spectrum = np.abs(np.fft.rfft(y_ac * window)) / (0.5 * window.sum())
    freqs_hz = np.fft.rfftfreq(y_ac.size, dt)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogy(
        freqs_hz[1:],
        spectrum[1:],
        color=LINE_COLORS["psrr"],
        linewidth=LINEWIDTH_SECONDARY,
    )
    ax.axvline(fundamental_hz, color=LINE_COLORS["gain"], linewidth=1.5, linestyle="--")
    ax.set_xlabel("Frequency (Hz)", fontsize=13)
    ax.set_ylabel("|Vout| (V)", fontsize=13)
    subtitle = ""
    if thd is not None and np.isfinite(thd["thd_db"]):
        subtitle = f"THD = {thd['thd_db']:.2f} dB"
    ax.set_title(f"{title}\n{subtitle}".rstrip(), fontsize=14)
    apply_style(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path
