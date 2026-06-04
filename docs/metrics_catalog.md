# Op-amp metrics catalog

Target figures of merit for **opamp-model**, mapped to benches and implementation status.

## Status legend

| Status | Meaning |
| --- | --- |
| **reported** | Written to `*_metrics.json` / CLI today |
| **param** | Taken from macromodel parameters (not a separate simulation) |
| **planned** | Bench or model path documented here; not wired yet |

## Small-signal / AC / STB

| Metric | Unit | Bench | Status |
| --- | --- | --- | --- |
| Open-loop gain `A0` | dB | AC | **reported** |
| Unity-gain BW `GBW` | Hz | AC / STB | **reported** |
| Phase margin `PM` | deg | STB | **reported** |
| Gain margin `GM` | dB | STB | **reported** (may be NaN if no −180° crossing) |
| Gain peaking | dB | AC | **planned** (`run_ac_comp.py`) |
| Second pole / zero fit | Hz | AC | **planned** (optional fit from Bode) |

## Impedance

| Metric | Unit | Bench | Status |
| --- | --- | --- | --- |
| Differential `Zin` | Ω | AC / param | **param** + **reported** \|Zin(1 Hz)\| from `Rin`∥`Cin` |
| Output `Zout` | Ω | AC / param | **param** + **reported** \|Zout(1 Hz)\| from `Rout`∥`Cout` |
| `Zin(f)`, `Zout(f)` curves | Ω | AC | **planned** (CSV/SVG export) |

## CMRR / PSRR

| Metric | Unit | Bench | Status |
| --- | --- | --- | --- |
| `CMRR` at DC | dB | AC | **reported** (`run_ac.py`, `cmrr_bench`) |
| `ACM` at DC | dB | AC | **reported** (`run_ac.py`) |
| `CMRR(f)`, `ACM(f)` curves | dB | AC | **reported** (`cmrr.csv`, `cmrr.svg`) |
| `PSRR` at DC | dB | PSRR | **reported** (`run_psrr.py`, `psrr_bench`) |
| `PSRR` vs `f` | dB | PSRR | **reported** (`psrr.csv`, `psrr.svg`) |

## Noise

| Metric | Unit | Bench | Status |
| --- | --- | --- | --- |
| Input-referred `en` spot @1 kHz | V/√Hz | Noise | **reported** (`run_noise.py`) |
| Output noise spot @1 kHz | V/√Hz | Noise | **reported** |
| Integrated noise RMS | V | Noise | **reported** |
| Flicker corner | Hz | Noise | **param** (`en_flicker_corner_hz`) |

## Large-signal / distortion

| Metric | Unit | Bench | Status |
| --- | --- | --- | --- |
| Slew rate SR+/SR− | V/s | TRAN | **reported** (`run_slew.py`, 10–90 % step) |
| Output swing | V | — | **param** |
| THD / HD2 / HD3 | dB | TRAN | **reported** (`run_thd.py`, Python engine) |

## TIA / Gm (derived)

| Metric | Unit | Bench | Status |
| --- | --- | --- | --- |
| Transimpedance `Zt` | Ω | TIA AC | **planned** (TIA benches) |
| `gm` | S | Gm AC | **planned** (Gm benches) |

## Unified report file

Each engine directory contains:

```text
outputs/<engine>/AC_REPORT.md       # open-loop Bode + impedance figures + metrics
outputs/<engine>/STB_REPORT.md        # loop-gain Bode + metrics
outputs/<engine>/REPORT.md            # index linking both benches + embedded figures
outputs/<engine>/opamp_metrics.json   # all available scalars + status
outputs/<engine>/*.svg                # ac_bode, stb_bode, impedance
```
