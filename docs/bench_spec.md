# Testbench specification

Self-contained bench definitions for **opamp-model**. Metric names and default sweeps were derived from a historical Cadence characterization flow; no external library or schematic is required to run this package.

## Multi-engine rule

Each engine (**python**, **ngspice**, **spectre**) runs its **own** implementation of the macromodel in [MODEL.md](MODEL.md). No engine is “golden”; reported metrics must come from that engine’s simulation and extractor. `compare_engines.py` (planned) will report cross-engine spread and optional comparison to `golden_metrics.yaml` (transistor-level reference only).

## Default frequency sweep (AC / STB / noise / PSRR)

| Parameter | Value |
| --- | --- |
| Start | 1 Hz |
| Stop | 100 MHz |
| Scale | Logarithmic |
| Points per decade | 10 |

## Benches

### AC (`run_ac.py`)

| Output | Definition | Unit |
| --- | --- | --- |
| `Aol_Gain` | Open-loop differential gain | dB |
| `ACM` | Differential closed-loop or auxiliary diff gain | dB |
| `CMRR` | Ratio of differential gain to common-mode gain | dB |

Historical extraction (documentation only):

- `Aol_Gain`: `dB20((VF("/OUTP") / VF("/VIN1")))`
- `ACM`: `dB20((VF("/OUTP1") / VF("/VIN1")))`
- `CMRR`: `dB20((VF("/OUTP") / VF("/VIN1")) / (VF("/OUTP1") / VF("/VIN1")))`

### STB (`run_stb.py`)

| Output | Definition |
| --- | --- |
| Loop gain magnitude | dB vs frequency |
| Loop gain phase | degrees vs frequency |
| GBW | Frequency where \|loop gain\| = 0 dB |
| Phase margin | Phase at GBW + 180° (sign per convention in `MODEL.md`) |
| Gain margin | dB at 0° phase crossing |

Probe notes (documentation): differential STB probe on loop break, local ground at negative supply.

### Noise (`run_noise.py`)

| Output | Definition | Unit |
| --- | --- | --- |
| Output noise spectrum | Spot noise at output | V/√Hz |
| Integrated noise | RMS over analysis band | V |

### PSRR (`run_psrr.py`)

| Output | Definition | Unit |
| --- | --- | --- |
| PSRR | Supply ripple to output transfer | dB vs f |

### Slew rate (`run_slew.py`)

| Output | Definition | Unit |
| --- | --- | --- |
| SR+ / SR− | Max dVout/dt on step response | V/s |

### THD (`run_thd.py`)

| Output | Definition |
| --- | --- |
| THD | Total harmonic distortion on sinusoidal steady state |
| HD2, HD3 | Individual harmonic levels (optional) |

### AC compensation / distortion (`run_ac_comp.py`)

| Output | Definition |
| --- | --- |
| Peaking | Gain peaking near GBW / compensation null |
| Distortion terms | Small-signal distortion near band edge |

### TIA (`run_tia_*.py`)

| Output | Definition | Unit |
| --- | --- | --- |
| `Zt` | Transimpedance Vout/Iin (or Vout/Vin for voltage TIA) | Ω or V/V |
| Bandwidth | −3 dB frequency of `Zt` | Hz |
| Noise | Input-referred current or voltage noise | A/√Hz or V/√Hz |

### Gm (`run_gm_*.py`)

| Output | Definition | Unit |
| --- | --- | --- |
| `gm` | Transconductance | S |
| Rout | Output resistance | Ω |
| Noise | Input-referred voltage noise | V/√Hz |

## Port naming (documentation)

Historical net names used only in docs and golden targets:

| Net | Role |
| --- | --- |
| `/VIN1` | Differential input reference |
| `/OUTP` | Main output |
| `/OUTP1` | Auxiliary / CM test output |
| `/VSS` | Negative supply / ground |
| `/IPRB0` | STB injection probe |

Behavioral models use abstract ports `inp`, `inn`, `out`, `vdd`, `vss`.
