# Testbench specification

Self-contained bench definitions for **opamp-model**. Metric names and default sweeps were derived from a historical Cadence characterization flow; no external library or schematic is required to run this package.

## Multi-engine rule

Each engine (**python**, **ngspice**, **spectre**) must run its **own** implementation of the macromodel in [MODEL.md](MODEL.md). No engine is “golden”; reported metrics must come from that engine’s simulation and extractor. `compare_engines.py` (planned) will report cross-engine spread and optional comparison to `golden_metrics.yaml` (transistor-level reference only).

Until Spectre PSF and full ngspice noise parsers land, some benches document **scaffolding** behavior in the engine matrix below.

## Default frequency sweep (AC / STB / noise / PSRR)

Configured via ``OpampConfig.sweep`` (``BenchSweepConfig``):

| Parameter | Default | CLI override |
| --- | --- | --- |
| Start | 1 Hz | (fixed in config today) |
| Stop | 100 MHz | (fixed in config today) |
| Scale | Logarithmic | — |
| Points per decade | 10 | — |

Spectre netlists receive ``f_start``, ``f_stop``, ``dec`` from ``OpampConfig`` at render time.

## Engine support matrix

| Bench | Script | python | ngspice | spectre |
| --- | --- | --- | --- | --- |
| AC open-loop | `run_ac.py` | Full | ``ac_open_loop.cir`` + ``wrdata`` | Netlist + Python Bode (PSF TBD) |
| STB loop gain | `run_stb.py` | Full | AC netlist + ``loop_beta`` in Python | Netlist + Python Bode (PSF TBD) |
| Noise | `run_noise.py` | Full | Stub + Python spectrum | Stub + Python spectrum |
| PSRR | `run_psrr.py` | Full | Python macromodel | Python macromodel |
| Slew rate | `run_slew.py` | Full | — | — |
| THD | `run_thd.py` | Full | — | — |

Batch runner ``scripts/run_all_simulations.sh`` invokes AC/STB/noise/PSRR for all three engines (when binaries exist) and slew/THD for **python** only.

## Benches

### AC (`run_ac.py`)

| Output | Definition | Unit |
| --- | --- | --- |
| `Aol_Gain` | Open-loop differential gain | dB |
| `ACM` | Common-mode gain | dB |
| `CMRR` | ``Aol - ACM`` (dB) | dB |
| `Zin`, `Zout` | ``|Rin‖Cin|``, ``|Rout‖Cout|`` vs f | Ω |

**Artifacts:** ``ac_bode.csv``, ``ac_bode.svg``, ``cmrr.csv``, ``cmrr.svg``, ``impedance.svg``, ``AC_REPORT.md``.

Historical Cadence extraction (documentation only):

- `Aol_Gain`: `dB20((VF("/OUTP") / VF("/VIN1")))`
- `ACM`: `dB20((VF("/OUTP1") / VF("/VIN1")))`
- `CMRR`: `dB20((VF("/OUTP") / VF("/VIN1")) / (VF("/OUTP1") / VF("/VIN1")))`

**Netlists:** ``testbench/ngspice/ac_open_loop.cir``, ``testbench/spectre/ac_open_loop.scs``.

### STB (`run_stb.py`)

| Output | Definition |
| --- | --- |
| Loop gain magnitude | dB vs frequency |
| Loop gain phase | degrees vs frequency |
| GBW | Frequency where \|loop gain\| = 0 dB |
| Phase margin | Phase at GBW + 180° (see `MODEL.md`) |
| Gain margin | dB at −180° phase crossing |

``loop_beta`` scales open-loop gain in dB after simulation for ngspice/Spectre (same AC netlist as open-loop).

**Artifacts:** ``stb_bode.csv``, ``stb_bode.svg``, ``STB_REPORT.md``.

**Netlists:** ``testbench/ngspice/stb_loop.cir``, ``testbench/spectre/stb_loop.scs``.

### Noise (`run_noise.py`)

| Output | Definition | Unit |
| --- | --- | --- |
| Output noise spectrum | ``e_out(f) = e_n(f) · |A_ol(f)|`` | V/√Hz |
| Input-referred spot @ 1 kHz | Interpolated ``e_n`` | V/√Hz |
| Integrated noise | RMS over analysis band | V |

**Artifacts:** ``noise_spectrum.csv``, ``noise_spectrum.svg``, ``NOISE_REPORT.md``.

**Netlists (stub):** ``testbench/ngspice/noise_stub.cir``, ``testbench/spectre/noise_stub.scs``.

### PSRR (`run_psrr.py`)

| Output | Definition | Unit |
| --- | --- | --- |
| PSRR | ``-20 log10 |H_ps(jω)|`` | dB vs f |

**Artifacts:** ``psrr.csv``, ``psrr.svg``, ``PSRR_REPORT.md``.

**Netlists (stub):** ``testbench/spectre/psrr.scs`` (supply AC stimulus TBD in Verilog-A).

### Slew rate (`run_slew.py`)

| Output | Definition | Unit |
| --- | --- | --- |
| SR+ / SR− | Max dVout/dt on unity-gain step (10–90 %) | V/s |

**Artifacts:** ``slew_step.csv``, ``slew.svg``, ``SLEW_REPORT.md``.

**Engine:** Python only.

### THD (`run_thd.py`)

| Output | Definition |
| --- | --- |
| THD | Total harmonic distortion on sinusoidal steady state |
| HD2, HD3 | Individual harmonic levels |

Disabled when ``--ideal`` or ``nl_a2`` / ``nl_a3`` are zero.

**Artifacts:** ``thd_waveform.csv``, ``thd_spectrum.csv``, ``thd_waveform.svg``, ``thd_spectrum.svg``, ``THD_REPORT.md``.

**Engine:** Python only.

### TIA (`run_tia_*.py`) — planned

| Output | Definition | Unit |
| --- | --- | --- |
| `Zt` | Transimpedance Vout/Iin (or Vout/Vin for voltage TIA) | Ω or V/V |
| Bandwidth | −3 dB frequency of `Zt` | Hz |
| Noise | Input-referred current or voltage noise | A/√Hz or V/√Hz |

### Gm (`run_gm_*.py`) — planned

| Output | Definition | Unit |
| --- | --- | --- |
| `gm` | Transconductance | S |
| Rout | Output resistance | Ω |
| Noise | Input-referred voltage noise | V/√Hz |

### AC compensation / distortion (`run_ac_comp.py`) — planned

Not implemented. See [metrics_catalog.md](metrics_catalog.md).

## Unified outputs

Each engine directory ``outputs/<engine>/`` accumulates:

| File | Content |
| --- | --- |
| `opamp_metrics.json` | Scalar metrics by section (merged across benches) |
| `AC_REPORT.md`, `STB_REPORT.md`, … | Per-bench Markdown + embedded SVG |
| `REPORT.md` | Index of bench reports and figures |
| `logs/` | Simulation logs, rendered netlists, Verilog-A copies |

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
