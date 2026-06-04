# opamp-model

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-green?logo=creativecommons&logoColor=white)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776ab.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-blue?logo=semver&logoColor=white)](https://github.com/SJTU-YONGFU-RESEARCH-GRP/amplifier-model)

Configurable **voltage op-amp**, **transimpedance (TIA)**, and **transconductance (Gm)** behavioral models with a Python package, Verilog-A shells, and SPICE/Spectre testbenches. Peer engines (Python, ngspice, Spectre) implement the same macromodel equations documented in [docs/MODEL.md](docs/MODEL.md).

**Repository:** [SJTU-YONGFU-RESEARCH-GRP/amplifier-model](https://github.com/SJTU-YONGFU-RESEARCH-GRP/amplifier-model) (this package lives in the `opamp-model/` directory)

- **License:** CC BY 4.0 (see [LICENSE](LICENSE))
- **Planned:** TIA/Gm benches, Spectre PSF / full ngspice noise parsers (see [docs/metrics_catalog.md](docs/metrics_catalog.md))

**Status:** Full macromodel on **python** (AC/STB, noise, CMRR/PSRR, slew, THD). **ngspice** parses AC/STB Bode from netlists; **spectre** runs decks but AC/noise curves are still Python-backed until PSF parsers land. Details: [docs/README.md](docs/README.md#quick-reference).

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
  - [Run a single bench](#run-a-single-bench)
  - [Run all benches](#run-all-benches)
  - [Engine support (summary)](#engine-support-summary)
- [Python API](#python-api)
- [Scripts and CLI](#scripts-and-cli)
- [Multi-engine workflow](#multi-engine-workflow)
- [Metrics and outputs](#metrics-and-outputs)
- [Project layout](#project-layout)
- [Documentation](#documentation)
- [Development](#development)
- [Independence](#independence)
- [License](#license)

## Features

| Area | What is included |
| --- | --- |
| **Models** | Voltage op-amp (dominant-pole core, CMRR/PSRR, impedance, noise, slew, weak nonlinearity); TIA/Gm configs and Verilog-A shells (closed-loop **planned**) |
| **Engines** | `python` (closed-form / numerical), `ngspice` (`testbench/ngspice/`, AC `wrdata`), `spectre` (`veriloga/` + `testbench/spectre/`) |
| **Benches** | `run_ac.py`, `run_stb.py`, `run_noise.py`, `run_psrr.py`, `run_slew.py`, `run_thd.py` |
| **Artifacts** | Per-bench `*_REPORT.md`, SVG plots, CSV sweeps, `opamp_metrics.json`, engine-level `REPORT.md` |

No engine is treated as golden: each simulator must produce its own curves and metrics (see [docs/bench_spec.md](docs/bench_spec.md)).

## Requirements

- **Python** 3.10+ (NumPy, Matplotlib, PyYAML — installed via `pyproject.toml`)
- **Optional:** [ngspice](https://ngspice.sourceforge.io/) on `PATH` for `testbench/ngspice/` netlists
- **Optional:** Cadence Spectre on `PATH` for `testbench/spectre/` and `veriloga/configurable_opamp.va`

## Installation

```bash
cd opamp-model
./scripts/install_python.sh
source .venv/bin/activate
```

Runtime-only install (no pytest/ruff):

```bash
./scripts/install_python.sh --no-dev
```

Editable install without the helper script:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```bash
pytest
./scripts/check_independence.sh
./scripts/run_all_simulations.sh --skip-missing
```

Open `outputs/<engine>/REPORT.md` for Markdown reports with embedded SVG figures.

### Run a single bench

Default engine is Python; results go to `outputs/python/` unless `--output-dir` is set.

```bash
python scripts/run_ac.py --output-dir outputs/python
python scripts/run_stb.py --output-dir outputs/python
python scripts/run_noise.py --output-dir outputs/python
python scripts/run_psrr.py --output-dir outputs/python
python scripts/run_slew.py --output-dir outputs/python
python scripts/run_thd.py --output-dir outputs/python
```

Example with ngspice (requires `ngspice` on `PATH`):

```bash
python scripts/run_ac.py --simulator ngspice --output-dir outputs/ngspice
```

Common macromodel overrides (defaults match [docs/golden_metrics.yaml](docs/golden_metrics.yaml)):

```bash
python scripts/run_ac.py --a0-db 80 --gbw-hz 10e6 --cmrr-db 90 --ideal
```

### Run all benches

```bash
./scripts/run_all_simulations.sh --skip-missing
```

Options: `--output-root DIR`, `--skip-missing` (skip ngspice/Spectre when binaries are absent), `--ideal` (linear small-signal only). Transient benches (slew, THD) run on the **python** engine only.

### Engine support (summary)

| Bench | python | ngspice | spectre |
| --- | --- | --- | --- |
| AC / STB | Full | Parsed Bode | Netlist + Python Bode (PSF TBD) |
| Noise | Full | Stub + Python spectrum | Stub + Python spectrum |
| PSRR | Full | Python macromodel | Python macromodel |
| Slew / THD | Full | — | — |

See [docs/bench_spec.md](docs/bench_spec.md) for artifacts and netlist paths.

## Python API

```python
from opamp_model import (
    OpampConfig,
    OpampNoiseConfig,
    simulate_ac,
    simulate_stb,
    simulate_cmrr,
    simulate_psrr,
    simulate_noise,
)

cfg = OpampConfig(a0_db=80.0, gbw_hz=10e6)
noise = OpampNoiseConfig()
ac = simulate_ac(cfg, noise)
stb = simulate_stb(cfg, noise)
cmrr = simulate_cmrr(cfg)
psrr = simulate_psrr(cfg)
n = simulate_noise(cfg, noise)
```

Exported symbols are listed in [`src/opamp_model/__init__.py`](src/opamp_model/__init__.py). Configuration dataclasses live in [`src/opamp_model/config.py`](src/opamp_model/config.py).

## Scripts and CLI

| Script | Purpose |
| --- | --- |
| `scripts/install_python.sh` | Create `.venv` and editable install |
| `scripts/run_ac.py` | Open-loop Bode, CMRR curve, impedance plots |
| `scripts/run_stb.py` | Loop gain, GBW, phase margin |
| `scripts/run_noise.py` | Spot and integrated noise |
| `scripts/run_psrr.py` | PSRR vs frequency |
| `scripts/run_slew.py` | Step response and slew rate |
| `scripts/run_thd.py` | THD / harmonics (Python engine) |
| `scripts/run_all_simulations.sh` | Batch all benches per engine |
| `scripts/write_engine_report.py` | Aggregate `REPORT.md` for an engine |
| `scripts/compare_engines.py` | Cross-engine spread on `opamp_metrics.json` |
| `scripts/check_independence.sh` | Guard against legacy repo imports |

Shared CLI flags: `--simulator {python,ngspice,spectre}`, `--output-dir`, `--ideal`, and macromodel parameters (`--a0-db`, `--gbw-hz`, `--cmrr-db`, `--psrr-db`, `--psrr-pole-hz`, `--rin-ohm`, `--rout-ohm`, `--loop-beta`, noise args, …). See [`src/opamp_model/cli_helpers.py`](src/opamp_model/cli_helpers.py).

## Multi-engine workflow

```text
  docs/MODEL.md (equations)
        │
        ├── Python (src/opamp_model/)
        ├── ngspice (testbench/ngspice/*.cir)
        └── Spectre (veriloga/*.va + testbench/spectre/*.scs)
        │
        ▼
  scripts/run_*.py  →  outputs/<engine>/
```

`docs/golden_metrics.yaml` holds optional transistor-level reference targets; it is **not** an engine truth file. No engine is golden — temporary Python curve substitution for some spectre/ngspice paths is documented in [docs/MODEL.md](docs/MODEL.md#engine-parity-current).

### Compare engines

After running benches for each engine, check peer spread against [docs/MODEL.md](docs/MODEL.md) limits (A0 0.1 dB, GBW 5%, phase margin 2°, integrated noise RMS 5%):

```bash
./scripts/run_all_simulations.sh --skip-missing
python scripts/compare_engines.py --output-root outputs
```

Optional reference column from `docs/golden_metrics.yaml` (not used for pass/fail). Exit code is non-zero when spread exceeds tolerance.

## Metrics and outputs

Each engine directory under `outputs/<engine>/` accumulates metrics across benches:

| File | Content |
| --- | --- |
| `opamp_metrics.json` | Merged scalars: `ac`, `stb`, `impedance`, `cmrr_psrr`, `noise`, `large_signal` |
| `AC_REPORT.md`, `STB_REPORT.md`, `NOISE_REPORT.md`, `PSRR_REPORT.md` | Per-bench Markdown + figures |
| `SLEW_REPORT.md`, `THD_REPORT.md` | Transient benches (**python** only) |
| `REPORT.md` | Index of bench reports and SVG gallery |
| `*.csv`, `*.svg` | e.g. `ac_bode`, `stb_bode`, `cmrr`, `impedance`, `psrr`, `noise_spectrum`, `slew`, `thd_*` |
| `logs/` | Simulation logs, rendered netlists, Verilog-A copies |

Default frequency sweep: 1 Hz–100 MHz, 10 points per decade ([docs/bench_spec.md](docs/bench_spec.md)).

## Project layout

```text
opamp-model/
├── LICENSE
├── pyproject.toml
├── docs/
│   ├── README.md            # Documentation index + engine quick reference
│   ├── MODEL.md             # Equations, impedance/noise, engine parity
│   ├── bench_spec.md        # Benches, outputs, netlists
│   ├── metrics_catalog.md   # Metric names and status
│   └── golden_metrics.yaml  # Optional reference targets
├── veriloga/
│   ├── configurable_opamp.va
│   ├── configurable_tia.va
│   └── configurable_gm.va
├── src/opamp_model/         # Python package
├── testbench/
│   ├── ngspice/             # SPICE netlists
│   └── spectre/             # Spectre decks
├── scripts/                 # Install, benches, batch runner
├── tests/                   # pytest suite
└── outputs/                 # Generated reports (gitignored in normal use)
```

## Documentation

| Document | Purpose |
| --- | --- |
| [docs/README.md](docs/README.md) | Documentation index and [engine quick reference](docs/README.md#quick-reference) |
| [docs/MODEL.md](docs/MODEL.md) | Macromodel equations, impedance/noise/large-signal, [engine parity](docs/MODEL.md#engine-parity-current) |
| [docs/bench_spec.md](docs/bench_spec.md) | Bench definitions, per-engine support, outputs, netlists |
| [docs/metrics_catalog.md](docs/metrics_catalog.md) | Metric names, units, status (`reported` / `scaffold` / `planned`) |
| [docs/golden_metrics.yaml](docs/golden_metrics.yaml) | Optional transistor-level reference targets |
| [testbench/ngspice/README.md](testbench/ngspice/README.md) | ngspice netlist notes |
| [testbench/spectre/README.md](testbench/spectre/README.md) | Spectre deck notes |

## Development

```bash
source .venv/bin/activate
pytest
ruff check src tests scripts
ruff format src tests scripts
```

Tests live under `tests/` (AC, STB, noise, CM/PS, transient, engines, independence, reporting).

## Independence

This package is self-contained: it must not import or reference legacy `adc-model/` or `OPAMP_RAK/` paths. CI-style checks:

```bash
./scripts/check_independence.sh
```

Implementation: [`tests/test_independence.py`](tests/test_independence.py), [`src/opamp_model/independence.py`](src/opamp_model/independence.py).

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [LICENSE](LICENSE) for the deed notice.

Copyright (c) 2026 SJTU-YONGFU-RESEARCH-GRP.
