# opamp-model

Configurable **voltage op-amp**, **transimpedance (TIA)**, and **transconductance (Gm)** behavioral models with Python, Verilog-A, and SPICE testbenches.

**Status:** Phases 1–4 — AC/STB, noise, PSRR/CMRR, slew, THD across **peer engines** (Python, ngspice, Spectre each run independently; same macromodel spec in `MODEL.md`). See [../PLAN.md](../PLAN.md) for Phases 5–6.

## Requirements

- Python 3.10+
- NumPy, Matplotlib, PyYAML (installed automatically)
- Optional (later phases): Cadence Spectre, ngspice

## Installation

```bash
cd opamp-model
./scripts/install_python.sh
source .venv/bin/activate
```

## Quick checks

```bash
pytest
./scripts/check_independence.sh
python scripts/run_ac.py --output-dir outputs/python
python scripts/run_stb.py --output-dir outputs/python
python scripts/run_noise.py --output-dir outputs/python
python scripts/run_psrr.py --output-dir outputs/python
python scripts/run_slew.py --output-dir outputs/python
python scripts/run_thd.py --output-dir outputs/python
# Per-engine artifacts under outputs/<engine>/:
#   *_REPORT.md, REPORT.md — Markdown with embedded SVG figures
#   opamp_metrics.json — scalar metrics across benches
./scripts/run_all_simulations.sh --skip-missing
```

## Project layout

```text
opamp-model/
├── reference/          # bench_spec.md, golden_metrics.yaml
├── veriloga/           # configurable_opamp.va, tia, gm (shells in Phase 0)
├── src/opamp_model/    # Python package
├── testbench/          # Spectre / ngspice netlists (Phase 1+)
├── scripts/
└── tests/
```

## Documentation

| Document | Purpose |
| --- | --- |
| [MODEL.md](MODEL.md) | Equations and signal flow (stub until Phase 1) |
| [reference/bench_spec.md](reference/bench_spec.md) | Bench and metric definitions |
| [../PLAN.md](../PLAN.md) | Full project plan and phased delivery |

## Independence

This package is self-contained: CI scans the tree for forbidden legacy imports and paths (see `tests/test_independence.py` and `scripts/check_independence.sh`).

## License

CC BY 4.0 — see [LICENSE](LICENSE).
