# opamp-model documentation

Index for the **opamp-model** package (voltage op-amp macromodel, multi-engine benches, reporting).

## Documents

| Document | Purpose |
| --- | --- |
| [MODEL.md](MODEL.md) | Macromodel equations, impedance/noise/large-signal, peer-engine rules |
| [bench_spec.md](bench_spec.md) | Bench stimuli, per-engine support, outputs, default sweeps |
| [metrics_catalog.md](metrics_catalog.md) | Metric names, units, implementation status |
| [golden_metrics.yaml](golden_metrics.yaml) | Optional transistor-level reference targets |

## Quick reference

| Engine | AC / STB | Noise | PSRR | Slew / THD |
| --- | --- | --- | --- | --- |
| **python** | Full macromodel | Full macromodel | Full macromodel | Full macromodel |
| **ngspice** | Parsed `wrdata` Bode | Stub netlist; spectrum from Python | Python macromodel | Not run in batch |
| **spectre** | Netlist runs; Bode from Python until PSF parser | Stub netlist; spectrum from Python | Python macromodel | Not run in batch |

Run all benches:

```bash
./scripts/run_all_simulations.sh --skip-missing
```

Results land under `outputs/<engine>/` with per-bench `*_REPORT.md`, `opamp_metrics.json`, CSV/SVG sweeps, and a top-level `REPORT.md`.

## Planned work

Tracked in [metrics_catalog.md](metrics_catalog.md) as **planned**:

- TIA / Gm closed-loop models and `run_tia_*.py` / `run_gm_*.py` benches
- `compare_engines.py` (cross-engine spread vs `golden_metrics.yaml`)
- Spectre PSF export and full ngspice/Spectre noise parsers (remove Python curve substitution)
