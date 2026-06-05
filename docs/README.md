# opamp-model documentation

Index for the **opamp-model** package (voltage op-amp, TIA, and Gm macromodels; multi-engine benches; reporting).

## Documents

| Document | Purpose |
| --- | --- |
| [MODEL.md](MODEL.md) | Macromodel equations, impedance/noise/large-signal, peer-engine rules |
| [bench_spec.md](bench_spec.md) | Bench stimuli, per-engine support, outputs, default sweeps |
| [metrics_catalog.md](metrics_catalog.md) | Metric names, units, implementation status |
| [golden_metrics.yaml](golden_metrics.yaml) | Optional transistor-level reference targets |
| [Flicker_Noise_Analysis_on_Chopper_Amplifier.pdf](Flicker_Noise_Analysis_on_Chopper_Amplifier.pdf) | Flicker noise modeling reference (device-level; NEWCAS 2021) |

## Quick reference

| Engine | AC / STB | Noise | PSRR | Slew / THD |
| --- | --- | --- | --- | --- |
| **python** | Full macromodel | Full macromodel | Full macromodel | Full macromodel |
| **ngspice** | Parsed `wrdata` Bode | Open-loop `.noise` + AC; flicker from config or VA | Python macromodel | Not run in batch |
| **spectre** | PSF Bode when available | PSF `.noise` + VA `flicker_noise` | Python macromodel | Not run in batch |

Run all benches:

```bash
./scripts/run_all_simulations.sh --skip-missing
```

Results land under `outputs/<engine>/` with per-bench `*_REPORT.md`, `opamp_metrics.json`, CSV/SVG sweeps, and a top-level `REPORT.md`.

## Flicker noise (op-amp macromodel)

Open-loop flicker follows [Flicker_Noise_Analysis_on_Chopper_Amplifier.pdf](Flicker_Noise_Analysis_on_Chopper_Amplifier.pdf) device equations (Eq. 1, Coram `KF`/`AF`, Verilog-A `flicker_noise`). Chopper/system-level noise is **out of scope** for this package.

| Layer | Scope |
| --- | --- |
| **Python** | `en_flicker_1hz`, `en_flicker_ef`, `kf`/`af`/`bias_current_a`, computed `flicker_corner_hz` |
| **Verilog-A** | `white_noise` + `flicker_noise` on `configurable_opamp.va` |
| **Benches** | `run_noise.py` (open-loop only) |

See [MODEL.md — Noise](MODEL.md#noise).

### Legacy parameter note

The pre-paper pair `en_flicker_at_1hz_v_per_sqrt_hz` + `en_flicker_corner_hz` (scale-factor form) remains supported as aliases during migration. New benches and `golden_metrics.yaml` prefer **`en_flicker_1hz`** (true density at 1 Hz) and **`en_flicker_ef`** (exponent `EF`).

## Planned work

Tracked in [metrics_catalog.md](metrics_catalog.md) as **planned**:

- TIA / Gm closed-loop models and `run_tia_*.py` / `run_gm_*.py` benches
- `compare_engines.py` (cross-engine spread vs `golden_metrics.yaml`)
- Full ngspice controlled-source noise transfer (remove post-merge flicker where VA supplies it)
