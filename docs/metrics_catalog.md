# Op-amp metrics catalog

Target figures of merit for **opamp-model**, mapped to benches and implementation status.

## Status legend

| Status | Meaning |
| --- | --- |
| **reported** | Written to `opamp_metrics.json` from simulation or extraction |
| **param** | Taken from macromodel parameters (not a separate simulation) |
| **ideal** | Disabled by `--ideal` or zero nonlinearity/noise |
| **planned** | Bench or model path documented here; not wired yet |
| **scaffold** | Netlist runs but curves/metrics still from Python (ngspice noise, Spectre AC/noise) |

## Small-signal / AC / STB

| Metric | Unit | Bench | Status |
| --- | --- | --- | --- |
| Open-loop gain `a0_db` | dB | AC | **reported** (python, ngspice); **scaffold** (spectre AC) |
| Unity-gain BW `gbw_hz` | Hz | AC / STB | **reported** |
| Phase margin `phase_margin_deg` | deg | STB | **reported** |
| Gain margin `gain_margin_db` | dB | STB | **reported** (may be null if no −180° crossing) |
| Gain peaking `peak_db` | dB | AC comp | **reported** (`run_ac_comp.py`) |
| Second pole / zero fit | Hz | AC | **planned** (optional fit from Bode) |

## Impedance

| Metric | Unit | Bench | Status |
| --- | --- | --- | --- |
| `zin_ohm` @ 1 Hz | Ω | AC | **reported** (`impedance_model`) |
| `zout_ohm` @ 1 Hz | Ω | AC | **reported** |
| `rin_ohm`, `cin_f`, `rout_ohm`, `cout_f` | Ω / F | AC | **param** |
| `Zin(f)`, `Zout(f)` curves | Ω | AC | **reported** (`impedance.svg`, macromodel) |

## CMRR / PSRR

| Metric | Unit | Bench | Status |
| --- | --- | --- | --- |
| `cmrr_db` at DC | dB | AC | **reported** (`cmrr_bench` / `cmrr.svg`) |
| `acm_db` at DC | dB | AC | **reported** |
| `CMRR(f)`, `ACM(f)` curves | dB | AC | **reported** (`cmrr.csv`, `cmrr.svg`) |
| `psrr_db` at DC | dB | PSRR | **reported** (`psrr_bench`) |
| `PSRR` vs `f` | dB | PSRR | **reported** (`psrr.csv`, `psrr.svg`) |

PSRR uses the Python macromodel for all engines until supply AC is modeled in Verilog-A / SPICE.

## Noise

| Metric | Unit | Bench | Status |
| --- | --- | --- | --- |
| `en_white_v_per_sqrt_hz` | V/√Hz | Noise | **param** (or **ideal** with `--ideal`) |
| `en_flicker_corner_hz` | Hz | Noise | **param** |
| `en_in_spot_1khz_v_per_sqrt_hz` | V/√Hz | Noise | **reported** (python); **scaffold** (ngspice/spectre) |
| `en_out_spot_1khz_v_per_sqrt_hz` | V/√Hz | Noise | **reported** / **scaffold** |
| `integrated_noise_rms_v` | V | Noise | **reported** / **scaffold** |
| Flicker @ 1 Hz | V/√Hz | Noise | **param** (`en_flicker_at_1hz_v_per_sqrt_hz`) |

## Large-signal / distortion

| Metric | Unit | Bench | Status |
| --- | --- | --- | --- |
| `slew_pos_vps` / `slew_neg_vps` | V/s | Slew | **reported** (measured 10–90 % step) or **param** |
| Output swing | V | — | **param** (`vswing_high_v`, `vswing_low_v`) |
| `thd_db` / `hd2_db` / `hd3_db` | dB | THD | **reported** (python) or **ideal** |

## TIA / Gm (derived)

| Metric | Unit | Bench | Status |
| --- | --- | --- | --- |
| Transimpedance `Zt` | Ω | TIA AC | **reported** (`run_tia_ac.py`, python) |
| `Zt(f)` curve | Ω | TIA AC | **reported** (`tia_zt.csv`, `tia_zt.svg`) |
| TIA bandwidth (−3 dB) | Hz | TIA AC | **reported** |
| `gm` | S | Gm AC | **reported** (python) |
| `rout_ohm`, `cout_f` | Ω / F | Gm AC | **param** |
| Loaded gain (DC) | dB | Gm AC | **reported** (`gm_ac_bode`) |
| Bandwidth (−3 dB) | Hz | Gm AC | **reported** |
| `gm(f)` curve | S | Gm AC | **reported** (`gm_vs_f.svg`) |

## Unified report file (`opamp_metrics.json`)

Each engine directory merges metrics across benches via ``preserve_metrics_sections``:

```json
{
  "engine": "python",
  "config": { "a0_db": 80, "gbw_hz": 10000000, ... },
  "ac": { "a0_db": { "value": 80, "unit": "dB", "status": "reported", "source": "ac_bode" }, ... },
  "stb": { ... },
  "impedance": { "zin_ohm": { ... }, ... },
  "cmrr_psrr": { "cmrr_db": { ... }, "psrr_db": { ... }, ... },
  "noise": { ... },
  "large_signal": { "slew_pos_vps": { ... }, "thd_db": { ... }, ... }
}
```

Markdown reports:

```text
outputs/<engine>/AC_REPORT.md
outputs/<engine>/STB_REPORT.md
outputs/<engine>/NOISE_REPORT.md
outputs/<engine>/PSRR_REPORT.md
outputs/<engine>/SLEW_REPORT.md      # python only
outputs/<engine>/THD_REPORT.md       # python only
outputs/<engine>/TIA_REPORT.md       # TIA AC
outputs/<engine>/REPORT.md           # index + figure gallery
outputs/<engine>/opamp_metrics.json
outputs/<engine>/*.csv, *.svg
outputs/<engine>/logs/
```
