# Partial, independent replication of El-Sayed et al. (2026) (not an exact reproduction)

"Prediction of channel propagation impairments in LEO satellite networks using
machine learning", EURASIP J. Wireless Commun. Netw. 2026:102,
doi:10.1186/s13638-026-02667-w.

> **Note (2026-10-27):** this document covers the earlier **stratified**
> hourly protocol. The current main result is the strictly causal evaluation
> in [docs/CAUSAL_HOURLY_MODEL_EXPERIMENT.md](docs/CAUSAL_HOURLY_MODEL_EXPERIMENT.md);
> overview: [README.md](README.md).

> **What is on GitHub.** Many paths below point to local outputs that are
> not in the repository: `results/paper/` (metrics, tables, predictions and
> figures of this protocol), `models/` (checkpoints and run logs), `data/`
> and `backups/` (local safety copies). They are git-ignored because they are
> large or local; the scripts regenerate the outputs. A curated snapshot of
> this protocol's key tables and figures is in
> [docs/assets/](docs/assets/) (for example `tables.md`,
> `experiments_summary.json`, `audit_report.json`). The `backups/` copies
> are not published and cannot be regenerated.

Every number in this report is copied from files written by the scripts
(`results/paper/metrics_test.json`, `results/paper/tables/*.csv`,
`results/paper/table8_*.csv`); none were typed by hand from memory. The
paper's numbers appear only as references. All result tables are collected
in `results/paper/tables.md`.

**Status (2026-10-26): complete, independently audited, with repeated
seeds; documentation and reproducibility tooling finalised (the dated log
was in `STATUS.md`, removed in commit `8c4f2a1`; see
`git show 8f0b551:STATUS.md`).** Seven variants trained to early stopping (seed 42), plus seeds 1
and 2 for `multitask`, `multitask_fig1heads`, `epl_only` and `no_itu` (15
runs in all, every one complete). Evaluation, unseen-city study (three
models), Figures 1-7 plus supporting figures and Tables 7-11 are generated
from these runs; `evaluation/audit_results.py` re-checks them (127/127
PASS). Earlier result states are kept in `backups/2026-10-24/` and
`backups/2026-10-25/`. Machine-readable run summary:
`results/paper/experiments_summary.json`.

---

## 1. Summary

* We rebuilt the full pipeline (ERA5 weather, ITU-R labels, dataset, 3-layer
  multi-task LSTM with a Gaussian EPL head, evaluation, unseen-city
  inference) as described in the paper and documented every choice the
  paper leaves open.
* On a strict chronological test period (Apr-Dec 2024, six cities) the model
  reaches **EPL RMSE 0.936 dB, MAE 0.351 dB**. A persistence forecast
  reaches RMSE 1.042 dB, MAE 0.352 dB. The paper reports RMSE 0.026 dB.
* The paper's Fig. 2 (not its text) describes **1-minute sampling, a 60-min
  window and a 5-step horizon**. On hourly ERA5 data interpolated to
  1-minute steps, trivial forecasts already reach the paper's error level
  (persistence RMSE 0.033 dB on our test anchors). This is consistent with
  the paper's 0.026 dB being a 5-minute-ahead result on interpolated data,
  not a 5-hour forecast. We test this reading as a separate experiment.
  Trained on that reading, the LSTM (RMSE 0.438 dB) is far worse than
  persistence (0.033 dB) because it fails on rare heavy-rain windows; linear
  extrapolation is exact on 92.5 % of the interpolated windows by
  construction (Sect. 5.4).
* For the ten unseen cities, the paper's input-synthesis method gives an
  error of period means of 0.189 dB (paper: 0.213 dB), but its hourly MAE
  (0.622 dB) is worse than simply repeating the city's current EPL
  (0.357 dB).
* Ablations, three seeds each (Sect. 5.5): removing the ITU-R-derived
  features changes nothing (RMSE 0.940 +- 0.006 vs 0.940 +- 0.004 dB; the
  paired difference changes sign between seeds; paper 0.028 vs 0.026).
  Multi-task is slightly better than EPL-only in all three seeds (mean
  -0.006 +- 0.005 dB), the opposite direction to the paper (0.026 vs
  0.015) but too small to be called a clear effect with n = 3.
  Weather-only forecasts the weather better than the multi-task model
  (standardised RMSE 0.480, one seed, vs 0.525 +- 0.007 over three seeds;
  paper 0.892 vs 0.761, the opposite direction).
* **Pressure:** with mean-sea-level instead of surface pressure
  (`multitask_msl`), the two high-altitude unseen cities are no longer
  outliers when run on their own weather: Davos error of means 2.25 -> 0.05
  dB, Madrid 0.99 -> 0.04 dB. Six-city test error is unchanged (0.935 dB).
  This strongly supports out-of-range surface pressure as the cause of those
  two failures (one seed per model); it does not tell us which pressure the
  authors actually used.
* Best six-city model (on validation and on test): the Fig. 1 head variant
  (`multitask_fig1heads`), RMSE 0.916 +- 0.006 dB over three seeds (seed 42:
  0.910 dB) vs 0.940 +- 0.004 dB for the main model; it is better in every
  seed (paired difference -0.024 +- 0.003 dB). It narrows but does not
  close the hour-1 gap to persistence (0.643 vs 0.596 dB).

---

## 2. How to run (Windows, Python 3.12)

Environment: `py -3.12 -m venv .venv` then
`.venv\Scripts\python -m pip install -r requirements.txt` (exact versions
used for the results; a fresh install was not tested on another machine).

```powershell
.\run_pipeline.ps1                # labels -> datasets -> training -> evaluation -> figures/tables -> audit -> tests
.\run_pipeline.ps1 -Download      # also download any missing weather files first
.\run_pipeline.ps1 -EvaluateOnly  # existing checkpoints only: evaluate, figures, tables, audit, tests
```

What each kind of file needs:

| Files | Source | Needed for |
|---|---|---|
| `data/processed/weather/open_meteo_v2/`, `open_meteo/`, `open_meteo_msl/` | downloaded from the Open-Meteo archive API (no account) | labels, datasets |
| `data/raw/era5/` | ERA5 cloud water from the Copernicus CDS (free account, `~/.cdsapirc`) | only re-fitting the LWP proxy |
| `physics/lwp_proxy_calibration.json` | committed (fitted coefficients) | labels |
| `data/processed/physics_paper/`, `paper/`, `paper_msl/`, `paper_minute/` | generated (deterministic) | training, evaluation |
| `models/paper/**` | generated by training (CPU, about 1-2.5 h per run) | evaluation |
| `results/paper/**` | generated by evaluation | reports; snapshot in `docs/assets/` |

Individual steps (same order as the script):

```powershell
.venv\Scripts\python physics\calibrate_lwp_proxy.py     # optional, needs data\raw\era5
.venv\Scripts\python physics\build_attenuation.py
.venv\Scripts\python physics\build_attenuation.py --unseen
.venv\Scripts\python data_pipeline\build_paper_dataset.py
.venv\Scripts\python data_pipeline\build_paper_dataset.py --msl
.venv\Scripts\python data_pipeline\build_minute_dataset.py
.venv\Scripts\python evaluation\plot_dataset.py
.venv\Scripts\python training\run_queue.py          # trains everything in training\queue.txt
.venv\Scripts\python evaluation\evaluate_paper_model.py
.venv\Scripts\python evaluation\evaluate_unseen_cities.py
.venv\Scripts\python evaluation\evaluate_unseen_cities.py multitask_msl
.venv\Scripts\python evaluation\evaluate_unseen_cities.py multitask_fig1heads
.venv\Scripts\python training\run_queue.py training\queue_seeds.txt   # repeated seeds 1 and 2
.venv\Scripts\python evaluation\summarize_runs.py    # seed_runs.csv, seed_summary.csv, experiments_summary.json
.venv\Scripts\python evaluation\calibration_analysis.py   # calibration_conditional.csv
.venv\Scripts\python evaluation\minute_diagnostics.py     # minute_diagnostics.json
.venv\Scripts\python evaluation\make_diagrams.py
.venv\Scripts\python evaluation\make_figures.py
.venv\Scripts\python evaluation\make_tables.py
.venv\Scripts\python evaluation\audit_results.py     # independent re-check, audit_report.json
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python evaluation\export_docs_assets.py     # refresh docs\assets snapshot
```

Completed vs optional experiments: all runs in `training/queue.txt` (7) and
`training/queue_seeds.txt` (8) are complete. Optional, not run: further seeds,
repeated seeds for `weather_only` / `multitask_msl` / `minute_multitask`, and
unseen-city evaluation of the seed-1/2 models.

A single run: `.venv\Scripts\python training\train_paper_model.py <variant> [seed]`.
Seed 42 (default) writes `models/paper/<variant>/`; any other seed writes
`models/paper/seeds/<variant>_seed<N>/`, so originals are never overwritten.
Training is deterministic (`enable_op_determinism`): re-running a seed
reproduces its losses to 1e-7 on this machine (checked for `multitask`,
2 epochs). Other CPUs or TensorFlow versions may differ slightly.

Downloads (only needed once; existing files are skipped, and a file that
must be rebuilt is first kept as `*.before_rebuild.csv`):

```powershell
.venv\Scripts\python data_pipeline\download_openmeteo.py <City> <Year> data\processed\weather\open_meteo_v2
.venv\Scripts\python data_pipeline\download_openmeteo.py <UnseenCity> <2023|2024>
.venv\Scripts\python data_pipeline\download_pressure_msl.py
```

Training runs on the CPU (about 3-5 min per epoch, 1-2.5 h per model). Each
epoch is saved; re-running `training\run_queue.py` resumes an interrupted
model and skips finished ones. On this laptop, closing the lid or a
critical battery puts Windows into standby, which pauses training.

---

## 3. Paper vs implementation

| Item | Paper | Implemented | Status |
|---|---|---|---|
| Frequency | 39 GHz | 39 GHz | same |
| Training cities, coordinates | 6, Table 3 | same | same |
| Period, source | 5 years, ERA5 via Open-Meteo, hourly | 2020-2024, same source | same (years not stated) |
| Samples | 10,000 per city, stratified by month and hour | proportional allocation over 288 (month, hour) strata, seed 42 | interpretation |
| Gas | ITU-R P.676 | P.676-12: Annex 1 surface gamma_o, gamma_w x Annex 2 equivalent heights / sin(el) | P.676-13 cited; not in itur |
| Cloud | P.840, LWP proxy from RH and precipitation | P.840-8, K_l at 0 degC, proxy fitted to ERA5 tclw | proxy formula not given |
| Rain | P.838 + P.618, rain height P.839 | P.838-3, P.618-13 steps 1-9 with hourly R, P.839-4 | P.618-14 cited |
| EPL | gas + cloud + rain | same | same |
| Inputs | 14 (Table 2) | same 14, same order | Table 4 lists 17 |
| Pressure input | Table 2: mean sea level; Sect. 3.1: surface | surface (main); MSL tested separately | paper inconsistent |
| Sequence / horizon | text: 60 min, 5 h; Fig. 2: 1-min sampling, 5 steps | 60 hourly steps, 5 h (main); 1-min reading tested separately | paper inconsistent |
| LSTM | 3 x 256, dropout 0.2 | same | same |
| Heads | Fig. 1: Dense 256 -> 256 -> 25 / 10 | Dense 256 -> 25 / 10 (main); Fig. 1 heads tested separately | deviation, tested |
| Loss | 1.0 MSE(weather) + 1.0 NLL(EPL), Eq. 10-11 | same | same |
| Training | Adam 1e-4, batch 128, <= 50 epochs, patience 6, seed 42 | same | same |
| Split | chronological 70/15/15 per location, training statistics | same + 65-h purge gap | stricter |
| Metrics | RMSE per city/hour, MAE, PICP 95 %, NLL, sharpness | same + baselines, calibration | extended |
| Unseen cities | 10 cities, K = 3, softmax(-d/tau) | same, tau = 500 km (+250/1000); Fig. 2 median variant | tau not given |

### 3.1 Confirmed bugs fixed (earlier prototype `physics/build_attenuation.py`, original in `backups/2026-10-23/`)

1. **P.676 ignored measured T and P.** `itur` in `mode="exact"` integrates a
   standard atmosphere; only humidity varied. Now Annex 1 gammas at the
   measured surface state x Annex 2 equivalent heights (matches `itur`'s
   approximate slant path exactly; matches the paper's Eq. 8 form).
2. **P.840 K_l 5.6x too small.** `itur` expects degC; the prototype passed
   Kelvin (K_l 0.219 instead of 1.233 (dB/km)/(g/m^3) at 39 GHz, checked
   by hand from P.840 Eq. 2).
3. **Rain path was not P.618.** gamma_R x h_R / sin(el) replaced by P.618
   steps 1-9 (equal to `itur.rain_attenuation(R001=R, p=0.01)` to 0.0 dB).
4. **P.676 pressure.** Annex 1 needs dry-air pressure; total pressure was
   passed. Now p_dry = P - e.
5. **Snow in the rain label.** Total precipitation (up to 13 % snow in
   Warsaw) was used as rain rate. Now ERA5 liquid rain (`rain_mm`,
   downloaded into `open_meteo_v2/`, identical to the original files on all
   shared columns).

Other fixes in this project: the downloader no longer silently replaces a
file (keeps `*.before_rebuild.csv`); training saves every epoch and resumes.

### 3.2 Choices the paper leaves open (our documented assumptions)

1. Elevation 30 deg ("fixed, representative of LEO"); constant, so it
   carries no information after scaling.
2. Circular polarisation (tau = 45 deg in P.838).
3. mu features = the instantaneous component (Eq. 8); sigma features =
   trailing 24-h standard deviation of that component (no formula given;
   past hours only).
4. LWP proxy `a * clip((RH - RH_c)/(100 - RH_c), 0, 1) + b * precip`,
   fitted (non-negative least squares) to ERA5 total column cloud liquid
   water for Athens 2020-2022 and Bergen 2020-01..2021-02: RH_c = 65 %,
   a = 0.101 kg/m^2, b = 0.144 kg/m^2 per mm, fit R^2 = 0.41. Held-out
   Athens 2023-2024: correlation 0.38, R^2 = -0.01. Surface RH and
   precipitation explain little of the true LWP; this weakness belongs to
   the method the paper describes.
5. Hourly ERA5 rain accumulation used as the hourly-mean rain rate.
6. P.618's r_0.01 and nu_0.01 factors were derived for high rain rates; at
   low hourly rates they exceed 1 and lengthen the path (0.1 mm/h gives
   about 0.3-0.4 dB). Kept, because the paper prescribes P.618.
7. Each sampled hour is the anchor of a window taken from the continuous
   series (60 h of inputs, 5 h of targets).
8. One feature scaler over all six cities' training windows; EPL targets in
   dB; weather targets standardised.
9. Unseen cities: standard city-centre coordinates; evaluation on every
   hour 2024-04-09..2024-12-31; true EPL = our ITU-R label from the city's
   own ERA5 weather; lat/lon inputs = the target city's coordinates.
10. ITU-R versions in `itur` 0.4.0: P.676-12, P.618-13, P.840-8, P.838-3,
    P.839-4, P.1511 (station height).

### 3.3 Inconsistencies found in the paper

| Where | Inconsistency | Our handling |
|---|---|---|
| Text vs Fig. 2 | "60 min" history and "5 h" horizon vs "1-minute uniform sampling", "5 steps", "stride 5 minutes" | main: 60 h / 5 h; separate minute experiment |
| Fig. 1 vs Table 2 | "past hour" with 60 time steps; data are hourly | as above |
| Table 2 vs Table 4 | 14 features vs 14 + 3 temporal features | 14 |
| Table 2 vs Sect. 3.1 | mean-sea-level vs surface pressure | surface; MSL experiment |
| Fig. 1 vs text | heads have a hidden Dense(256) layer only in the figure | separate experiment |
| Eq. 12 vs Fig. 2 | weighted sum vs "K=3 nearest cities median" | both evaluated |
| Table 8 | listed absolute errors average 0.211, stated 0.213 | noted |
| Table 7 vs Sect. 4.6 | RMSE almost flat from hour 1 to 5; "accuracy" per weather class is undefined | we report RMSE/MAE by condition |
| Abstract vs Sect. 5 | 0.213 dB called "average error" and "average RMSE" | we report both mean-error and hourly MAE/RMSE |

---

## 4. Data and labels

* Weather: 30 city-years (6 cities x 2020-2024), 43,848 hours per city, no
  missing values, complete hourly timestamps. Unseen cities: 2023-2024
  (17,544 hours each). MSL pressure for all 16 cities.
* Mean EPL per training city (`results/paper/physics_summary.csv`): Athens
  0.92, Bergen 1.54, Budapest 1.11, Chilbolton 1.21, Vigo 1.67, Warsaw
  1.10 dB; gas contributes about 0.75 dB everywhere, rain dominates the
  variability (maximum 16.7-25.6 dB).
* Dataset (`data/processed/paper/manifest.json`): 10,000 anchors per city,
  33-36 per (month, hour) stratum; windows 42,000 / 8,898 / 8,916
  (train / val / test); test period 2024-03-24..2024-12-31 depending on city.

---

## 5. Results: six cities, chronological test set

### 5.1 Main model (`models/paper/multitask`)

20 epochs, best epoch 14 (val loss -0.2187), 1,337,123 parameters.

| | RMSE (dB) | MAE (dB) |
|---|---|---|
| **Multi-task LSTM** | **0.936** | **0.351** |
| Persistence (EPL at t) | 1.042 | 0.352 |
| Climatology (city x month x hour, training period) | 1.235 | 0.590 |

RMSE by forecast hour, hours 1-5: LSTM 0.702, 0.905, 0.985, 1.007, 1.041;
persistence 0.596, 0.946, 1.097, 1.188, 1.251. **The LSTM loses to
persistence at hour 1** and wins from hour 2 on. Per-city table:
`results/paper/tables.md` (Table 7).

Uncertainty: PICP 96.6 % (paper 99.23 %), NLL (Eq. 10) -0.632 (paper
-2.393), sharpness 0.557 dB (paper 0.035 dB). The overall 95 % coverage
is **not** evidence of good calibration (`evaluation/calibration_analysis.py`,
`results/paper/calibration_conditional.csv`, `figures/calibration.png`):

* Nominal 50 / 80 / 90 / 99 % intervals cover 77.8 / 92.8 / 95.3 / 97.9 %
  of outcomes: too wide in the centre, too narrow in the tails (errors are
  heavier-tailed than Gaussian).
* By weather at the target hour: dry hours (79 % of hours) 99.4 % coverage
  with 1.43 dB mean width; hours with precipitation 86.4 % (4.95 dB); >= 2.5
  mm/h only 37.0 % (10.1 dB).
* By true EPL: 0-1 dB 99.3 %, 2-5 dB 86.1 %, 5-10 dB 66.5 %, >= 10 dB 20.4 %
  (mean error -10.2 dB there). These groups use the outcome and are
  diagnostics only.

The intervals are wider when rain is likely, but not wide enough; the
Gaussian lower bound can also go below 0 dB. The Fig. 1-head model behaves
the same (>= 2.5 mm/h coverage 45.2 %).

By weather at the target hour (LSTM / persistence RMSE): clear 0.139 / 0.193
(27 % of hours), cloudy 0.388 / 0.532 (52 %), light precipitation 1.245 /
1.709 (21 %), >= 2.5 mm/h 7.96 / 6.60 (0.7 %). (The class is defined by
the true weather at each target hour; it is used only to group errors, never
as a model input.) The model never predicts more than 5.8 dB (Fig. 1-head
variant: 8.3 dB), while 2.2 % of true test values exceed 5 dB and 32 test
windows exceed 14 dB; only 66 of 42,000 training windows exceed 14 dB, so
heavy rain is badly under-predicted.

Weather head (RMSE, LSTM / persistence): temperature 1.87 / 2.40 degC,
humidity 6.7 / 9.5 %, pressure 2.14 / 1.21 hPa, precipitation 0.34 /
0.39 mm, cloud cover 25.0 / 25.4 %. The LSTM is worse than persistence at
hour 1 for every variable and never better for pressure
(`figures/weather_head_rmse.png`).

All variants use the same anchors, chronological split, scalers, seed (42),
optimiser and early-stopping rule; they differ only in what the caption
says. One run each, so differences of a few thousandths of a dB are within
run-to-run noise (not measured; see Limitations).

| Variant | Epochs (best) | Train time (h) | EPL RMSE | EPL MAE | RMSE hours 1-5 | PICP 95 % | Weather RMSE (std.) |
|---|---|---|---|---|---|---|---|
| `multitask` (main) | 20 (14) | 7.60* | 0.936 | 0.351 | 0.702 0.905 0.985 1.007 1.041 | 96.6 | 0.522 |
| `epl_only` | 19 (13) | 1.23 | 0.942 | 0.358 | 0.706 0.919 0.987 1.018 1.041 | 96.9 | - |
| `weather_only` | 18 (12) | 0.98 | - | - | - | - | 0.480 |
| `no_itu` | 24 (18) | 3.39 | 0.933 | 0.352 | 0.698 0.902 0.983 1.001 1.038 | 96.5 | 0.512 |
| `multitask_msl` | 22 (16) | 1.31 | 0.935 | 0.344 | 0.698 0.902 0.983 1.009 1.041 | 96.1 | 0.520 |
| `multitask_fig1heads` | 22 (16) | 1.34 | **0.910** | **0.342** | 0.636 0.873 0.961 0.998 1.027 | 96.6 | 0.508 |
| Persistence | - | - | 1.042 | 0.352 | 0.596 0.946 1.097 1.188 1.251 | - | - |

\* includes laptop standby pauses. All times are wall-clock on a shared
laptop CPU and are not comparable across runs.

Every saved prediction file was re-checked against a fresh prediction from
the variant's `best.keras` (1,000 random test windows each): maximum
difference 1.2e-6.

### 5.2 Table 9: single-task vs multi-task

| Configuration | EPL RMSE (dB) | Weather RMSE (standardised) | Paper EPL RMSE | Paper weather RMSE |
|---|---|---|---|---|
| EPL only | 0.942 | - | 0.015 | - |
| Weather only | - | **0.480** | - | 0.892 |
| Multi-task | **0.936** | 0.522 | 0.026 | 0.761 |

Weather RMSE in physical units (temperature degC / RH % / pressure hPa /
precipitation mm / cloud %): weather-only 1.31 / 5.26 / 1.31 / 0.330 / 22.9;
multi-task 1.86 / 6.66 / 2.14 / 0.338 / 25.0.

* EPL: multi-task is 0.006 dB better than EPL-only (seed 42). Over three
  seeds: 0.940 +- 0.004 vs 0.946 +- 0.007 dB, multi-task better in every
  seed by 0.001-0.010 dB (Sect. 5.5). The paper reports EPL-only as
  *better* (0.015 vs 0.026). Our difference points the other way but is
  small and, with three seeds, not established.
* Weather: the single-task model is clearly better (by 30 % for
  temperature, 39 % for pressure). The paper reports the multi-task model
  as better (0.761 vs 0.892). **We do not reproduce the paper's claim that
  joint training helps the weather forecast**; in our runs it costs
  weather accuracy and buys almost no EPL accuracy. `weather_only` was run
  with one seed only; its gap to the multi-task model (0.480 vs 0.525 +-
  0.007 over three multi-task seeds) is about six multi-task standard
  deviations, so seed noise is an unlikely explanation, but its own seed
  spread was not measured.
* The paper does not say how its weather RMSE is scaled; ours is the RMSE
  of standardised targets (training mean/std), so only the direction of
  the comparison can be checked.

### 5.3 Table 10: without ITU-R-derived features

| Model configuration | RMSE (dB) | MAE (dB) | PICP 95 % | Paper RMSE | Paper MAE |
|---|---|---|---|---|---|
| Full feature set (14) | 0.936 | 0.351 | 96.6 | 0.026 | 0.009 |
| Without the six ITU-R gas/cloud/rain mu and sigma features (8) | 0.933 | 0.352 | 96.5 | 0.028 | 0.012 |

No measurable effect, confirmed over three seeds: 0.940 +- 0.006 (without)
vs 0.940 +- 0.004 dB (with); paired differences +0.005, -0.002, -0.003 dB
change sign between seeds (Sect. 5.5). This shows only that *this* model
did not benefit from them; it does not show the features carry no
information. A plausible explanation (hypothesis, not tested): gas and
cloud features are fixed functions of inputs the model already sees (T,
RH, P, precipitation); the rain feature uses ERA5 liquid rain, which is
*not* an input (the input is total precipitation, including snow), and the
sigma features are 24-h summaries, so some information is new but appears
to be largely redundant for the network. Testing this would need, e.g.,
an ablation that removes only the rain features, or a smaller model where
engineered features matter more. The paper's gain (0.026 vs 0.028) is not
reproduced; our seed-to-seed spread (about 0.005 dB) is larger than the
paper's reported difference of 0.002 dB.

### 5.4 Extra experiments

**`multitask_msl`** (Table 2 reading: mean-sea-level pressure as input and
weather target; dataset `data/processed/paper_msl/`, identical anchors and
EPL targets, verified). Six-city test: RMSE 0.935 dB, essentially equal to
the main model. Its value is for unseen cities (Sect. 6).

**`minute_multitask`** (Fig. 2 reading: hourly ERA5 linearly interpolated to
1 minute, 60-min input, 5 one-minute steps; `data/processed/paper_minute/`,
8,916 test windows).

| | RMSE (dB) | MAE (dB) | RMSE steps 1-5 |
|---|---|---|---|
| LSTM | 0.438 | 0.035 | 0.443 0.538 0.376 0.389 0.423 |
| Persistence | 0.033 | 0.0093 | 0.010 0.020 0.030 0.040 0.050 |
| Linear extrapolation | 0.013 | 0.0010 | 0.004 0.008 0.012 0.016 0.020 |
| Paper Table 7 | 0.026 | 0.009 | 0.026 (flat) |

Why linear extrapolation wins (measured,
`evaluation/minute_diagnostics.py`, results in `results/paper/minute_diagnostics.json`):

1. The data are piecewise linear within each hour. Anchors sit at minute
   offsets 0, 5, ..., 55, so for offsets 5-55 the last input minute and all
   five target minutes lie on the same straight segment. Linear
   extrapolation is then exact (max error 4e-6 dB); it errs only at offset
   0 (max 0.73 dB), where the previous minute belongs to the previous hour.
   92.5 % of test windows are predicted to within 1e-4 dB. This is a
   property of the interpolation, not forecasting skill.
2. The current EPL is exactly the sum of the three instantaneous component
   inputs (checked, max difference 2e-6 dB, float rounding), so even a "repeat the input" rule
   is available to the model.
3. The LSTM fails on rare extremes: only 14 of 42,000 training windows
   exceed 14 dB. On those *training* windows it predicts 5.96 dB on average
   for a true 15.94 dB, with sigma 9.0 dB (RMSE 10.2 dB); on 4-8 dB training
   windows it is accurate (RMSE 0.28 dB). Its highest test prediction is
   8.1 dB for targets up to 20.9 dB. The Gaussian NLL lets it widen the
   interval instead of following the level.

The test RMSE is dominated by the 68 windows (0.8 %) whose error exceeds
1 dB; their worst cases have true EPL 14-21 dB. Diagnostic only (not a
reported result): on the other 99.2 % of windows its RMSE is 0.026 dB. By
condition: clear 0.004, cloudy 0.004, light precipitation 0.129,
>= 2.5 mm/h 5.51 dB. Conclusion: on interpolated minute data the task is
almost trivial (linear extrapolation beats the paper's reported number), so
under this reading the paper's 0.026 dB would not show forecasting skill
either; and our LSTM does not even match persistence there.

**`multitask_fig1heads`** (hidden Dense(256, ReLU) in each head, as drawn in
Fig. 1; 1,468,707 parameters). Best six-city result: RMSE 0.910 dB, MAE
0.342 dB, NLL -0.663, sharpness 0.540 dB; better than the main model at
every horizon (largest gain at hour 1: 0.636 vs 0.702).

Comparability with the main model (checked): same dataset, split, scalers,
features, loss, optimiser, batch size, early-stopping rule and seed; the
training code differs only by the extra head layer (the code that trained
`multitask` and `no_itu` predates the variant options but builds an
identical graph for them; a 2-epoch retrain of `multitask` with today's code
reproduces the original losses to 1e-7). Neither run was interrupted or
resumed. The ranking is the same on the validation set (EPL RMSE 0.896 vs
0.936 dB), so it is not a test-set selection effect, and the gain holds in
all three seeds (Sect. 5.5).

Unseen cities: it still uses surface pressure, so with own inputs Davos and
Madrid fail as before (mean error of means 0.446 dB;
`results/paper/table8_unseen_summary_multitask_fig1heads.csv`).
Synthesized mode: 0.185 dB error of means, 0.618 dB hourly MAE.

### 5.5 Repeated seeds (seeds 42, 1, 2)

The four variants that carry the paper's main claims were re-trained with
seeds 1 and 2 (`training/queue_seeds.txt`), with the same data, split,
scalers, settings and early-stopping rule; only the seed changes
(initial weights, dropout masks, batch order). All eight extra runs
completed (none resumed; `multitask_fig1heads_seed1` was paused for about
52 min by a battery standby, which only adds wall-clock time).
`weather_only`, `multitask_msl` and `minute_multitask` were not repeated.
Source: `results/paper/seed_runs.csv`, `seed_summary.csv`,
`seed_comparisons.csv`; figure `seed_variation`.

| Variant | Seeds | Test EPL RMSE, mean +- std (dB) | Min-max | MAE mean | Hour-1 RMSE mean | PICP 95 % mean |
|---|---|---|---|---|---|---|
| `multitask` | 42, 1, 2 | 0.940 +- 0.004 | 0.936-0.943 | 0.349 | 0.708 | 96.4 |
| `epl_only` | 42, 1, 2 | 0.946 +- 0.007 | 0.942-0.954 | 0.356 | 0.715 | 96.6 |
| `no_itu` | 42, 1, 2 | 0.940 +- 0.006 | 0.933-0.945 | 0.349 | 0.712 | 96.3 |
| `multitask_fig1heads` | 42, 1, 2 | **0.916 +- 0.006** | 0.910-0.920 | 0.343 | 0.643 | 96.3 |
| Persistence (reference) | - | 1.042 | - | 0.352 | 0.596 | - |

Paired differences (same seed; negative = first variant better):

| Comparison | Per seed (1, 2, 42) | Mean +- std | Same sign in all seeds |
|---|---|---|---|
| fig1heads - multitask, RMSE | -0.021, -0.025, -0.026 | -0.024 +- 0.003 | yes |
| fig1heads - multitask, hour-1 RMSE | -0.062, -0.064, -0.067 | -0.064 +- 0.002 | yes |
| multitask - epl_only, RMSE | -0.001, -0.010, -0.006 | -0.006 +- 0.005 | yes |
| no_itu - multitask, RMSE | +0.005, -0.002, -0.003 | -0.000 +- 0.004 | no |

Reading (measured; n = 3 is small, so no formal significance test is
claimed):

* Run-to-run spread is 0.004-0.007 dB, so single-run differences of a few
  thousandths of a dB (Tables 9-10 at seed 42) are within noise.
* The Fig. 1 hidden head layer is a consistent improvement (about 8x the
  paired standard deviation, every seed).
* ITU-R features: no effect. Multi-task vs EPL-only: a small, consistent
  edge for multi-task, not large enough to call established.
* Seed-42 results lie inside the three-seed range for every variant, but
  seed 42 is the best of the three for `multitask`, `no_itu` and
  `multitask_fig1heads`, so the single-run numbers quoted elsewhere in this
  report are slightly optimistic (by 0.004-0.007 dB versus the means).

---

## 6. Results: ten unseen cities (separate from the six-city test)

Forecasts for every hour 2024-04-09..2024-12-31 (6,385 per city), main
model. Mean over the ten cities:

| Input mode | Error of period means | Hourly MAE | Hourly RMSE | PICP 95 % |
|---|---|---|---|---|
| Synthesized, weighted (Eq. 12-13, tau 500 km) | 0.189 | 0.622 | 1.214 | 85.1 % |
| Synthesized, median of 3 (Fig. 2) | 0.151 | 0.536 | 1.160 | 84.1 % |
| Own inputs (extra, not the paper's protocol) | 0.373 | 0.615 | 1.122 | 94.7 % |
| Persistence of own EPL (needs local data) | 0.000 | 0.357 | 1.052 | - |
| Persistence of weighted neighbour EPL | 0.188 | 0.656 | 1.346 | - |

Paper: average absolute error 0.213 dB. Per-city values: Table 8 in
`results/paper/tables.md`. tau sensitivity (hourly MAE): 250 km 0.624,
500 km 0.622, 1000 km 0.626.

Findings:

* The paper's measure (|mean prediction - mean truth| over a period) is
  close to the paper's value, but it hides hour-to-hour errors. Hourly,
  synthesized inputs do worse than local persistence.
* With their own inputs, eight cities reach hourly MAE 0.340 dB, but
  **Davos (+2.25 dB bias) and Madrid (+0.99 dB)** fail: their surface
  pressure (810-859 and 912-960 hPa) lies outside the training range
  (946-1048 hPa). Swapping their coordinates for the nearest training
  city's changes the bias by less than 0.01 dB, which points to the
  pressure input rather than the coordinates. With MSL pressure both lie
  inside the training range (984-1044 hPa); the `multitask_msl` experiment
  supports this (below).
* Unseen-city coordinates lie outside the six training locations, so the
  latitude/longitude inputs are extrapolated in every mode.
* Scope: ten cities, one seed per model, one evaluation period (Apr-Dec
  2024). Differences between individual cities or between models of a few
  hundredths of a dB should not be generalised.
* The paper's own Table 8 predictions are nearly constant across cities
  (1.036-1.150 dB) while its true values vary (0.787-1.228 dB).

**MSL-pressure model (`multitask_msl`)**, same protocol, mean over ten cities:

| Input mode | Error of period means | Hourly MAE | Hourly RMSE | PICP 95 % |
|---|---|---|---|---|
| Synthesized, weighted (tau 500 km) | 0.179 | 0.614 | 1.212 | 82.7 % |
| Synthesized, median of 3 | 0.144 | 0.514 | 1.151 | 79.9 % |
| Own inputs | **0.049** | **0.340** | **0.890** | 93.1 % |

Own inputs, surface -> MSL pressure: Davos error of means 2.252 -> 0.046 dB
(hourly MAE 2.36 -> 0.36), Madrid 0.991 -> 0.044 dB (MAE 1.08 -> 0.19); the
other eight cities change by less than 0.05 dB. With its own weather the
MSL model beats local persistence on hourly MAE (0.340 vs 0.357 dB) and
RMSE (0.890 vs 1.052 dB). Synthesized inputs are barely affected (Madrid
still 0.64 dB: its nearest training city, Vigo, is much wetter), so with
MSL pressure most of the remaining unseen-city error appears to come from
the input synthesis rather than from the model itself (one seed, ten
cities). Figure: `msl_vs_surface_unseen`; tables:
`results/paper/tables/table8_multitask_msl.csv`. tau sensitivity (MAE):
250 km 0.618, 500 km 0.614, 1000 km 0.616.

---

## 7. Comparison with the paper

| Quantity | Paper | Ours |
|---|---|---|
| Trained-city RMSE | 0.026 dB | 0.936 dB |
| Trained-city MAE | 0.009 dB | 0.351 dB |
| PICP 95 % | 99.23 % | 96.6 % |
| Sharpness | 0.035 dB | 0.557 dB |
| Early stopping | 23 epochs | best epoch 14 of 20 |
| Unseen cities, average error | 0.213 dB | 0.189 dB (error of means), 0.622 dB (hourly MAE) |
| Unseen true EPL range | 0.79-1.23 dB | 0.78-1.34 dB |
| Unseen cities, MSL model, own inputs | - | 0.049 dB (error of means), 0.340 dB (hourly MAE) |
| Table 10 (no ITU features) | 0.028 dB (worse than full) | 0.940 +- 0.006 dB, 3 seeds (no difference) |
| Table 9, EPL only | 0.015 dB (better than multi-task) | 0.946 +- 0.007 dB, 3 seeds (0.006 worse) |
| Table 9, weather only (std. RMSE) | 0.892 (worse than multi-task 0.761) | 0.480, 1 seed (better than multi-task 0.525 +- 0.007) |
| Fig. 2 reading (minute data) | 0.026 dB | LSTM 0.438 dB; persistence 0.033; linear extrapolation 0.013 |
| Best variant (Fig. 1 heads) | - | 0.916 +- 0.006 dB, 3 seeds (seed 42: 0.910) |
| Main model, 3 seeds | - | 0.940 +- 0.004 dB |

What we can and cannot confirm:

* **Cannot confirm** RMSE 0.026 dB for a 5-hour forecast. On our labels the
  naive persistence forecast already has RMSE 1.04 dB, because hourly rain
  changes EPL by several dB within five hours; any 5-hour forecast of
  ITU-R EPL built from ERA5 rain will have errors of this order.
* **Consistent with** the paper's numbers: a 5-minute-ahead forecast on
  minute-interpolated data (Fig. 2), where persistence alone gives
  0.02-0.04 dB and the error is almost flat across steps, like the paper's
  Table 7.
* **Partly confirm** the unseen-city "average error" when measured the
  paper's way; **do not confirm** that the synthesized inputs give accurate
  hourly predictions.
* **Do not confirm** the ablation conclusions (Tables 9 and 10): ITU-R
  features give no measurable gain for this model (three seeds), multi-task training gives
  at most a small EPL gain (three seeds) and costs weather accuracy
  (`weather_only` one seed).
* **Cannot resolve** which pressure variable the paper used (Table 2 says
  MSL, Sect. 3.1 says surface). We show only that, in our pipeline, the MSL
  input avoids the out-of-range failures at Davos and Madrid when a city is
  run on its own weather. The paper's Table 8 cannot discriminate: its
  inputs are synthesized from neighbouring training cities, whose pressure
  is in range under either variable (our synthesized mode shows no such
  failure with surface pressure either).
* The paper's labels cannot be compared directly: its dataset is
  available only on request, and its LWP proxy, sigma features, elevation
  and polarisation are not specified.

---

## 8. Figures and tables

All in `results/paper/figures/` as PNG (300 dpi) and PDF:

| File | Content |
|---|---|
| `fig1_architecture` | Fig. 1, model as implemented (notes the head difference) |
| `fig2_pipeline` | Fig. 2, pipeline as implemented |
| `fig3_pred_vs_true` | Fig. 3, predicted vs true EPL |
| `fig4_horizon_timeseries` | Fig. 4, hours +1..+5 with 95 % intervals (Chilbolton) |
| `fig4_city_timeseries_hour1`, `fig4b_horizon_examples` | Fig. 4 per city; largest-EPL 5-h windows |
| `fig5_unseen_cities` | Fig. 5, unseen-city means and hourly errors |
| `fig6_error_by_region` | Fig. 6 (region grouping is ours) |
| `fig7_unseen_map` | Fig. 7, locations coloured by hourly MAE |
| `training_curves` | train / validation loss for every variant |
| `residuals`, `error_vs_horizon` | error distribution, error vs EPL level, RMSE vs hour |
| `rmse_mae_by_city`, `rmse_by_city_and_hour` | location-wise results (Table 7) |
| `calibration` | interval calibration, PICP per city |
| `weather_head_rmse` | weather head vs persistence per variable and hour |
| `baselines_and_conditions` | model vs baselines; error by weather condition |
| `dataset_components`, `dataset_epl_hist`, `dataset_sampling` | EPL components, distributions, sampling |
| `unseen_timeseries` | unseen cities, hour +1 time series |
| `ablations` | Tables 9/10 and extra variants: EPL RMSE, MAE, weather RMSE (seed 42) |
| `seed_variation` | every seed of the four repeated variants, average and hour-1 RMSE |
| `minute_experiment` | Fig. 2 reading: LSTM vs persistence vs linear extrapolation per minute step |
| `msl_vs_surface_unseen` | unseen-city hourly MAE, surface vs MSL pressure, own vs synthesized inputs |
| `fig5/6/7_*_multitask_msl`, `unseen_timeseries_multitask_msl` | Figs. 5-7 for the MSL model |

Tables 7-11 and a summary of all variants: `results/paper/tables.md`
(CSV versions in `results/paper/tables/`).

---

## 9. Tests performed

Mostly scratch scripts (kept out of the repository by choice); all on real
data. The final-results audit is in the repository and can be re-run:
`.venv\Scripts\python evaluation\audit_results.py` (report in
`results/paper/audit_report.json`). It recomputes every test metric from the
saved predictions and labels with independent code and checks the JSON, the
table CSVs, `tables.md`, unseen-city summaries and figure freshness.

| Check | Result |
|---|---|
| `audit_results.py` (latest run 2026-10-26): run files complete for all 15 runs, logs = configs, no resumed runs, metrics recomputed = JSON = tables, seed-run RMSEs = `seed_runs.csv`, baselines recomputed, weather inverse transform = raw values, scalers fitted on training data, unseen summaries consistent, every figure PNG/PDF pair from the same run and newer than the metrics | PASS, 127/127 checks |
| `tests/test_pipeline.py` (`python -m unittest discover -s tests -v`): metric formulas, chronological split with purge gap, saved splits share no hour, scaler fitted on training targets, minute interpolation makes linear extrapolation exact within an hour, neighbour weights, seed-run folders, committed LWP coefficients reproduce the cloud labels, EPL = sum of components, seed summary = seed-level results | PASS, 12/12 (2026-10-26) |
| Seed means, std, ranges and paired differences recomputed independently from seed-level results | PASS (2026-10-26) |
| Main unseen-city evaluation re-executed | PASS: identical tables (max difference 0.0) |
| `requirements.txt` pins equal the installed versions; `pip check` | PASS |
| Retrain `multitask` 2 epochs with today's code, seed 42 | PASS: losses equal the original run to 1e-7 |
| Validation-set ranking of variants equals test-set ranking | PASS (fig1heads best on both) |
| Earlier scratch tests re-run on 2026-10-24 (`test_dataset.py`, `verify_pipeline.py`, `test_physics.py`) | PASS |
| Re-downloaded weather equals the 30 original files on all shared columns | PASS (0 differ) |
| MSL download: surface pressure equals the physics tables, all 16 cities | PASS (max diff 0.000 hPa) |
| P.618 steps 1-9 vs `itur.rain_attenuation`, 3 sites x 8 rates | PASS (0.0 dB) |
| P.676 vs `itur` approximate slant path | PASS (identical) |
| P.840 K_l(39 GHz, 0 degC) = 1.233 by hand | PASS |
| Labels finite, non-negative, EPL = sum, rain > 0 only with rain (16 cities) | PASS |
| No hour shared between train / val / test in any city | PASS |
| Inputs equal feature rows t-59..t (54 random windows, all splits) | PASS |
| sigma features use only t-23..t | PASS |
| Targets equal the physics tables at t+1..t+5 | PASS |
| Scaler statistics equal training-window statistics | PASS (first attempt failed: float32 summation in the test) |
| Dataset rebuild bit-identical; MSL dataset has identical anchors and EPL targets | PASS |
| Minute dataset: one window checked by hand against interpolation | PASS |
| Keras NLL equals NumPy Eq. 10 | PASS |
| Training deterministic (epoch 1 identical in two runs) | PASS |
| Saved predictions equal a fresh prediction from `best.keras` | PASS, all 7 variants (max 1.9e-6) |
| Quick NumPy metrics equal `evaluate_paper_model.py` output | PASS (all variants) |
| All ablation variants build, train one step and predict | PASS |
| Restructured evaluation reproduces the earlier metrics exactly | PASS |
| `run_pipeline.ps1` syntax | PASS (parser); the complete script has not been executed end to end (each step was run individually) |

---

## 10. Limitations

0. The authors' dataset (available only on request) was not used, so our
   labels and inputs cannot be compared with theirs value by value.
1. Labels are ITU-R model output, not link measurements (as in the paper).
2. The cloud liquid-water proxy is our own approximation (the paper gives no
   formula) and is weak (held-out correlation 0.38, R^2 about 0); cloud
   attenuation is therefore only roughly modelled.
3. Hourly-mean rain rates and P.618 factors outside their intended range;
   ERA5 drizzle adds many small rain events (rain in 42 % of Bergen hours).
4. Fixed 30 deg elevation; no orbital geometry or scintillation.
5. The chronological test period covers Apr-Dec 2024 only.
6. Only three seeds for four variants, one seed for `weather_only`,
   `multitask_msl` and `minute_multitask`; no formal significance tests.
   Unseen-city results use the seed-42 models only.
7. Heavy rain is badly under-predicted (rare in training: 66 of 42,000
   windows above 14 dB); Gaussian intervals can be negative.
8. High-altitude unseen cities are out of distribution for surface pressure
   (avoided by the MSL variant, which is not the main model).
9. Training times are wall-clock and include laptop standby pauses
   (`multitask`, `multitask_fig1heads_seed1`); not comparable across runs.
10. Differences below about 0.01 dB between variants are within the
    measured seed spread (Sect. 5.5); only the Fig. 1-head gain exceeds it.
11. The minute experiment uses interpolated hourly data, not real
    1-minute observations.

---

## 11. Files

Created: `config.py`, `run_pipeline.ps1`, `README.md`, `REPLICATION.md`,
`STATUS.md` (since removed), `physics/calibrate_lwp_proxy.py`,
`data_pipeline/build_paper_dataset.py`, `data_pipeline/build_minute_dataset.py`,
`data_pipeline/download_pressure_msl.py`, `training/train_paper_model.py`,
`training/run_queue.py`, `training/queue.txt`,
`evaluation/evaluate_paper_model.py`, `evaluation/evaluate_unseen_cities.py`,
`evaluation/plot_dataset.py`, `evaluation/make_diagrams.py`,
`evaluation/make_figures.py`, `evaluation/make_tables.py`,
`evaluation/audit_results.py`, `evaluation/summarize_runs.py`,
`evaluation/calibration_analysis.py`, `evaluation/minute_diagnostics.py`,
`evaluation/export_docs_assets.py`,
`training/queue_seeds.txt`, `tests/test_pipeline.py`, `requirements.txt`,
`physics/lwp_proxy_calibration.json` (fitted coefficients, moved here from
`results/paper/` so a fresh clone can build the labels), `docs/`
(report, interview notes, references, `assets/` result snapshot),
`GITHUB_CHECKLIST.md` (since removed).

Backups: `backups/2026-10-23/` (original prototype files, REPLICATION v1),
`backups/2026-10-24/` (results and docs just before the final evaluation;
training scripts before the seed option; the stale first evaluation log),
`backups/2026-10-25/` (results and docs before the repeated-seed update),
`backups/2026-10-26/` (docs, scripts and an MD5 list of all model artefacts
before the final documentation pass).

Experiment configuration: every run folder has `run_config.json` (variant,
seed, dataset, features, heads, all hyperparameters, epochs, best epoch,
wall-clock hours, parameter count, TensorFlow version) and
`training_log.csv`. All runs in one file: `results/paper/experiments_summary.json`
(also `seed_runs.csv`).

Modified (originals in `backups/2026-10-23/`): `physics/build_attenuation.py`,
`data_pipeline/cities.py`, `data_pipeline/download_openmeteo.py`,
`.gitignore`.

Not touched: original weather CSVs (`open_meteo/`), earlier arrays, scalers
and prototype models in `training/`, `models/` and `data/processed/`.

### Version control

What is tracked is defined by `.gitignore`. Generated results are not
tracked except the curated snapshot in `docs/assets/` and the small result
files cited by the docs (causal metrics, the 13 figures, audit outputs).
Publication history is in `git log`.

---

## 12. Replication gap audit (2026-10-26)

A specification-by-specification audit against the paper (text, equations,
tables and the figure images) is in `docs/REPLICATION_GAP_AUDIT.md`, with
results in `docs/REPLICATION_RESULTS.md` and the remaining optional
experiment in `docs/REPLICATION_EXPERIMENT_PLAN.md`. Main new findings:

* Fig. 4's test-sample axis (about 470,000) is reproduced exactly by the full
  2020-2024 series interpolated to 1 minute with a 5-minute stride (473,484
  test windows); on those windows persistence gives 0.035 dB and linear
  extrapolation 0.011 dB for +1..+5 minutes (`replication_audit_checks.py`).
* Fig. 1 draws Dense 256 -> 256 heads, so `multitask_fig1heads` is the
  paper-faithful architecture; the "main" model is a deviation.
* Paper-internal inconsistencies: Table 7's overall 0.026 does not follow
  from its city values (mean 0.036, pooled RMS 0.063); Table 8's 0.213 is
  the average of the Fig. 5 values, which differ from Table 8 for Berlin and
  Stockholm; Table 9's "40 %" is an EPL-only advantage; Fig. 6 implies
  Berlin is grouped in Western Europe (our earlier figure used Eastern; the
  paper grouping is in `results/replication_audit/`).
* Label sensitivity: elevation changes the label level (paper's unseen mean
  0.911 dB lies between our 30 and 45 deg values) but no tested elevation or
  rain input reproduces the paper's small rain tail (max about 6.7 dB). We did
  not change any choice to match published values.
* No implementation bug was found that explains the gap.

## 13. References

See [docs/REFERENCES.md](docs/REFERENCES.md) for the paper, the data sources
(Open-Meteo, ERA5), the ITU-R Recommendations and the `itur` library.
