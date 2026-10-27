# Replication results

Date: 2026-10-26. Companion to `docs/REPLICATION_GAP_AUDIT.md` (evidence for
every specification) and `docs/REPLICATION_EXPERIMENT_PLAN.md`. Published
values are labelled **(reference)**; all other numbers are measured in this
repository.

**Verdict: partial replication / independent reconstruction.** The
pipeline, model and evaluation follow the paper wherever it is specific; the
authors' dataset is unavailable; several method details are unspecified or
contradictory; the paper's headline accuracy was not reproduced.

## 1. Paper specifications

Six training cities, five years of hourly ERA5 (Open-Meteo), 10,000 stratified
samples per city, EPL = gas (P.676) + cloud (P.840 with an RH/precipitation
LWP proxy) + rain (P.838/P.618, P.839) at 39 GHz and a fixed elevation; 14
inputs over 60 steps; 3 x LSTM(256), dropout 0.2; weather head and Gaussian EPL
head (Dense 256 -> 256 -> 25 / 10, Fig. 1); loss MSE + NLL (Eq. 10-11);
Adam 1e-4, batch 128, <= 50 epochs, patience 6, seed 42; chronological
70/15/15 split; ten unseen cities with inputs synthesized from K = 3
neighbours (Eq. 12-13). Reported: RMSE 0.026 dB, unseen-city average error
0.213 dB, PICP 99.23 % (also 99.03 %), Tables 9-10 ablations.

## 2. Implemented methodology

As specified above, with these documented choices: 30 deg elevation, circular
polarisation, our fitted LWP proxy, P.618 steps 1-9 with the hourly rain rate,
sigma features = trailing 24-h standard deviation, one global scaler, surface
pressure (MSL as a variant), hourly sampling (60 h in, 5 h out) for the main
protocol. Paper-faithful architecture: `multitask_fig1heads` (Fig. 1 heads);
the earlier "main" model without the hidden head layer is kept as a
deviation for continuity.

## 3. Dataset provenance

* Weather: Open-Meteo archive API (ERA5), hourly, 2020-01-01..2024-12-31 for
  Athens, Bergen, Budapest, Chilbolton, Vigo, Warsaw (43,848 h each, no gaps);
  2023-2024 for Berlin, Oslo, Rome, Stockholm, Bucharest, London, Davos,
  Paris, Marseille, Madrid; MSL pressure for all 16.
* Labels: `physics/build_attenuation.py`, `itur` 0.4.0 (P.676-12, P.840-8,
  P.838-3, P.618-13, P.839-4), LWP coefficients `physics/lwp_proxy_calibration.json`.
* Hourly dataset: `data/processed/paper/` (manifest): 42,000 / 8,898 / 8,916
  windows; test about 24 March-9 April to 31 December 2024 per city.

## 4. Experiment configuration

15 completed runs, all with patience 6, best weights restored, TensorFlow op
determinism; settings in each `models/paper/**/run_config.json`; summary in
`results/paper/experiments_summary.json`. Seeds 42, 1, 2 for `multitask`,
`multitask_fig1heads`, `epl_only`, `no_itu`; seed 42 for `weather_only`,
`multitask_msl`, `minute_multitask`.

## 5. Numerical results (hourly protocol, six-city test set)

| Model | Test EPL RMSE (dB), seeds 42/1/2 | Hour-1 RMSE | MAE | PICP 95 % |
|---|---|---|---|---|
| `multitask_fig1heads` (paper-faithful heads) | **0.916 +- 0.006** | 0.643 | 0.343 | 96.3 % |
| `multitask` (no hidden head layer) | 0.940 +- 0.004 | 0.708 | 0.349 | 96.4 % |
| `no_itu` | 0.940 +- 0.006 | 0.712 | 0.349 | 96.3 % |
| `epl_only` | 0.946 +- 0.007 | 0.715 | 0.356 | 96.6 % |
| Persistence | 1.042 | 0.596 | 0.352 | - |
| Paper (reference) | 0.026 | 0.026 | 0.009 | 99.23 % |

* Overall RMSE is the pooled RMS over all windows and steps. Other
  aggregations differ by at most 0.03 dB (fig1heads seed 42: pooled 0.910,
  mean of city RMSEs 0.889).
* Paired differences (same seed): fig1heads - main -0.024 +- 0.003 dB (all
  seeds); main - EPL-only -0.006 +- 0.005 dB (same sign, small); no_itu - main
  -0.000 +- 0.004 dB (sign changes).
* Weather (standardised RMSE): weather-only 0.480 (1 seed), multi-task 0.525
  +- 0.007; the paper reports the opposite direction (0.892 vs 0.761).
* Calibration: overall 96.6 % coverage but 99.4 % in dry hours, 86.4 % with
  precipitation, 37.0 % at >= 2.5 mm/h; nominal 50 % covers 77.8 %.

Unseen cities (seed-42 models, every hour 2024-04-09..2024-12-31):

| Model / inputs | Error of period means | Hourly MAE | Hourly RMSE |
|---|---|---|---|
| Main, synthesized (paper method) | 0.189 | 0.622 | 1.214 |
| Main, median of 3 neighbours (Fig. 2) | 0.151 | 0.536 | 1.160 |
| MSL model, own weather | 0.049 | 0.340 | 0.890 |
| Persistence of own EPL | 0.000 | 0.357 | 1.052 |
| Paper (reference) | 0.213 | - | - |

Minute readings:

| Setup | RMSE (dB) |
|---|---|
| Our minute model (10,000-anchor subset, 8,916 test windows) | 0.438 (fails on rare heavy rain) |
| Full-series minute protocol, persistence, +1..+5 min (473,484 windows) | 0.035 |
| Same, linear extrapolation | 0.011 |
| Same windows, +1..+5 h, persistence | 0.945 |
| Paper (reference) | 0.026 |

## 6. Tables and figures

| Paper item | Our version | Status |
|---|---|---|
| Table 7 | `results/paper/tables.md`, `results/replication_audit/paper_vs_ours.md` | reproduced in layout; values differ (protocol) |
| Table 8 | same | reproduced in layout and metric definition |
| Table 9 | same | reproduced; conclusion not reproduced |
| Table 10 | same | reproduced; conclusion not reproduced |
| Table 11 | `results/paper/tables.md` | qualitative, prior-work values quoted |
| Fig. 1, Fig. 2 | `results/paper/figures/fig1_architecture`, `fig2_pipeline` | diagrams of our implementation |
| Fig. 3, Fig. 4 | `fig3_pred_vs_true`, `fig4_horizon_timeseries` | our predictions |
| Fig. 5 | `fig5_unseen_cities` | our unseen-city results |
| Fig. 6 | `results/replication_audit/fig6_regions_paper_grouping` (paper grouping, published bars shown hatched as reference) | grouping reconstructed from the paper's arithmetic |
| Fig. 7 | `fig7_unseen_map` | our errors on the map; the paper's "accuracy %" is undefined and not reproduced |

No value digitised from the paper's figures is plotted as our result.

## 7. Reproducibility

```powershell
.venv\Scripts\python -m unittest discover -s tests -v       # 16 tests
.venv\Scripts\python evaluation\audit_results.py            # 127 checks
.venv\Scripts\python evaluation\replication_audit_checks.py  # paper arithmetic, minute baselines
.venv\Scripts\python evaluation\label_sensitivity.py         # elevation / rain-input sensitivity
.venv\Scripts\python evaluation\make_paper_comparison.py     # published vs ours, Fig. 6 grouping
.\run_pipeline.ps1 -EvaluateOnly                            # everything above from checkpoints
```

Training is deterministic on this machine (2-epoch retrain equals the
original losses to 1e-7).

## 8. Remaining deviations from the paper

1. Sampling/horizon protocol: hourly main protocol vs the minute reading the
   paper's numbers fit (audit B). The minute reconstruction with a trained
   model is planned, not run (needs about 55-70 h CPU or a GPU).
2. Rain-attenuation magnitudes: our label tail (max 25.6 dB) is far larger
   than the paper's visible range (about 6.7 dB) for every elevation tested.
3. Unspecified choices: elevation, polarisation, LWP proxy, sigma features,
   normalisation by location, unseen-city evaluation period, tau.
4. ITU-R versions P.676-12 / P.618-13 instead of the cited -13 / -14.
5. Temporal features of Table 4 not used (Table 2 and Fig. 1 list 14 features).

## 9. Limitations due to unavailable data

The authors' labels, splits and predictions are unavailable, so no
value-by-value comparison is possible; the internal inconsistencies of
Tables 7-9 and Figs. 5-7 (audit H) cannot be resolved; labels are ITU-R model
outputs, not link measurements, in both works.

## 10. Conclusion

This is a **partial replication** (independent reconstruction from public
data), **not an exact reproduction**. The architecture, loss, training
settings, split and unseen-city method were reproduced; the paper-faithful
model reaches 0.916 +- 0.006 dB on a genuine 5-hour hourly forecast, about
12 % better than persistence overall. The published 0.026 dB is consistent
with a minutes-ahead forecast on interpolated data, where trivial baselines
reach 0.011-0.035 dB, but this remains a hypothesis. No implementation bug was
found that explains the gap.
