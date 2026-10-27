# Early prototype: Athens, January 2020 (proxy labels)

This is the project's first end-to-end experiment, run before the main
pipeline existed. It is kept as evidence of how the project started. It is
**not** part of the main result, and its numbers must not be compared with
the causal evaluation in the [README](../README.md): the data, labels, period
and test set are all different.

## What it does

* **Site and period:** Athens only, 1-31 January 2020 (744 hours).
* **Weather:** ERA5 single levels downloaded directly from the Copernicus CDS
  (needs a CDS account), at the grid point 38.00 N, 23.75 E.
* **Labels:** a simplified 39 GHz, 30 deg attenuation proxy:
  `total_attenuation_proxy_dB = gas + cloud + rain`.
  * Gas: ITU-R P.676 slant path (`itur`, exact mode).
  * Cloud: ITU-R P.840 coefficient times ERA5 total column cloud liquid
    water (`tclw`). The main pipeline uses a fitted humidity proxy instead.
  * Rain: ITU-R P.838 specific attenuation times a fixed slant path from the
    P.839 rain height. This is an approximation, not a validated physical
    model: hourly ERA5 precipitation is treated as a rain rate in mm/h, and
    the full slant path is used without the ITU-R P.618 path-reduction
    factor. The rain values were not checked against P.618.
* **Inputs:** the same 14 feature names as the paper, but the attenuation
  statistics are trailing 60-hour rolling means and standard deviations.
* **Windows and split:** 60 h in, +1 to +5 h out. Chronological 70/15/15
  split by target hour. Forecast targets never cross a split boundary; input
  history may come from the previous split. Scalers are fitted on the
  training hours only.
* **Model:** 3 x 256 LSTM with a weather head (5 variables x 5 h, MSE) and an
  attenuation head (5 means + 5 log-variances, Gaussian NLL). Adam 1e-4,
  batch 128, early stopping (patience 6), seed 42.

## Scripts, in order

All are run from the repository root. Outputs go to the git-ignored
`data/processed/` and `models/` folders, except the training history.

| Step | Script | Output |
|---|---|---|
| 1 | `data_pipeline/build_dataset.py --city Athens --year 2020 --start-month 1 --end-month 1` | `data/processed/weather/monthly/Athens_2020_01.csv` |
| 2 | copy that file to `data/processed/athens_2020_01_weather.csv` (manual) | |
| 3 | `physics/gaseous_attenuation.py`, `physics/cloud_attenuation.py`, `physics/rain_attenuation.py` | `athens_2020_01_{gas,cloud,rain}.csv` |
| 4 | `data_pipeline/merge_physics.py` | `athens_2020_01_physics.csv` |
| 5 | `data_pipeline/create_features.py` | `athens_2020_01_features.csv` |
| 6 | **no script in the repository** (see Limitations) | `athens_2020_01_targets.csv` |
| 7 | `training/prepare_january_proxy.py`, then `training/prepare_january_multitask.py` | `athens_proxy_*.npy`, scalers |
| 8 | `models/multitask_january_proxy.py` | model, predictions, `models/multitask_january_proxy_history.json` |
| 9 | `training/evaluate_proxy_baseline.py`, `training/evaluate_persistence_baseline.py` | printed metrics |
| - | `training/inspect_extreme_targets.py`, `training/inspect_proxy_predictions.py` | printed diagnostics |

`data_pipeline/create_sequences.py` builds an older, unsplit set of windows
from the same files; step 7 replaced it.

## Result

The training script prints its metrics and does not save them. The numbers
below were recomputed on 2026-10-27 from the saved test predictions; the two
baselines were rerun on the same windows. The saved model was checked against
the saved history: it reproduces the best validation loss (-1.8703, epoch 20
of 26) and the saved test predictions.

Test set: 108 forecast origins (2020-01-27 08:00 to 2020-01-31 19:00)
x 5 horizons = 540 values.

| Method | MAE (dB) | RMSE (dB) |
|---|---|---|
| Multi-task LSTM | 0.201 | 0.581 |
| Persistence (last known proxy value) | 0.182 | 0.685 |
| Training-period mean | 0.250 | 0.580 |

Nominal 95 % intervals cover 78.7 % of the test values (mean width 0.435 dB).

**Reading:** the LSTM is worse than persistence on MAE and no better than a
constant on RMSE, and its intervals are too narrow. With under five days of
test data from one month and one city, this says little either way. These
limits are why the project moved to six cities, five years and a causal
evaluation.

## Limitations

* **Missing step:** the script that wrote `athens_2020_01_targets.csv` is not
  in the repository. The saved file equals gas + cloud + rain from
  `athens_2020_01_physics.csv` (checked on 2026-10-27, all 744 hours), so it
  can be recreated from that column sum.
* **Hard-coded paths:** the scripts use fixed file names for Athens, January
  2020.
* **Proxy labels:** these are simplified model outputs, not measurements, and
  they differ from the main pipeline's labels.
