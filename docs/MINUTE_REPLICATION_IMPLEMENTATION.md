# Full-series 1-minute reconstruction: implementation and smoke test

Date: 2026-10-27. This is a **separately named independent reconstruction**
of the protocol suggested by the paper's Figs. 2 and 4 (see
`docs/REPLICATION_GAP_AUDIT.md`, Sect. B). It does not replace the hourly
results, which remain the primary measured baseline
(`multitask_fig1heads`, 0.916 +- 0.006 dB over three seeds).

## 1. Baseline verification (before any change)

| Check | Result |
|---|---|
| Git status at start | 47 entries (pre-existing modified prototypes and untracked pipeline files), saved in `backups/2026-10-27/git_status_start.txt`; nothing staged |
| Unit tests | 20/20 pass |
| Results audit (`evaluation/audit_results.py`) | 127/127 pass |
| Model artefacts | 90/90 unchanged (MD5 list `backups/2026-10-26/model_artifacts_md5.txt`) |
| Results and datasets | MD5 of 214 files in `results/paper/`, `data/processed/paper*`, `physics_paper/` saved to `backups/2026-10-27/results_and_datasets_md5.txt` |
| Lazy source split counts | as previously reported, but see the boundary fix in Sect. 2.1 |

## 2. Data pipeline

### 2.1 Source validation (`data_pipeline/minute_full_source.py`)

* Features: the hourly `add_features` (14 features; sigma features are a
  trailing 24-hour rolling standard deviation of hourly values, past only),
  then linear interpolation to minutes with the existing `interpolate`
  (identical windows to `build_minute_dataset.py`, tested).
* Windows: inputs minutes t-59..t, targets t+1..t+5, anchors every 5 minutes
  from the first minute whose whole input window is finite, last anchor such
  that targets stay inside the series. No duplicate anchors; strictly
  increasing time; dimensions (60, 14), (5, 5), (5).
* **Implementation error found and fixed:** the prototype separated splits by
  SEQ_LEN + HORIZON = 65 minutes. Because a minute between hours h and h+1 is
  interpolated from both hourly rows, two hourly rows were still used by
  both sides of each split boundary (measured: rows 30,700-30,701 shared by
  train and validation in Athens). Fix: the first validation/test anchors are
  dropped until no hourly row is used by two splits. The prototype had not
  produced any results. New counts: **train 2,208,678, validation 473,076,
  test 473,112** windows (each city 368,113 / 78,846 / 78,852). Test:
  `TestMinuteFullSource.test_splits_share_no_minute_or_hourly_row_and_use_stride_5`.
* **Methodological property (not a coding error; inherent to the protocol):**
  interpolation puts information from the *next* hourly observation into
  every input minute between two hours. At minute t in hour h, the inputs at
  t-1 and t already determine the straight line to hour h+1, so the targets
  t+1..t+5 are known exactly unless they cross an hour boundary. This is why
  linear extrapolation is nearly exact (Sect. 4). Any score on this protocol
  measures reconstruction of interpolated values, not forecasting. A causal
  variant (only hourly values up to floor(t/60) available) would be a
  different protocol; it is not implemented.

### 2.2 Lazy batch loader (`data_pipeline/minute_full_loader.py`)

* `MinuteFullData`: the six city sources, anchor indices per split (optional
  deterministic fraction, the same fraction of each city), scalers.
* `MinuteBatches(keras.utils.PyDataset)`: builds each batch on request (6 ms
  per batch of 128 measured), returns `X (b, 60, 14)` float32 and
  `{"weather": (b, 5, 5), "epl": (b, 5)}`. Training order: seeded permutation
  over all (city, anchor) pairs, re-permuted at each epoch end; validation/test
  in time order. Memory: the hourly tables (42 MB in total) plus one batch
  (0.45 MB); materialised inputs would need 7.4 GB (train) and 1.6 GB (test).
* City balance: every city has the same number of anchors in every split, and
  fractional subsets take the same count per city.

### 2.3 Scalers (train only)

* Global `StandardScaler` (as in the hourly pipeline; the paper's "normalized
  by location" in Fig. 2 is not adopted, see the gap audit), fitted with
  `partial_fit` city by city in chunks of 200,000 minutes on the **training
  minute series**, every minute once (not on overlapping windows):
  features over first train anchor - 59 .. last train anchor (11,043,720
  minute rows), weather targets over first train anchor + 1 .. last + 5
  (11,043,390 rows). Validation/test minutes never enter. EPL targets in dB.
* Saved with feature order, means and scales in the experiment folder
  (`scalers.json`, `feature_scaler.pkl`, `weather_scaler.pkl`).
* Constant features (elevation; lat/lon within a city) get scale 1 for
  zero variance, as in the hourly pipeline.

### 2.4 Tests (`tests/test_pipeline.py`, `TestMinuteFullLoader`, 8 tests)

Shapes and finite values; deterministic seeded shuffle; each epoch is a
permutation of all anchors (no repeats); validation/test unshuffled;
deterministic, city-balanced fractions; scaler row counts equal the training
minute range and differ from validation statistics; batches equal direct
windows from the source (raw and after inverse scaling); the full anchor
index is batchable without materialising windows; streaming baseline RMSE
equals a direct computation.

## 3. Model variant `minute_full_fig1heads` (`training/train_minute_full.py`)

Reuses `build_model(14, ["weather", "epl"], head_hidden=256)` and
`gaussian_nll` from `training/train_paper_model.py` (that script and its
runs are unchanged). Tensor path:

| Stage | Shape |
|---|---|
| Input | (b, 60, 14) |
| LSTM 256 (sequences) -> Dropout 0.2 -> LSTM 256 (sequences) -> Dropout 0.2 -> LSTM 256 | (b, 256) |
| Weather head: Dense 256 ReLU -> Dense 25 -> Reshape | (b, 5, 5), MSE on standardised targets |
| EPL head: Dense 256 ReLU -> Dense 10 (5 means + 5 log-variances) | (b, 10), Gaussian NLL (Eq. 10) |

1,468,707 parameters (same as the hourly `multitask_fig1heads`). Loss
weights 1.0 / 1.0; Adam 1e-4; log-variance clipped to [-12, 6], so sigma lies
in [0.0025, 20.1] dB and the NLL cannot overflow. Matches Fig. 1 ("Dense:
256 -> 256 -> 25 / 10"); the ReLU in the hidden head layer is our choice (the
paper gives no activation). Checkpoints: `models/minute_full_<name>/`;
everything else: `results/minute_full_<name>/`; the script refuses to reuse a
finished name.

## 4. Streaming evaluation (`evaluation/evaluate_minute_full.py`)

Scores every requested test anchor exactly once (checked: anchors_scored =
anchors_requested), keeping only running sums. Same anchors, target minutes
and horizons for all methods:

* Persistence: EPL_hat(t+k) = EPL(t)
* Linear extrapolation: EPL_hat(t+k) = EPL(t) + k (EPL(t) - EPL(t-1)), k = 1..5 min
* Model: Gaussian mean; also 95 % coverage, interval width, NLL; weather RMSE
  (standardised and physical units); by city; by precipitation class at the
  target minute (dry, light 0-2.5 mm/h, heavy >= 2.5 mm/h).

## 5. Smoke test (the only training run)

Command: `python training/train_minute_full.py --name smoke --train-fraction 0.01 --val-fraction 0.01 --test-fraction 0.01 --epochs 1`
(time limit 5 min on training; not reached). Outputs: `results/minute_full_smoke/`,
`models/minute_full_smoke/`.

| Item | Value |
|---|---|
| Windows | train 22,086 (3,681 per city, 1 %), validation 4,728, test 4,734 |
| Epochs | 1 (173 batches of 128) |
| Wall time | 4 min 46 s in total: sources 2 s, scalers 11 s, training + validation 236 s (validation 21 s), save/reload 3 s, test evaluation 9 s, baselines on the full test set 17 s |
| Checkpoint | saved and reloaded; predictions identical after reload (max difference 0.0); all predictions finite; losses finite |
| Hardware state | **laptop on battery** (45 %), CPU at 2.0 GHz; 1.19 s per training batch (median). A profile showed the same 1.15-1.2 s per batch on hourly arrays, i.e. the loader (6 ms) is not the bottleneck |

Test results, 1 % of test anchors (4,734 windows x 5 minutes), seed 42:

| Method | RMSE (dB) | MAE (dB) | RMSE minute 1 / 5 |
|---|---|---|---|
| Model after 1 epoch on 1 % of data | 0.754 | 0.147 | 0.717 / 0.774 |
| Persistence | 0.039 | 0.010 | 0.012 / 0.060 |
| Linear extrapolation | 0.009 | 0.001 | 0.003 / 0.013 |
| Paper (reference) | 0.026 | 0.009 | |

Model uncertainty: PICP 97.5 %, mean 95 % interval width 0.69 dB, NLL -2.15.
By target-minute class (model / persistence / linear): dry 0.054 / 0.005 /
0.002; light 0.558 / 0.066 / 0.016; heavy 7.98 / 0.25 / 0.04 dB. The model
result only shows that the pipeline trains and evaluates end to end; one
epoch on 1 % of the data is not a measurement of the method.

Baselines on **every** test anchor (473,112 windows): persistence 0.0358 dB,
linear extrapolation 0.0145 dB (by city 0.029-0.048 and 0.011-0.020;
heavy-rain minutes 0.220 and 0.103). These differ slightly from
`results/replication_audit/` (0.035 / 0.011) because that check used a
slightly different anchor range and no split filtering.

## 6. Cost of the full experiment (from measurements)

| Component | On battery (measured today) | On mains (derived from the hourly fig1heads runs, same model and batch size: 3.65 min per epoch of 329 + 70 batches, i.e. about 0.6 s per training batch) |
|---|---|---|
| Training batches per epoch | 17,256 (2,208,678 / 128) | 17,256 |
| Training per epoch | about 5.7 h (1.19 s/batch) | about 2.9 h |
| Validation per epoch (473,076 windows; 4.5 ms per window measured) | about 35 min | about 18 min |
| Per epoch | about 6.3 h | about 3.2 h |
| 20 epochs | about 125 h | about 64 h |
| Final test evaluation (473,112 windows; 1.8 ms per window measured) | about 14 min | about 7 min |
| Sources + scalers | 13 s | 13 s |
| Storage | 17.7 MB per checkpoint (best + final); full predictions (mu, sigma, y) about 28 MB if saved | same |

Uncertainty: per-batch time was measured over 173 batches and assumed
constant; hourly runs stopped after 14-36 epochs, so the total could be
0.7-1.8x the 20-epoch figure. Laptop standby pauses training (observed
before).

**CPU execution is not practical** (about 3 days on mains, 5 days on
battery, with standby risk). A GPU is justified for this run only. A short
GPU trial (one epoch on 10 % of anchors) would measure the real speed-up
before committing; note that `enable_op_determinism` may slow or block
cuDNN LSTM kernels, which needs checking. No cloud work was started.

## 7. Status, deviations and what remains

* Technically ready: data source, loader, train-only scalers, model,
  checkpointing, streaming evaluation and baselines all run end to end and are
  tested.
* Not done (needs approval): the full run; streaming evaluation of the full
  test set with a trained model (the code supports it:
  `python evaluation/evaluate_minute_full.py results/minute_full_<name> --test-fraction 1`).
* Deviations from the paper: global (not per-location) scaling; ReLU in the
  hidden head layers; our labels (see `docs/RAIN_LABEL_DIAGNOSTIC.md`); the
  interpretation itself (minute targets) is a hypothesis.
* Scientific caveat: because of interpolation, this protocol cannot measure
  forecasting skill, and linear extrapolation already reaches 0.0145 dB on
  the full test set, better than the paper's reported 0.026 dB. A full run
  would show whether the network approaches that trivial bound; it would not
  validate the paper's claim of 5-hour forecasting.
