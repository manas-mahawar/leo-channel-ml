"""Why linear extrapolation beats the minute LSTM (Fig. 2 reading).

Measures, on data/processed/paper_minute/:
  - linear-extrapolation error by anchor minute offset (exact inside an hour);
  - how rare heavy-rain windows are in training and test;
  - the LSTM's predictions on heavy-rain *training* windows (best.keras);
  - that the current EPL equals the sum of the three instantaneous inputs.

Usage:
    python evaluation/minute_diagnostics.py
Output: results/paper/minute_diagnostics.json
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

D = config.DATASET_DIR_MINUTE
MODEL = ROOT / "models" / "paper" / "minute_multitask" / "best.keras"
H = config.HORIZON


def main():
    y = np.load(D / "epl_test.npy").astype(float)
    now = np.load(D / "epl_now_test.npy").astype(float)
    prev = np.load(D / "epl_prev_test.npy").astype(float)
    offset = pd.read_csv(D / "meta_test.csv", parse_dates=["anchor_time"]).anchor_time.dt.minute.to_numpy()
    linear = now[:, None] + (now - prev)[:, None] * np.arange(1, H + 1)
    lin_err = np.abs(linear - y).max(axis=1)

    out = {"linear_extrapolation_max_error_dB_by_offset": {
               int(o): float(lin_err[offset == o].max()) for o in np.unique(offset)},
           "share_windows_linear_error_below_1e-4_dB": float((lin_err < 1e-4).mean())}

    y_train = np.load(D / "epl_train.npy").astype(float)
    out["windows_above_14dB"] = {"train": int((y_train.max(1) > 14).sum()), "train_total": len(y_train),
                                 "test": int((y.max(1) > 14).sum()), "test_total": len(y)}

    pred = np.load(ROOT / "models" / "paper" / "minute_multitask" / "test_predictions.npz")
    err = np.abs(pred["epl_mean"] - y).max(axis=1)
    out["test_windows_error_above_1dB"] = int((err > 1).sum())
    out["test_max_predicted_mean_dB"] = float(pred["epl_mean"].max())

    model = tf.keras.models.load_model(MODEL, compile=False)
    X = np.load(D / "X_train.npy", mmap_mode="r")
    for name, mask in (("train_heavy_above_14dB", y_train.max(1) > 14),
                       ("train_moderate_4_to_8dB", (y_train.max(1) > 4) & (y_train.max(1) < 8))):
        idx = np.flatnonzero(mask)[:500]
        p = model.predict(np.asarray(X[idx]), verbose=0)["epl"]
        mu, sd = p[:, :H], np.exp(0.5 * np.clip(p[:, H:], *config.LOG_VAR_CLIP))
        out[name] = {"windows": len(idx), "mean_true_dB": float(y_train[idx].mean()),
                     "mean_pred_dB": float(mu.mean()), "mean_sigma_dB": float(sd.mean()),
                     "RMSE_dB": float(np.sqrt(((mu - y_train[idx]) ** 2).mean()))}

    scaler = joblib.load(D / "feature_scaler.pkl")
    last = scaler.inverse_transform(np.load(D / "X_test.npy", mmap_mode="r")[:, -1, :])
    mu_idx = [config.FEATURES.index(f) for f in ("gas_mean", "cloud_mean", "rain_mean")]
    out["max_abs_diff_sum_of_mu_inputs_vs_EPL_now_dB"] = float(np.abs(last[:, mu_idx].sum(1) - now).max())

    with open(config.RESULTS_DIR / "minute_diagnostics.json", "w", encoding="utf-8") as file:
        json.dump(out, file, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
