"""Alternative reading of the paper: 1-minute data, 60-min input, 5-step horizon.

Evidence in the paper for this reading (it contradicts the "5 h" in the text):
  * Fig. 2: "Temporal Alignment 1-minute uniform sampling",
    "Sequence Alignment L=60 min history window", "Prediction Horizon:
    5 steps", "Stride: 5 minutes".
  * Abstract / Sect. 2: "60 min historical sequences".
  * Table 4 lists a "Minute-of-hour" feature.
  * Table 7: RMSE is almost flat from "Hour 1" to "Hour 5" (0.026-0.027).

ERA5 is hourly, so 1-minute values can only come from interpolation. We use
linear interpolation of every hourly feature and target. Each existing
anchor (same 10,000 sampled hours per city and the same chronological
split as the main dataset) gets an anchor minute = anchor hour + offset,
offset drawn from {0, 5, ..., 55} (seed 42), matching the 5-minute stride.
Inputs: minutes m-59..m. Targets: minutes m+1..m+5.

Usage:
    python data_pipeline/build_minute_dataset.py
Output: data/processed/paper_minute/ (same file layout as data/processed/paper/)
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from data_pipeline.build_paper_dataset import add_features
from data_pipeline.cities import CITIES

OUT = config.DATASET_DIR_MINUTE


def interpolate(values, minutes):
    """Linear interpolation of an hourly table at fractional minute times.
    values: (n_hours, n_cols); minutes: array of minutes since hour 0."""
    pos = minutes / 60.0
    i = np.floor(pos).astype(int)
    w = (pos - i)[..., None]
    return (1 - w) * values[i] + w * values[np.minimum(i + 1, len(values) - 1)]


def main():
    rng = np.random.default_rng(config.SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for split in ("train", "val", "test"):
        meta = pd.read_csv(config.DATASET_DIR / f"meta_{split}.csv", parse_dates=["anchor_time"])
        parts = []
        for city, coords in CITIES.items():
            df = pd.read_csv(config.PHYSICS_DIR / f"{city}_physics.csv", parse_dates=["valid_time"])
            df = add_features(df, coords["latitude"], coords["longitude"])
            anchors = meta.loc[meta.city == city, "anchor_time"]
            hour_idx = ((anchors - df.valid_time[0]) / pd.Timedelta(hours=1)).to_numpy().astype(int)
            m = hour_idx * 60 + rng.choice(np.arange(0, 60, 5), len(hour_idx))   # anchor minute

            past = m[:, None] + np.arange(-config.SEQ_LEN + 1, 1)                 # 60 minutes
            future = m[:, None] + np.arange(1, config.HORIZON + 1)                 # 5 minutes
            feats = df[config.FEATURES].to_numpy(np.float64)
            weather = df[config.WEATHER_TARGETS].to_numpy(np.float64)
            epl = df[[config.EPL_COLUMN]].to_numpy(np.float64)
            parts.append({
                "X": interpolate(feats, past).astype(np.float32),
                "weather": interpolate(weather, future).astype(np.float32),
                "epl": interpolate(epl, future)[..., 0].astype(np.float32),
                "epl_now": interpolate(epl, m)[:, 0].astype(np.float32),
                "epl_prev": interpolate(epl, m - 1)[:, 0].astype(np.float32),
                "city": np.full(len(m), city),
                "anchor_time": df.valid_time[0] + pd.to_timedelta(m, unit="min"),
            })
        data[split] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    n_feat, n_weather = len(config.FEATURES), len(config.WEATHER_TARGETS)
    x_scaler = StandardScaler().fit(data["train"]["X"].reshape(-1, n_feat))
    w_scaler = StandardScaler().fit(data["train"]["weather"].reshape(-1, n_weather))
    joblib.dump(x_scaler, OUT / "feature_scaler.pkl")
    joblib.dump(w_scaler, OUT / "weather_scaler.pkl")
    for split, d in data.items():
        X = x_scaler.transform(d["X"].reshape(-1, n_feat)).reshape(d["X"].shape)
        W = w_scaler.transform(d["weather"].reshape(-1, n_weather)).reshape(d["weather"].shape)
        np.save(OUT / f"X_{split}.npy", X.astype(np.float32))
        np.save(OUT / f"weather_{split}.npy", W.astype(np.float32))
        np.save(OUT / f"weather_raw_{split}.npy", d["weather"])
        np.save(OUT / f"epl_{split}.npy", d["epl"])
        np.save(OUT / f"epl_now_{split}.npy", d["epl_now"])
        np.save(OUT / f"epl_prev_{split}.npy", d["epl_prev"])
        pd.DataFrame({"city": d["city"], "anchor_time": d["anchor_time"]}).to_csv(
            OUT / f"meta_{split}.csv", index=False)
        print(f"{split}: X {d['X'].shape}, epl {d['epl'].shape}")

    with open(OUT / "manifest.json", "w", encoding="utf-8") as file:
        json.dump({
            "created_by": "data_pipeline/build_minute_dataset.py",
            "interpretation": "1-min linear interpolation of hourly ERA5/ITU-R series; "
                              "60 one-minute input steps; targets 1..5 minutes ahead",
            "anchors": "same sampled hours and splits as data/processed/paper, plus a minute offset "
                       "from {0,5,...,55} (seed 42)",
            "features": config.FEATURES,
            "shapes": {s: {k: list(np.shape(v)) for k, v in d.items() if k in ("X", "weather", "epl")}
                       for s, d in data.items()},
        }, file, indent=2)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
