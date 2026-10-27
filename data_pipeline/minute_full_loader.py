"""Lazy batches and train-only scalers for the full-series 1-minute reconstruction.

Builds on data_pipeline/minute_full_source.py (windows interpolated on demand
from each city's hourly table). Nothing larger than one batch is materialised.

Scaling (global over the six cities, as in the hourly pipeline):
  * feature scaler: StandardScaler.partial_fit over the interpolated minute
    series of each city's *training input range* (first train anchor - 59 ..
    last train anchor), every minute once, city by city in chunks; it is fitted
    on the underlying time series, not on overlapping windows;
  * weather-target scaler: same, over the training target range (first train
    anchor + 1 .. last train anchor + HORIZON);
  * validation and test minutes never enter either scaler.
EPL targets stay in dB (as in the hourly pipeline).

City balance: every city has the same number of anchors per split (same
period, same stride), and fractional subsets take the same fraction of
each city, so no city dominates.

Usage (inspection):
    python data_pipeline/minute_full_loader.py
"""

import json
import sys
from pathlib import Path

import joblib
import keras
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from data_pipeline.build_minute_dataset import interpolate
from data_pipeline.cities import CITIES
from data_pipeline.minute_full_source import CityMinuteSource

N_FEAT, N_WEATHER, H = len(config.FEATURES), len(config.WEATHER_TARGETS), config.HORIZON
CHUNK_MIN = 200_000


class MinuteFullData:
    """The six training cities' minute sources, anchor indices and scalers."""

    def __init__(self, cities=tuple(CITIES)):
        self.cities = list(cities)
        self.sources = [CityMinuteSource(c) for c in self.cities]
        self.x_scaler = self.w_scaler = None

    def index(self, split, fraction=1.0, seed=config.SEED):
        """(city index, anchor minute) of every window in a split, in time order per city.
        fraction < 1 keeps the same deterministic random fraction of each city's anchors."""
        cities, anchors = [], []
        for i, src in enumerate(self.sources):
            a = src.splits[split]
            if fraction < 1.0:
                rng = np.random.default_rng([seed, i])
                a = np.sort(rng.choice(a, size=max(1, int(round(fraction * len(a)))), replace=False))
            cities.append(np.full(len(a), i))
            anchors.append(a)
        return np.concatenate(cities), np.concatenate(anchors)

    def fit_scalers(self):
        """Train-only scalers on the training minute series (see module docstring)."""
        x_scaler, w_scaler = StandardScaler(), StandardScaler()
        rows = {"features": 0, "weather": 0}
        for src in self.sources:
            train = src.splits["train"]
            for lo, hi, table, scaler, key in ((train[0] - config.SEQ_LEN + 1, train[-1], src.feats, x_scaler, "features"),
                                               (train[0] + 1, train[-1] + H, src.weather, w_scaler, "weather")):
                for start in range(lo, hi + 1, CHUNK_MIN):
                    minutes = np.arange(start, min(start + CHUNK_MIN, hi + 1))
                    scaler.partial_fit(interpolate(table, minutes))
                    rows[key] += len(minutes)
        self.x_scaler, self.w_scaler, self.scaler_rows = x_scaler, w_scaler, rows
        return rows

    def save_scalers(self, folder):
        folder = Path(folder)
        joblib.dump(self.x_scaler, folder / "feature_scaler.pkl")
        joblib.dump(self.w_scaler, folder / "weather_scaler.pkl")
        with open(folder / "scalers.json", "w", encoding="utf-8") as file:
            json.dump({"fitted_on": "interpolated training minutes only (inputs: first train anchor - 59 .. "
                                    "last train anchor; weather targets: first train anchor + 1 .. last + 5), "
                                    "all six cities, global (not per city)",
                       "rows": self.scaler_rows,
                       "features": config.FEATURES, "feature_mean": self.x_scaler.mean_.tolist(),
                       "feature_scale": self.x_scaler.scale_.tolist(),
                       "weather_targets": config.WEATHER_TARGETS, "weather_mean": self.w_scaler.mean_.tolist(),
                       "weather_scale": self.w_scaler.scale_.tolist()}, file, indent=2)

    def load_scalers(self, folder):
        self.x_scaler = joblib.load(Path(folder) / "feature_scaler.pkl")
        self.w_scaler = joblib.load(Path(folder) / "weather_scaler.pkl")


class MinuteBatches(keras.utils.PyDataset):
    """Keras batches built on request. Training batches are shuffled across cities with a
    seeded generator, reshuffled at each epoch end; validation/test keep their order."""

    def __init__(self, data, split, batch_size=128, shuffle=False, fraction=1.0, seed=config.SEED, **kwargs):
        super().__init__(**kwargs)
        self.data, self.split, self.batch_size, self.shuffle = data, split, batch_size, shuffle
        self.city_idx, self.anchors = data.index(split, fraction, seed)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(self.anchors)) if shuffle else np.arange(len(self.anchors))

    def __len__(self):
        return int(np.ceil(len(self.anchors) / self.batch_size))

    def raw_batch(self, i):
        """Unscaled windows of batch i plus city index and anchor minute, in batch order."""
        sel = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        cities, anchors = self.city_idx[sel], self.anchors[sel]
        out = {"X": np.empty((len(sel), config.SEQ_LEN, N_FEAT)), "weather": np.empty((len(sel), H, N_WEATHER)),
               "epl": np.empty((len(sel), H)), "epl_now": np.empty(len(sel)), "epl_prev": np.empty(len(sel))}
        for c in np.unique(cities):
            rows = np.flatnonzero(cities == c)
            w = self.data.sources[c].windows(anchors[rows])
            for key in out:
                out[key][rows] = w[key]
        out["city"], out["anchor"] = cities, anchors
        return out

    def __getitem__(self, i):
        raw = self.raw_batch(i)
        X = self.data.x_scaler.transform(raw["X"].reshape(-1, N_FEAT)).reshape(raw["X"].shape)
        W = self.data.w_scaler.transform(raw["weather"].reshape(-1, N_WEATHER)).reshape(raw["weather"].shape)
        return X.astype(np.float32), {"weather": W.astype(np.float32), "epl": raw["epl"].astype(np.float32)}

    def on_epoch_end(self):
        if self.shuffle:
            self.order = self.rng.permutation(len(self.anchors))


def main():
    data = MinuteFullData()
    for split in ("train", "val", "test"):
        c, a = data.index(split)
        print(split, len(a), "windows;", np.bincount(c).tolist(), "per city")
    bytes_batch = 128 * (config.SEQ_LEN * N_FEAT + H * N_WEATHER + H) * 4
    print(f"one batch of 128 (float32): {bytes_batch / 1e6:.2f} MB; hourly tables in memory: "
          f"{sum(s.feats.nbytes + s.weather.nbytes + s.epl.nbytes for s in data.sources) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
