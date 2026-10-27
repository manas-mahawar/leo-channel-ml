"""Lazy windows for the full-series 1-minute reconstruction (prototype, not used for training yet).

The paper's Fig. 2 / Fig. 4 suggest windows cut from each city's full hourly
series interpolated to 1 minute, every 5 minutes (about 473,000 test windows,
about 2.2 million training windows). Materialising the training inputs would
need about 7 GB (2.2e6 x 60 x 14 float32). This module keeps only the hourly
tables in memory (about 2 MB per city) and interpolates each window on demand,
using the same `interpolate` function and the same features as
data_pipeline/build_minute_dataset.py, so windows are identical to that
dataset's windows at the same anchor minute.

Split: chronological 70/15/15 of each city's anchors. A minute between hours
h and h+1 is interpolated from both hourly rows, so a gap of SEQ_LEN + HORIZON
minutes is not enough: the first anchors of the next split are dropped until
no hourly row is used by two splits (2026-10-27 fix; the first version
shared two hourly rows at each boundary and was never used for results).

Usage (inspection only):
    python data_pipeline/minute_full_source.py      # prints anchor counts and memory figures
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from data_pipeline.build_minute_dataset import interpolate
from data_pipeline.build_paper_dataset import add_features
from data_pipeline.cities import CITIES

STRIDE_MIN = 5                                   # paper Fig. 2: "Stride: 5 minutes"


def first_hour_row(anchors):
    """Earliest hourly row used to interpolate the input window of each anchor."""
    return np.floor((np.asarray(anchors) - config.SEQ_LEN + 1) / 60).astype(int)


def last_hour_row(anchor):
    """Latest hourly row used to interpolate the targets of an anchor."""
    return int(np.floor((anchor + config.HORIZON) / 60)) + 1


class CityMinuteSource:
    """All 5-minute-stride windows of one city, generated on demand."""

    def __init__(self, city):
        coords = CITIES[city]
        df = pd.read_csv(config.PHYSICS_DIR / f"{city}_physics.csv", parse_dates=["valid_time"])
        df = add_features(df, coords["latitude"], coords["longitude"])
        self.city, self.start = city, df.valid_time.iloc[0]
        self.feats = df[config.FEATURES].to_numpy(np.float64)
        self.weather = df[config.WEATHER_TARGETS].to_numpy(np.float64)
        self.epl = df[[config.EPL_COLUMN]].to_numpy(np.float64)

        # first anchor: all 60 input minutes must have finite sigma features (24-h rolling std)
        first_hour = int(np.flatnonzero(np.isfinite(self.feats).all(axis=1))[0])
        first = first_hour * 60 + config.SEQ_LEN - 1
        last = (len(df) - 1) * 60 - config.HORIZON                 # targets must stay inside the series
        first += (-first) % STRIDE_MIN
        anchors = np.arange(first, last + 1, STRIDE_MIN)
        n_train, n_val = int(round(0.70 * len(anchors))), int(round(0.15 * len(anchors)))
        train, val, test = anchors[:n_train], anchors[n_train:n_train + n_val], anchors[n_train + n_val:]
        val = val[first_hour_row(val) > last_hour_row(train[-1])]
        test = test[first_hour_row(test) > last_hour_row(val[-1])]
        self.splits = {"train": train, "val": val, "test": test}

    def windows(self, anchors):
        """Unscaled inputs (n, 60, 14), weather targets (n, 5, 5), EPL targets (n, 5), EPL now/prev."""
        anchors = np.asarray(anchors)
        past = anchors[:, None] + np.arange(-config.SEQ_LEN + 1, 1)
        future = anchors[:, None] + np.arange(1, config.HORIZON + 1)
        return {"X": interpolate(self.feats, past), "weather": interpolate(self.weather, future),
                "epl": interpolate(self.epl, future)[..., 0], "epl_now": interpolate(self.epl, anchors)[:, 0],
                "epl_prev": interpolate(self.epl, anchors - 1)[:, 0]}

    def anchor_times(self, anchors):
        return self.start + pd.to_timedelta(np.asarray(anchors), unit="min")


def main():
    total = {"train": 0, "val": 0, "test": 0}
    for city in CITIES:
        src = CityMinuteSource(city)
        n = {k: len(v) for k, v in src.splits.items()}
        total = {k: total[k] + n[k] for k in total}
        print(f"{city:10s} {n}  first anchor {src.anchor_times(src.splits['train'][:1])[0]}  "
              f"test from {src.anchor_times(src.splits['test'][:1])[0]}")
    window_bytes = config.SEQ_LEN * len(config.FEATURES) * 4
    print(f"total windows {total}")
    print(f"materialised float32 inputs would need: train {total['train'] * window_bytes / 1e9:.1f} GB, "
          f"test {total['test'] * window_bytes / 1e9:.1f} GB; this source keeps only the hourly tables")


if __name__ == "__main__":
    main()
