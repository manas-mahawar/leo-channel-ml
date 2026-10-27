"""Strictly causal forecast windows (protocol causal_hourly_v1) and a causal minute proxy.

causal_hourly_v1 (reference protocol, see docs/CAUSAL_FORECASTING_INVESTIGATION.md):
  forecast origin t = an hourly timestamp; inputs = the 14 features at hours
  t-59..t (all computed from data with timestamps <= t); targets = EPL at
  t+1..t+5 hours. Every hourly origin with a complete window is used (no
  stratified subsampling); chronological 70/15/15 split per city; windows whose
  input or target rows contain a missing value are excluded, never filled.

Time availability (retrospective reanalysis, not operational data):
  * weather at hour t is the ERA5 value valid at t (precipitation and rain are
    sums over the preceding hour, ending at t);
  * gas/cloud/rain attenuation at t are functions of the weather at t only;
  * sigma features are trailing 24-hour rolling standard deviations (rows
    t-23..t; pandas rolling windows are right-aligned);
  * latitude, longitude and elevation are static.
ERA5 is published with a delay of days, so this is causal *retrospective*
evaluation, not operational forecasting.

Causal minute proxy (only to quantify the interpolation leak of
data_pipeline/minute_full_source.py): at minute m, the latest hourly
observation is the one at hour floor(m/60) (zero-order hold). No value from
hour floor(m/60)+1 is used. Held values are hourly reanalysis values, not
measured minute data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from data_pipeline.build_paper_dataset import add_features

PROTOCOL = "causal_hourly_v1"
SEQ, H = config.SEQ_LEN, config.HORIZON
FRACTIONS = (0.70, 0.15)


def window_complete(ok, length):
    """complete[i] is True when ok[i-length+1..i] are all True (False for the first rows)."""
    bad = np.concatenate([[0], np.cumsum(~ok)])
    out = np.zeros(len(ok), dtype=bool)
    out[length - 1:] = (bad[length:] - bad[:-length]) == 0
    return out


class CausalHourlySource:
    """Hourly forecast origins of one city under causal_hourly_v1."""

    def __init__(self, table, city, lat, lon):
        times = pd.to_datetime(table["valid_time"]).reset_index(drop=True)
        if not (times.is_monotonic_increasing and times.is_unique):
            raise ValueError(f"{city}: timestamps must be sorted and unique.")
        if not (times.diff().dropna() == pd.Timedelta(hours=1)).all():
            raise ValueError(f"{city}: timestamps must be a regular hourly series.")
        df = add_features(table.reset_index(drop=True), lat, lon)
        self.city, self.times = city, times
        self.feats = df[config.FEATURES].to_numpy(np.float64)
        self.epl = df[config.EPL_COLUMN].to_numpy(np.float64)
        self.precip = df["precipitation_mm"].to_numpy(np.float64)

        n = len(df)
        input_ok = window_complete(np.isfinite(self.feats).all(axis=1), SEQ)        # rows o-59..o
        epl_ok = np.isfinite(self.epl)
        target_ok = np.zeros(n, dtype=bool)
        target_ok[:n - H] = window_complete(epl_ok, H)[H:]                           # rows o+1..o+5
        prev_ok = np.concatenate([[False], epl_ok[:-1]]) & epl_ok                    # y(t-1), y(t) for baselines
        candidates = np.arange(n)
        usable = input_ok & target_ok & prev_ok
        self.excluded = {"no_complete_history_or_missing_input": int((~input_ok[SEQ - 1:n - H]).sum()),
                         "missing_target_or_baseline_value": int((input_ok & ~(target_ok & prev_ok))[SEQ - 1:n - H].sum())}
        origins = candidates[usable]

        n_train, n_val = int(round(FRACTIONS[0] * len(origins))), int(round(FRACTIONS[1] * len(origins)))
        train, val, test = origins[:n_train], origins[n_train:n_train + n_val], origins[n_train + n_val:]
        # no row may be an input or target of two splits: next split's first input row > previous last target row
        val = val[val - SEQ + 1 > train[-1] + H]
        if len(val) == 0:
            raise ValueError(f"{city}: series too short for a validation split after the {SEQ + H}-hour gap.")
        test = test[test - SEQ + 1 > val[-1] + H]
        if len(test) == 0:
            raise ValueError(f"{city}: series too short for a test split after the {SEQ + H}-hour gap.")
        self.splits = {"train": train, "val": val, "test": test}

    def windows(self, origins):
        origins = np.asarray(origins)
        past = origins[:, None] + np.arange(-SEQ + 1, 1)
        future = origins[:, None] + np.arange(1, H + 1)
        return {"X": self.feats[past], "y": self.epl[future], "y_now": self.epl[origins],
                "y_prev": self.epl[origins - 1], "y_last24_mean": np.array([self.epl[o - 23:o + 1].mean() for o in origins]),
                "origin_time": self.times.to_numpy()[origins], "input_times": self.times.to_numpy()[past],
                "target_times": self.times.to_numpy()[future], "precip_target": self.precip[future]}

    def split_boundaries(self):
        t = self.times
        return {s: {"first_origin": str(t[o[0]]), "last_origin": str(t[o[-1]]), "origins": int(len(o)),
                    "first_input_row_time": str(t[o[0] - SEQ + 1]), "last_target_time": str(t[o[-1] + H])}
                for s, o in self.splits.items()}

    @staticmethod
    def metadata():
        return {"protocol": PROTOCOL, "input_frequency": "1h", "input_steps": SEQ, "target_frequency": "1h",
                "horizon_steps": H, "horizon_unit": "hours", "measured_minute_data": False,
                "source_resolution": "hourly ERA5 reanalysis (Open-Meteo archive) and ITU-R labels computed from it",
                "interpolation": "none", "missing_values": "windows with a missing input or target row are excluded"}


def fit_feature_scaler(sources):
    """Global StandardScaler on the training input rows only (first train origin - 59 .. last train origin)."""
    scaler = StandardScaler()
    for src in sources:
        train = src.splits["train"]
        scaler.partial_fit(src.feats[train[0] - SEQ + 1:train[-1] + 1])
    return scaler


# ---------------------------------------------------------------- causal minute proxy
def held_hour(minutes):
    """Index of the latest hourly observation available at each minute (timestamp <= minute)."""
    return np.floor(np.asarray(minutes) / 60).astype(int)


def causal_minute_proxy_baselines(epl_hourly, anchors, horizon=H):
    """Persistence and linear extrapolation at minute anchors using only hours <= floor(anchor/60).

    persistence:  y_hat(t+k) = y(h)
    linear:       y_hat(t+k) = y(h) + ((t + k - 60 h) / 60) * (y(h) - y(h-1)),  k = 1..horizon minutes
    """
    h = held_hour(anchors)
    now, prev = epl_hourly[h], epl_hourly[h - 1]
    elapsed = (np.asarray(anchors)[:, None] + np.arange(1, horizon + 1)) - 60 * h[:, None]
    return {"persistence": np.repeat(now[:, None], horizon, axis=1),
            "linear_extrapolation": now[:, None] + elapsed / 60.0 * (now - prev)[:, None],
            "staleness_minutes": np.asarray(anchors) - 60 * h}
