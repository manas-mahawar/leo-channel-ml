"""Leakage tests for the causal protocol causal_hourly_v1 on tiny synthetic data.

Usage (from the repository root):
    .venv\\Scripts\\python -m unittest tests.test_causal -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import config
from data_pipeline.causal_forecast_source import (CausalHourlySource, causal_minute_proxy_baselines,
                                                  fit_feature_scaler, held_hour)

N = 1200


def synthetic_table(seed=0, n=N):
    rng = np.random.default_rng(seed)
    gas = 0.7 + 0.05 * rng.standard_normal(n)
    cloud = np.abs(0.2 + 0.1 * rng.standard_normal(n))
    rain = np.where(rng.random(n) < 0.2, rng.exponential(2.0, n), 0.0)
    return pd.DataFrame({
        "valid_time": pd.date_range("2024-01-01", periods=n, freq="h"),
        "temperature_C": 10 + rng.standard_normal(n), "relative_humidity_percent": 70 + 10 * rng.random(n),
        "pressure_hPa": 1010 + rng.standard_normal(n), "precipitation_mm": np.where(rain > 0, rain / 2, 0.0),
        "cloud_cover_percent": 100 * rng.random(n), "gas_attenuation_dB": gas, "cloud_attenuation_dB": cloud,
        "rain_attenuation_dB": rain, "epl_dB": gas + cloud + rain, "elevation_deg": 30.0})


def source(table):
    return CausalHourlySource(table, "Synthetic", 50.0, 5.0)


def perturb_after(table, row):
    t = table.copy()
    cols = [c for c in t.columns if c not in ("valid_time", "elevation_deg")]
    t.loc[row + 1:, cols] = t.loc[row + 1:, cols] * 3.7 + 11.0
    return t


class TestCausalHourly(unittest.TestCase):
    def setUp(self):
        self.table = synthetic_table()
        self.src = source(self.table)

    def test_future_perturbation_does_not_change_inputs(self):
        origin = int(self.src.splits["val"][3])
        X0 = self.src.windows([origin])["X"]
        X1 = source(perturb_after(self.table, origin)).windows([origin])["X"]
        np.testing.assert_array_equal(X0, X1)

    def test_rolling_sigma_features_are_trailing(self):
        row = 200
        sig = [config.FEATURES.index(f) for f in ("gas_std", "cloud_std", "rain_std")]
        before = self.src.feats[:row + 1, sig]
        after = source(perturb_after(self.table, row)).feats[:row + 1, sig]
        np.testing.assert_array_equal(before, after)

    def test_scaler_ignores_validation_and_test_rows(self):
        s0 = fit_feature_scaler([self.src])
        last_train_row = int(self.src.splits["train"][-1])
        s1 = fit_feature_scaler([source(perturb_after(self.table, last_train_row))])
        np.testing.assert_array_equal(s0.mean_, s1.mean_)
        np.testing.assert_array_equal(s0.scale_, s1.scale_)

    def test_inputs_not_after_origin_and_targets_at_exact_horizons(self):
        o = self.src.splits["test"]
        w = self.src.windows(o)
        self.assertTrue((w["input_times"] <= w["origin_time"][:, None]).all())
        self.assertTrue((w["input_times"][:, -1] == w["origin_time"]).all())
        lead = (w["target_times"] - w["origin_time"][:, None]) / np.timedelta64(1, "h")
        np.testing.assert_array_equal(lead, np.tile(np.arange(1, config.HORIZON + 1), (len(o), 1)))

    def test_splits_do_not_share_input_or_target_rows(self):
        s = self.src.splits
        H, SEQ = config.HORIZON, config.SEQ_LEN
        self.assertGreater(s["val"][0] - SEQ + 1, s["train"][-1] + H)
        self.assertGreater(s["test"][0] - SEQ + 1, s["val"][-1] + H)

    def test_origins_sorted_unique_and_deterministic(self):
        for split in self.src.splits.values():
            self.assertTrue(np.all(np.diff(split) > 0))
        again = source(synthetic_table())
        for k in self.src.splits:
            np.testing.assert_array_equal(self.src.splits[k], again.splits[k])

    def test_missing_past_value_excludes_windows_without_filling(self):
        t = self.table.copy()
        bad = 250
        t.loc[bad, "temperature_C"] = np.nan
        s = source(t)
        all_origins = np.concatenate(list(s.splits.values()))
        self.assertFalse(np.any((all_origins >= bad) & (all_origins <= bad + config.SEQ_LEN - 1)))
        # windows ending before the gap are unaffected; no value was filled in
        self.assertTrue(np.isnan(s.feats[bad, config.FEATURES.index("temperature_C")]))
        early = all_origins[all_origins < bad]
        np.testing.assert_array_equal(s.windows(early)["X"], self.src.windows(early)["X"])

    def test_too_short_series_is_rejected(self):
        with self.assertRaises(ValueError):
            source(synthetic_table(n=300))

    def test_metadata_does_not_claim_minute_measurements(self):
        meta = CausalHourlySource.metadata()
        self.assertEqual(meta["input_frequency"], "1h")
        self.assertEqual(meta["horizon_unit"], "hours")
        self.assertFalse(meta["measured_minute_data"])
        self.assertEqual(meta["interpolation"], "none")

    def test_baselines_use_identical_keys_and_are_deterministic(self):
        import evaluate_causal_baselines as ecb
        r1, keys1, y1, p1 = ecb.evaluate_hourly([self.src])
        r2, keys2, y2, p2 = ecb.evaluate_hourly([source(synthetic_table())])
        self.assertEqual(len({p.shape for p in p1.values()}), 1)          # same (origin, horizon) grid
        for p in p1.values():
            self.assertEqual(p.shape, y1.shape)
        np.testing.assert_array_equal(keys1, keys2)
        for b in r1:
            self.assertEqual(r1[b]["overall"]["RMSE"], r2[b]["overall"]["RMSE"])
        # persistence and linear extrapolation are computable from past values only
        w = self.src.windows(self.src.splits["test"])
        np.testing.assert_array_equal(p1["persistence"][:, 0], w["y_now"])


class TestCausalMinuteProxy(unittest.TestCase):
    def test_minute_inputs_never_use_the_next_hour(self):
        epl = np.array([1.0, 2.0, 4.0, 8.0])
        anchors = np.array([65, 90, 119])                 # inside hour 1, before hour 2 arrives at minute 120
        a = causal_minute_proxy_baselines(epl, anchors)
        epl_changed = epl.copy()
        epl_changed[2:] = [-50.0, 99.0]                   # change hour 2 and later
        b = causal_minute_proxy_baselines(epl_changed, anchors)
        for k in ("persistence", "linear_extrapolation"):
            np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(held_hour(anchors), [1, 1, 1])
        np.testing.assert_array_equal(a["persistence"][:, 0], [2.0, 2.0, 2.0])

    def test_existing_interpolation_uses_the_next_hour(self):
        """Documents the leak of the existing minute source: minute 90 depends on hour 2."""
        from data_pipeline.build_minute_dataset import interpolate
        hourly = np.array([[1.0], [2.0], [4.0]])
        changed = hourly.copy()
        changed[2] = 40.0
        self.assertNotEqual(interpolate(hourly, np.array([90.0]))[0, 0], interpolate(changed, np.array([90.0]))[0, 0])

    def test_held_value_switches_exactly_on_the_hour(self):
        np.testing.assert_array_equal(held_hour([59, 60, 119, 120]), [0, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()
