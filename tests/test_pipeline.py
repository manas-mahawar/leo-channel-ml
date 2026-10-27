"""Lightweight tests for the replication pipeline (standard-library unittest).

Pure-function tests always run. Tests that need the local datasets, physics
tables or saved results are skipped when those files are absent (for
example in a fresh clone without data).

Usage (from the repository root):
    .venv\\Scripts\\python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "evaluation"))

import config
from data_pipeline.build_minute_dataset import interpolate
from data_pipeline.build_paper_dataset import GAP_H, split_anchors
from physics.build_attenuation import cloud_attenuation, load_lwp_proxy
import evaluate_paper_model as ep
import run_queue

HAVE_DATA = (config.DATASET_DIR / "meta_test.csv").exists()
HAVE_PHYSICS = (config.PHYSICS_DIR / "Athens_physics.csv").exists()
HAVE_RESULTS = (config.RESULTS_DIR / "seed_runs.csv").exists()


class TestMetrics(unittest.TestCase):
    def test_rmse(self):
        self.assertAlmostEqual(ep.rmse(np.array([3.0, -4.0])), np.sqrt(12.5))

    def test_uncertainty_metrics(self):
        y = np.array([0.0, 0.0, 0.0, 0.0])
        mu = np.array([0.0, 1.0, 2.0, 3.0])
        sigma = np.ones(4)
        m = ep.uncertainty_metrics(y, mu, sigma)
        self.assertAlmostEqual(m["PICP_95_pct"], 50.0)          # |z| = 0, 1 inside; 2, 3 outside 1.96
        self.assertAlmostEqual(m["NLL_eq10"], np.mean(0.5 * mu ** 2))   # log(1) = 0
        self.assertAlmostEqual(m["sharpness_dB"], 1.0)


class TestSplits(unittest.TestCase):
    def test_split_is_chronological_with_purge_gap(self):
        anchors = np.arange(0, 10000, 7)
        splits, _ = split_anchors(anchors)
        self.assertGreaterEqual(splits["val"].min() - splits["train"].max(), GAP_H)
        self.assertGreaterEqual(splits["test"].min() - splits["val"].max(), GAP_H)
        self.assertEqual(GAP_H, config.SEQ_LEN + config.HORIZON)

    @unittest.skipUnless(HAVE_DATA, "dataset not built")
    def test_saved_splits_share_no_hour(self):
        meta = {s: pd.read_csv(config.DATASET_DIR / f"meta_{s}.csv", parse_dates=["anchor_time"])
                for s in ("train", "val", "test")}
        h = pd.Timedelta(hours=1)
        for city in meta["train"].city.unique():
            tr, va, te = (meta[s][meta[s].city == city].anchor_time for s in ("train", "val", "test"))
            # last target hour of one split < first input hour of the next
            self.assertLess(tr.max() + config.HORIZON * h, va.min() - (config.SEQ_LEN - 1) * h, city)
            self.assertLess(va.max() + config.HORIZON * h, te.min() - (config.SEQ_LEN - 1) * h, city)

    @unittest.skipUnless(HAVE_DATA, "dataset not built")
    def test_weather_scaler_fitted_on_training_targets(self):
        w = np.load(config.DATASET_DIR / "weather_train.npy").reshape(-1, len(config.WEATHER_TARGETS))
        np.testing.assert_allclose(w.mean(0), 0, atol=1e-3)
        np.testing.assert_allclose(w.std(0), 1, atol=1e-3)


class TestMinuteInterpolation(unittest.TestCase):
    def test_linear_extrapolation_exact_inside_an_hour(self):
        """Why linear extrapolation wins on the 1-minute data: within an hour
        the interpolated series is a straight line."""
        hourly = np.array([[1.0], [4.0], [2.0]])
        m = np.array([65.0])                                 # anchor at minute 5 of hour 1
        now, prev = interpolate(hourly, m)[:, 0], interpolate(hourly, m - 1)[:, 0]
        future = interpolate(hourly, m[:, None] + np.arange(1, 6))[..., 0]
        extrap = now[:, None] + (now - prev)[:, None] * np.arange(1, 6)
        np.testing.assert_allclose(extrap, future, atol=1e-12)

    def test_extrapolation_breaks_at_hour_boundary(self):
        hourly = np.array([[1.0], [4.0], [2.0]])
        m = np.array([60.0])                                 # offset 0: previous minute is in the last hour
        now, prev = interpolate(hourly, m)[:, 0], interpolate(hourly, m - 1)[:, 0]
        future = interpolate(hourly, m[:, None] + np.arange(1, 6))[..., 0]
        extrap = now[:, None] + (now - prev)[:, None] * np.arange(1, 6)
        self.assertGreater(np.abs(extrap - future).max(), 0.1)


class TestUnseenCityWeights(unittest.TestCase):
    def test_haversine_and_weights(self):
        import evaluate_unseen_cities as eu
        self.assertAlmostEqual(eu.haversine_km(52.52, 13.405, 52.2297, 21.0122), 517, delta=2)
        nearest, dist, w = eu.neighbour_weights(52.52, 13.405, 500.0)
        self.assertEqual(len(nearest), config.K_NEIGHBOURS)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertTrue(np.all(np.diff(dist) >= 0) and np.all(np.diff(w) <= 0))


class TestRunLocations(unittest.TestCase):
    def test_seed_runs_never_use_the_original_folder(self):
        self.assertEqual(run_queue.run_dir("multitask", config.SEED), ROOT / "models" / "paper" / "multitask")
        self.assertEqual(run_queue.run_dir("multitask", 1),
                         ROOT / "models" / "paper" / "seeds" / "multitask_seed1")
        self.assertEqual(run_queue.DEFAULT_SEED, config.SEED)


class TestPhysicsLabels(unittest.TestCase):
    @unittest.skipUnless(HAVE_PHYSICS, "physics tables not built")
    def test_committed_lwp_coefficients_reproduce_cloud_labels(self):
        phys = pd.read_csv(config.PHYSICS_DIR / "Athens_physics.csv", nrows=2000)
        cloud = cloud_attenuation(phys["relative_humidity_percent"].to_numpy(),
                                  phys["precipitation_mm"].to_numpy(), load_lwp_proxy())
        np.testing.assert_allclose(cloud["cloud_attenuation_dB"], phys["cloud_attenuation_dB"], atol=1e-9)

    @unittest.skipUnless(HAVE_PHYSICS, "physics tables not built")
    def test_epl_is_sum_of_components(self):
        phys = pd.read_csv(config.PHYSICS_DIR / "Bergen_physics.csv", nrows=5000)
        total = phys[["gas_attenuation_dB", "cloud_attenuation_dB", "rain_attenuation_dB"]].sum(axis=1)
        np.testing.assert_allclose(total, phys[config.EPL_COLUMN], atol=1e-9)
        self.assertTrue((phys[["gas_attenuation_dB", "cloud_attenuation_dB", "rain_attenuation_dB"]] >= 0).all().all())


class TestSeedSummary(unittest.TestCase):
    @unittest.skipUnless(HAVE_RESULTS, "results not generated")
    def test_seed_summary_matches_seed_level_results(self):
        runs = pd.read_csv(config.RESULTS_DIR / "seed_runs.csv")
        runs = runs[runs.status == "complete"]
        summary = pd.read_csv(config.RESULTS_DIR / "seed_summary.csv").set_index("variant")
        for variant, group in runs.groupby("variant"):
            if group.EPL_RMSE_dB.notna().sum() > 1:
                self.assertAlmostEqual(group.EPL_RMSE_dB.mean(), summary.loc[variant, "EPL_RMSE_dB_mean"], places=9)
                self.assertAlmostEqual(group.EPL_RMSE_dB.std(ddof=1), summary.loc[variant, "EPL_RMSE_dB_std"], places=9)


class TestPaperConsistencyChecks(unittest.TestCase):
    """Arithmetic on the paper's published values (replication_audit_checks.py)."""

    def test_table7_overall_not_reproducible_from_city_values(self):
        import replication_audit_checks as rc
        t7 = rc.paper_checks()["table7"]
        self.assertAlmostEqual(t7["mean_of_city_averages"], 0.0362, places=4)
        self.assertAlmostEqual(t7["rms_of_city_averages_equal_weights"], 0.0629, places=4)
        self.assertNotAlmostEqual(t7["mean_of_city_averages"], rc.PAPER_T7_OVERALL, places=3)

    def test_table8_average_comes_from_figure5_values(self):
        import replication_audit_checks as rc
        t8 = rc.paper_checks()["table8"]
        self.assertEqual(round(t8["figure5_average_of_bars"], 3), 0.213)
        self.assertNotEqual(round(t8["mean_of_listed_abs_errors"], 3), 0.213)

    def test_inferred_figure6_grouping_reproduces_published_bars(self):
        import replication_audit_checks as rc
        fig6 = rc.paper_checks()["figure6"]
        for region, value in fig6["reported"].items():
            self.assertAlmostEqual(fig6["recomputed_from_fig5_with_inferred_groups"][region], value, delta=0.001)

    @unittest.skipUnless((ROOT / "results" / "replication_audit" / "audit_checks.json").exists(),
                         "replication_audit_checks.py not run")
    def test_minute_protocol_matches_fig4_sample_count(self):
        import json
        r = json.load(open(ROOT / "results" / "replication_audit" / "audit_checks.json", encoding="utf-8"))
        n = r["full_series_minute_protocol"]["total_test_windows_six_cities"]
        self.assertTrue(460_000 <= n <= 480_000, n)


class TestRainLabelReference(unittest.TestCase):
    """Independent ITU-R references (evaluation/rain_label_diagnostic.py) vs stored labels."""

    @unittest.skipUnless(HAVE_PHYSICS, "physics tables not built")
    def test_independent_p618_reproduces_stored_rain_attenuation(self):
        import rain_label_diagnostic as rd
        coef = rd.p838_combined(config.FREQUENCY_GHZ, config.ELEVATION_DEG, config.POLARIZATION_TILT_DEG)
        phys = pd.read_csv(config.PHYSICS_DIR / "Vigo_physics.csv")
        wet = phys[phys.rain_mm > 0].iloc[::500]                     # spread over all rain rates
        lat = rd.CITIES["Vigo"]["latitude"]
        for _, r in wet.iterrows():
            ref = rd.p618_reference(r.rain_mm, lat, r.rain_height_km, r.station_height_km, coef["k"], coef["alpha"])
            self.assertAlmostEqual(ref["A"], r.rain_attenuation_dB, places=9)
            self.assertAlmostEqual(ref["r001"], r.r001, places=9)
            self.assertAlmostEqual(ref["nu"], r.nu001, places=9)

    def test_independent_p840_kl(self):
        import rain_label_diagnostic as rd
        from physics.build_attenuation import cloud_attenuation, load_lwp_proxy
        kl_pipeline = cloud_attenuation(np.array([80.0]), np.array([0.0]), load_lwp_proxy())["cloud_kl_dB_km_per_g_m3"][0]
        self.assertAlmostEqual(rd.kl_double_debye(config.FREQUENCY_GHZ, config.CLOUD_KL_TEMP_C), kl_pipeline, places=9)


class TestMinuteFullSource(unittest.TestCase):
    """Lazy full-series minute windows (data_pipeline/minute_full_source.py)."""

    @unittest.skipUnless(HAVE_PHYSICS, "physics tables not built")
    def test_splits_share_no_minute_or_hourly_row_and_use_stride_5(self):
        from data_pipeline.minute_full_source import CityMinuteSource, first_hour_row, last_hour_row
        s = CityMinuteSource("Athens").splits
        self.assertLess(s["train"][-1] + config.HORIZON, s["val"][0] - (config.SEQ_LEN - 1))
        self.assertLess(s["val"][-1] + config.HORIZON, s["test"][0] - (config.SEQ_LEN - 1))
        # interpolation uses rows floor(m/60) and floor(m/60)+1: no row may feed two splits
        self.assertGreater(first_hour_row(s["val"][:1])[0], last_hour_row(s["train"][-1]))
        self.assertGreater(first_hour_row(s["test"][:1])[0], last_hour_row(s["val"][-1]))
        for split in s.values():
            self.assertTrue(np.all(np.diff(split) == 5))

    @unittest.skipUnless((config.DATASET_DIR_MINUTE / "meta_test.csv").exists(), "minute dataset not built")
    def test_windows_equal_existing_minute_dataset(self):
        import joblib
        from data_pipeline.minute_full_source import CityMinuteSource
        src = CityMinuteSource("Bergen")
        meta = pd.read_csv(config.DATASET_DIR_MINUTE / "meta_test.csv", parse_dates=["anchor_time"])
        rows = np.flatnonzero(meta.city == "Bergen")[:25]
        minutes = ((meta.anchor_time[rows] - src.start) / pd.Timedelta(minutes=1)).to_numpy().astype(int)
        w = src.windows(minutes)
        scaler = joblib.load(config.DATASET_DIR_MINUTE / "feature_scaler.pkl")
        X = np.load(config.DATASET_DIR_MINUTE / "X_test.npy", mmap_mode="r")[rows]
        X_raw = scaler.inverse_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        np.testing.assert_allclose(w["X"], X_raw, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(w["epl"], np.load(config.DATASET_DIR_MINUTE / "epl_test.npy")[rows], atol=1e-4)


@unittest.skipUnless(HAVE_PHYSICS, "physics tables not built")
class TestMinuteFullLoader(unittest.TestCase):
    """Lazy batches, train-only scalers and streaming evaluation (minute reconstruction)."""

    @classmethod
    def setUpClass(cls):
        from data_pipeline.minute_full_loader import MinuteFullData
        cls.data = MinuteFullData(cities=("Athens", "Bergen"))
        cls.data.fit_scalers()

    def batches(self, split, **kw):
        from data_pipeline.minute_full_loader import MinuteBatches
        return MinuteBatches(self.data, split, **kw)

    def test_shapes_and_finite_values(self):
        X, y = self.batches("train", batch_size=32, shuffle=True, fraction=0.01)[0]
        self.assertEqual(X.shape, (32, config.SEQ_LEN, len(config.FEATURES)))
        self.assertEqual(y["weather"].shape, (32, config.HORIZON, len(config.WEATHER_TARGETS)))
        self.assertEqual(y["epl"].shape, (32, config.HORIZON))
        self.assertTrue(np.isfinite(X).all() and np.isfinite(y["weather"]).all() and np.isfinite(y["epl"]).all())

    def test_deterministic_shuffle_and_epoch_permutation(self):
        a = self.batches("train", shuffle=True, fraction=0.01, seed=7)
        b = self.batches("train", shuffle=True, fraction=0.01, seed=7)
        c = self.batches("train", shuffle=True, fraction=0.01, seed=8)
        np.testing.assert_array_equal(a.order, b.order)
        self.assertFalse(np.array_equal(a.order, c.order))
        self.assertTrue(np.array_equal(np.sort(a.order), np.arange(len(a.anchors))))   # every anchor once per epoch
        first = a.order.copy()
        a.on_epoch_end()
        self.assertFalse(np.array_equal(first, a.order))
        self.assertTrue(np.array_equal(np.sort(a.order), np.arange(len(a.anchors))))

    def test_validation_and_test_are_not_shuffled(self):
        for split in ("val", "test"):
            b = self.batches(split, fraction=0.01)
            np.testing.assert_array_equal(b.order, np.arange(len(b.anchors)))
            b.on_epoch_end()
            np.testing.assert_array_equal(b.order, np.arange(len(b.anchors)))

    def test_fraction_subset_is_deterministic_and_balanced(self):
        c1, a1 = self.data.index("train", 0.01, seed=42)
        c2, a2 = self.data.index("train", 0.01, seed=42)
        np.testing.assert_array_equal(a1, a2)
        counts = np.bincount(c1)
        self.assertEqual(counts[0], counts[1])

    def test_scalers_use_training_minutes_only(self):
        from data_pipeline.build_minute_dataset import interpolate
        train_rows = sum(s.splits["train"][-1] - (s.splits["train"][0] - config.SEQ_LEN + 1) + 1
                         for s in self.data.sources)
        self.assertEqual(self.data.scaler_rows["features"], train_rows)
        src = self.data.sources[0]
        last_val = src.splits["val"][-1]
        self.assertLess(src.splits["train"][-1], src.splits["val"][0])
        # a scaler that also saw validation minutes would have a different mean
        val_mean = interpolate(src.feats, np.arange(src.splits["val"][0] - 59, last_val + 1)).mean(0)
        self.assertFalse(np.allclose(val_mean, self.data.x_scaler.mean_, atol=1e-6))

    def test_batches_equal_direct_windows(self):
        b = self.batches("test", batch_size=16, fraction=0.01)
        raw = b.raw_batch(2)
        for city in np.unique(raw["city"]):
            rows = raw["city"] == city
            w = self.data.sources[city].windows(raw["anchor"][rows])
            np.testing.assert_array_equal(w["X"], raw["X"][rows])
            np.testing.assert_array_equal(w["epl"], raw["epl"][rows])
        X, _ = b[2]
        np.testing.assert_allclose(self.data.x_scaler.inverse_transform(X.reshape(-1, X.shape[-1])),
                                   raw["X"].reshape(-1, X.shape[-1]), rtol=1e-5, atol=1e-3)

    def test_full_index_is_batchable_without_materialising(self):
        b = self.batches("train", batch_size=128)
        self.assertEqual(len(b), int(np.ceil(len(b.anchors) / 128)))
        self.assertEqual(len(b.anchors), sum(len(s.splits["train"]) for s in self.data.sources))

    def test_streaming_baselines_equal_direct_computation(self):
        sys.path.insert(0, str(ROOT / "evaluation"))
        from evaluate_minute_full import evaluate
        b = self.batches("test", batch_size=64, fraction=0.005)
        result, _ = evaluate(b, None)
        self.assertEqual(result["anchors_scored"], len(b.anchors))
        y, now, prev = [], [], []
        for i in range(len(b)):
            raw = b.raw_batch(i)
            y.append(raw["epl"]); now.append(raw["epl_now"]); prev.append(raw["epl_prev"])
        y, now, prev = np.concatenate(y), np.concatenate(now), np.concatenate(prev)
        pers = np.sqrt(((now[:, None] - y) ** 2).mean())
        lin = np.sqrt(((now[:, None] + (now - prev)[:, None] * np.arange(1, 6) - y) ** 2).mean())
        self.assertAlmostEqual(result["metrics"]["persistence"]["RMSE"], pers, places=9)
        self.assertAlmostEqual(result["metrics"]["linear_extrapolation"]["RMSE"], lin, places=9)


if __name__ == "__main__":
    unittest.main()
