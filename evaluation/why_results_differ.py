"""Measurements for docs/WHY_RESULTS_DIFFER_FROM_PAPER.md (read-only; no training).

Reads the physics tables, the causal_hourly_v1 test predictions and the baseline
keys; writes new files only to results/why_results_differ/ and refuses to
overwrite them:
  epl_distribution.csv     EPL label distribution per city (full 2020-2024 series
                           and causal test targets), plus component means
  error_budget.json        where the model's squared error comes from
  aggregation.json         the same predictions scored with alternative aggregations
  units_check.json         independent recomputation of the targets in dB
  city_ranking.csv         our per-city values next to the paper's Table 7

Usage:
    python evaluation/why_results_differ.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from data_pipeline.cities import CITIES

H = config.HORIZON
OUT = ROOT / "results" / "why_results_differ"
THRESHOLDS = [2, 5, 6.2, 6.7, 10, 20]
PAPER_TABLE7 = {"Athens": 0.017, "Bergen": 0.148, "Budapest": 0.005, "Chilbolton": 0.005, "Vigo": 0.003, "Warsaw": 0.039}
PCTS = [1, 5, 25, 50, 75, 90, 95, 99, 99.9]


def describe(x):
    x = np.asarray(x, dtype=float).ravel()
    row = {"n": x.size, "mean": x.mean(), "std": x.std(), "min": x.min()}
    row.update({f"p{p}": np.percentile(x, p) for p in PCTS})
    row["max"] = x.max()
    row.update({f"pct_above_{t}dB": 100 * (x > t).mean() for t in THRESHOLDS})
    return row


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwriting.")
    pred = np.load(ROOT / "results" / "causal_hourly_fig1heads_seed42" / "test_predictions.npz")
    base = np.load(ROOT / "results" / "causal_baseline_audit" / "aligned_keys_hourly.npz")
    city, origin = pred["city_index"], pred["origin_hour_since_epoch"]
    y = pred["y"].astype(float)
    forecasts = {"model": pred["epl_mean"].astype(float), "persistence": base["pred_persistence"].astype(float)}

    # 1. Independent recomputation of targets from the physics CSVs (dB), and rain at target hours
    tables, y_csv, y_sum, rain_t, rain_o = {}, np.empty_like(y), np.empty_like(y), np.empty_like(y), np.empty(len(y))
    for c, name in enumerate(CITIES):
        t = pd.read_csv(config.PHYSICS_DIR / f"{name}_physics.csv", parse_dates=["valid_time"])
        tables[name] = t
        idx = pd.Index(t.valid_time.to_numpy().astype("datetime64[h]").astype(np.int64))
        rows = city == c
        pos = idx.get_indexer((origin[rows][:, None] + np.arange(1, H + 1)).ravel()).reshape(-1, H)
        assert (pos >= 0).all()
        y_csv[rows] = t.epl_dB.to_numpy()[pos]
        y_sum[rows] = (t.gas_attenuation_dB + t.cloud_attenuation_dB + t.rain_attenuation_dB).to_numpy()[pos]
        rain_t[rows] = t.rain_attenuation_dB.to_numpy()[pos]
        rain_o[rows] = t.rain_attenuation_dB.to_numpy()[idx.get_indexer(origin[rows])]
    m = forecasts["model"]
    units = {
        "max_abs_diff_stored_y_vs_physics_csv_epl_dB": float(np.abs(y - y_csv).max()),
        "max_abs_diff_epl_vs_gas_plus_cloud_plus_rain": float(np.abs(y_csv - y_sum).max()),
        "max_abs_diff_persistence_vs_csv_epl_at_origin": None,
        "target_mean_dB": float(y.mean()), "prediction_mean_dB": float(m.mean()),
        "target_range_dB": [float(y.min()), float(y.max())], "prediction_range_dB": [float(m.min()), float(m.max())],
        "rmse_recomputed_from_csv_targets_dB": rmse(m - y_csv),
        "epl_head": "linear Dense(10), no output scaler; EPL targets fed to the loss in dB (training/train_causal_hourly.py)",
    }
    pers_csv = np.empty(len(y))
    for c, name in enumerate(CITIES):
        t = tables[name]
        idx = pd.Index(t.valid_time.to_numpy().astype("datetime64[h]").astype(np.int64))
        rows = city == c
        pers_csv[rows] = t.epl_dB.to_numpy()[idx.get_indexer(origin[rows])]
    units["max_abs_diff_persistence_vs_csv_epl_at_origin"] = float(np.abs(base["pred_persistence"][:, 0] - pers_csv).max())

    # 2. EPL distribution: full series per city, and causal test targets per city
    rows = []
    for c, name in enumerate(CITIES):
        t = tables[name]
        rows.append({"city": name, "set": "full 2020-2024 hourly", **describe(t.epl_dB),
                     "gas_mean": t.gas_attenuation_dB.mean(), "cloud_mean": t.cloud_attenuation_dB.mean(),
                     "rain_mean": t.rain_attenuation_dB.mean(), "pct_hours_rain_gt0": 100 * (t.rain_attenuation_dB > 0).mean(),
                     "rain_mm_max": t.rain_mm.max()})
        rows.append({"city": name, "set": "causal test targets", **describe(y[city == c])})
    full = np.concatenate([tables[n].epl_dB.to_numpy() for n in CITIES])
    rows.append({"city": "ALL", "set": "full 2020-2024 hourly", **describe(full)})
    rows.append({"city": "ALL", "set": "causal test targets", **describe(y)})
    OUT.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(OUT / "epl_distribution.csv", index=False, float_format="%.4f")

    # 3. Error budget: share of squared error by target level and rain state
    budget = {}
    for k, p in forecasts.items():
        se = (p - y) ** 2
        b = {"rmse_all": rmse(p - y)}
        for t in (2, 5, 6.2, 6.7, 10):
            hi = y > t
            b[f"share_values_above_{t}dB_pct"] = float(100 * hi.mean())
            b[f"share_sq_error_from_above_{t}dB_pct"] = float(100 * se[hi].sum() / se.sum())
            b[f"rmse_targets_le_{t}dB"] = rmse((p - y)[~hi])
        no_rain = (rain_o == 0) & (rain_t == 0).all(1)
        b["origins_without_rain_at_origin_and_targets_pct"] = float(100 * no_rain.mean())
        b["rmse_no_rain_at_origin_or_targets"] = rmse((p - y)[no_rain])
        b["rmse_target_value_rain_free"] = rmse((p - y)[rain_t == 0])
        b["rmse_target_value_with_rain"] = rmse((p - y)[rain_t > 0])
        top = se.ravel() >= np.percentile(se, 99)
        b["share_sq_error_from_worst_1pct_values"] = float(100 * se.ravel()[top].sum() / se.sum())
        budget[k] = b

    # 4. Alternative aggregations of the same predictions
    agg = {}
    y_range = y.max() - y.min()
    train_std = float(np.concatenate([tables[n].epl_dB.to_numpy()[:30_000] for n in CITIES]).std())
    for k, p in forecasts.items():
        e = p - y
        city_rmse = np.array([rmse(e[city == c]) for c in range(len(CITIES))])
        city_bias = np.array([e[city == c].mean() for c in range(len(CITIES))])
        agg[k] = {
            "pooled_rmse_dB": rmse(e),
            "mean_of_city_rmse_dB": float(city_rmse.mean()),
            "rms_of_city_rmse_dB": float(np.sqrt((city_rmse ** 2).mean())),
            "mse_dB2": float(np.mean(e ** 2)),
            "mae_dB": float(np.abs(e).mean()),
            "median_abs_error_dB": float(np.median(np.abs(e))),
            "abs_overall_mean_bias_dB": float(abs(e.mean())),
            "mean_abs_city_mean_bias_dB (Table 8 style)": float(np.abs(city_bias).mean()),
            "rmse_of_horizon_averaged_forecast_dB": rmse(e.mean(1)),
            "rmse_divided_by_target_range (min-max scaled)": rmse(e) / y_range,
            "rmse_divided_by_approx_train_std (z-scored)": rmse(e) / train_std,
            "per_city_rmse_dB": dict(zip(CITIES, city_rmse.round(4).tolist())),
            "per_city_mean_bias_dB": dict(zip(CITIES, city_bias.round(4).tolist())),
            "per_city_rmse_min_max_scaled": {n: rmse(e[city == c]) / (y[city == c].max() - y[city == c].min())
                                             for c, n in enumerate(CITIES)},
        }
    agg["notes"] = {"target_range_dB": float(y_range), "approx_train_std_dB": train_std,
                    "approx_train_std_basis": "first 30,000 hourly rows per city (about the training period)"}
    t7 = np.array(list(PAPER_TABLE7.values()))
    agg["paper_table7_consistency"] = {
        "reported_overall": 0.026, "mean_of_city_values": float(t7.mean()),
        "rms_of_city_values": float(np.sqrt((t7 ** 2).mean())), "median_of_city_values": float(np.median(t7))}

    # 5. City ranking vs paper Table 7
    rank_rows = []
    for c, n in enumerate(CITIES):
        t = tables[n]
        rank_rows.append({"city": n, "paper_table7_rmse": PAPER_TABLE7[n],
                          "model_rmse": agg["model"]["per_city_rmse_dB"][n],
                          "persistence_rmse": agg["persistence"]["per_city_rmse_dB"][n],
                          "epl_std_full": t.epl_dB.std(), "pct_hours_rain_gt0": 100 * (t.rain_attenuation_dB > 0).mean(),
                          "hourly_change_rms": float(np.sqrt(np.mean(np.diff(t.epl_dB.to_numpy()) ** 2)))})
    rank = pd.DataFrame(rank_rows)
    rank.to_csv(OUT / "city_ranking.csv", index=False, float_format="%.4f")
    rank_corr = {col: spearman(rank.paper_table7_rmse, rank[col]) for col in rank.columns[2:]}

    for name, obj in (("error_budget.json", budget), ("aggregation.json", agg),
                      ("units_check.json", {**units, "spearman_vs_paper_table7": rank_corr})):
        with open(OUT / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    print(json.dumps({"units": units, "rank_corr": rank_corr}, indent=1))
    print(json.dumps(budget, indent=1))
    print(json.dumps(agg, indent=1))
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
