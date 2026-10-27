"""Evaluate every trained variant on its held-out test windows.

Metrics (paper Sect. 2, 4.1): RMSE / MAE per city and forecast step
(Table 7), PICP of the 95% interval mu +- 1.96 sigma, NLL (Eq. 10) and
Sharpness (mean predicted sigma); weather RMSE in standardised and physical
units. Each variant is scored against the test set of the dataset it was
trained on (models/paper/<variant>/run_config.json -> "dataset").

Baselines, computed from the same test windows:
  hourly datasets:  persistence (EPL at t) and climatology (city x month x
                    hour mean over the training period);
  minute dataset:   persistence and linear extrapolation of the last minute.

Usage:
    python evaluation/evaluate_paper_model.py
Outputs (results/paper/): metrics_test.json, table7_rmse_<name>.csv,
test_predictions_<variant>.csv, test_weather_predictions_<variant>.csv
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from data_pipeline.cities import CITIES

MODELS_DIR = ROOT / "models" / "paper"
OUT = config.RESULTS_DIR
H = config.HORIZON
DATASETS = {"paper": config.DATASET_DIR, "paper_msl": config.DATASET_DIR_MSL,
            "paper_minute": config.DATASET_DIR_MINUTE}
VARIANT_DATASET = {"multitask_msl": "paper_msl", "minute_multitask": "paper_minute"}


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def uncertainty_metrics(y, mu, sigma):
    z = (y - mu) / sigma
    return {
        "PICP_95_pct": float(100 * np.mean(np.abs(z) <= 1.96)),
        "NLL_eq10": float(np.mean(0.5 * (np.log(sigma**2) + z**2))),
        "NLL_with_const": float(np.mean(0.5 * (np.log(2 * np.pi * sigma**2) + z**2))),
        "sharpness_dB": float(np.mean(sigma)),
    }


def load_test(name):
    d = DATASETS[name]
    data = {
        "meta": pd.read_csv(d / "meta_test.csv", parse_dates=["anchor_time"]),
        "y": np.load(d / "epl_test.npy").astype(float),
        "epl_now": np.load(d / "epl_now_test.npy").astype(float),
        "w_true": np.load(d / "weather_test.npy"),
        "w_raw": np.load(d / "weather_raw_test.npy"),
        "w_scaler": joblib.load(d / "weather_scaler.pkl"),
        "x_scaler": joblib.load(d / "feature_scaler.pkl"),
        "x_last": np.load(d / "X_test.npy", mmap_mode="r")[:, -1, :],
    }
    if name == "paper_minute":
        data["epl_prev"] = np.load(d / "epl_prev_test.npy").astype(float)
    return data


def climatology_baseline(meta, dataset_dir):
    """Mean EPL per (city, month, hour) over each city's training period."""
    pred = np.zeros((len(meta), H))
    train_meta = pd.read_csv(dataset_dir / "meta_train.csv", parse_dates=["anchor_time"])
    for city in CITIES:
        phys = pd.read_csv(config.PHYSICS_DIR / f"{city}_physics.csv", parse_dates=["valid_time"])
        end = train_meta.loc[train_meta.city == city, "anchor_time"].max() + pd.Timedelta(hours=H)
        hist = phys[phys.valid_time <= end]
        table = hist.groupby([hist.valid_time.dt.month, hist.valid_time.dt.hour])[config.EPL_COLUMN].mean()
        rows = np.flatnonzero(meta.city == city)
        for h in range(H):
            t = meta.anchor_time.iloc[rows] + pd.Timedelta(hours=h + 1)
            pred[rows, h] = table.loc[list(zip(t.dt.month, t.dt.hour))].to_numpy()
    return pred


def baselines(name, data):
    persistence = np.repeat(data["epl_now"][:, None], H, axis=1)
    if name == "paper_minute":
        slope = data["epl_now"] - data["epl_prev"]
        linear = data["epl_now"][:, None] + slope[:, None] * np.arange(1, H + 1)
        return {"persistence": persistence, "linear_extrapolation": linear}
    return {"persistence": persistence, "climatology": climatology_baseline(data["meta"], DATASETS[name])}


def per_city_table(meta, y, pred):
    rows = []
    for city in list(CITIES) + ["Overall"]:
        m = np.ones(len(meta), bool) if city == "Overall" else (meta.city == city).to_numpy()
        e = pred[m] - y[m]
        row = {"city": city}
        row.update({f"step_{h + 1}": rmse(e[:, h]) for h in range(H)})
        row["average_RMSE"] = rmse(e)
        row["MAE"] = float(np.mean(np.abs(e)))
        rows.append(row)
    return pd.DataFrame(rows)


def condition_masks(weather_raw):
    precip = weather_raw[:, :, config.WEATHER_TARGETS.index("precipitation_mm")]
    cloud = weather_raw[:, :, config.WEATHER_TARGETS.index("cloud_cover_percent")]
    return {
        "clear (no precip, cloud<=20%)": (precip == 0) & (cloud <= 20),
        "cloudy (no precip, cloud>20%)": (precip == 0) & (cloud > 20),
        "light precip (0-2.5 mm/h)": (precip > 0) & (precip < 2.5),
        "moderate+ precip (>=2.5 mm/h)": precip >= 2.5,
    }


def weather_metrics(pred_scaled, data):
    true_raw = data["w_raw"]
    pred_raw = data["w_scaler"].inverse_transform(
        pred_scaled.reshape(-1, len(config.WEATHER_TARGETS))).reshape(true_raw.shape)
    out = {"RMSE_standardised": rmse(pred_scaled - data["w_true"])}
    for i, name in enumerate(config.WEATHER_TARGETS):
        out[f"RMSE_{name}"] = rmse(pred_raw[:, :, i] - true_raw[:, :, i])
        out[f"RMSE_{name}_by_step"] = [rmse(pred_raw[:, h, i] - true_raw[:, h, i]) for h in range(H)]
    return out, pred_raw


def weather_persistence(data):
    w_idx = [config.FEATURES.index(c) for c in config.WEATHER_TARGETS]
    last_raw = data["x_scaler"].inverse_transform(np.asarray(data["x_last"]))[:, w_idx]
    scaled = data["w_scaler"].transform(last_raw)[:, None, :].repeat(H, axis=1)
    return weather_metrics(scaled, data)[0]


def save_predictions(variant, data, mu, sigma, base):
    meta, n = data["meta"], len(data["meta"])
    table = pd.DataFrame({
        "city": np.repeat(meta.city.to_numpy(), H),
        "anchor_time": np.repeat(meta.anchor_time.to_numpy(), H),
        "step": np.tile(np.arange(1, H + 1), n),
        "true_epl_dB": data["y"].ravel(),
        "pred_mean_dB": mu.ravel(),
        "pred_std_dB": sigma.ravel(),
    })
    for name, pred in base.items():
        table[f"{name}_dB"] = pred.ravel()
    table.to_csv(OUT / f"test_predictions_{variant}.csv", index=False)


def save_weather_predictions(variant, data, pred_raw):
    meta, n = data["meta"], len(data["meta"])
    table = pd.DataFrame({"city": np.repeat(meta.city.to_numpy(), H),
                          "anchor_time": np.repeat(meta.anchor_time.to_numpy(), H),
                          "step": np.tile(np.arange(1, H + 1), n)})
    for i, name in enumerate(config.WEATHER_TARGETS):
        table[f"true_{name}"] = data["w_raw"][:, :, i].ravel()
        table[f"pred_{name}"] = pred_raw[:, :, i].ravel()
    table.to_csv(OUT / f"test_weather_predictions_{variant}.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    variants = sorted(p.name for p in MODELS_DIR.iterdir() if (p / "test_predictions.npz").exists())
    needed = sorted({VARIANT_DATASET.get(v, "paper") for v in variants})
    results = {"datasets": {}, "variants": {}}
    data_by_name, base_by_name = {}, {}

    for name in needed:
        data = load_test(name)
        base = baselines(name, data)
        data_by_name[name], base_by_name[name] = data, base
        entry = {"test_windows": int(len(data["y"])), "baselines": {}}
        for bname, pred in base.items():
            entry["baselines"][bname] = {"RMSE": rmse(pred - data["y"]),
                                         "MAE": float(np.mean(np.abs(pred - data["y"]))),
                                         "RMSE_by_step": [rmse(pred[:, h] - data["y"][:, h]) for h in range(H)]}
            per_city_table(data["meta"], data["y"], pred).to_csv(
                OUT / f"table7_rmse_{name}_{bname}.csv", index=False)
        entry["baselines"]["weather_persistence"] = weather_persistence(data)
        results["datasets"][name] = entry

    for variant in variants:
        name = VARIANT_DATASET.get(variant, "paper")
        data, base = data_by_name[name], base_by_name[name]
        y = data["y"]
        pred = np.load(MODELS_DIR / variant / "test_predictions.npz")
        run = json.load(open(MODELS_DIR / variant / "run_config.json", encoding="utf-8"))
        res = {"dataset": name, "epochs_run": run["epochs_run"], "best_epoch": run["best_epoch"],
               "train_hours": run["train_hours"], "features": run.get("features")}
        if "epl_mean" in pred:
            mu, sigma = pred["epl_mean"].astype(float), pred["epl_std"].astype(float)
            per_city_table(data["meta"], y, mu).to_csv(OUT / f"table7_rmse_{variant}.csv", index=False)
            res.update({"RMSE": rmse(mu - y), "MAE": float(np.mean(np.abs(mu - y))),
                        "RMSE_by_step": [rmse(mu[:, h] - y[:, h]) for h in range(H)]})
            res.update(uncertainty_metrics(y, mu, sigma))
            res["per_city_uncertainty"] = {
                c: uncertainty_metrics(y[m], mu[m], sigma[m])
                for c in CITIES for m in [(data["meta"].city == c).to_numpy()]}
            res["by_condition"] = {
                k: {"share_pct": float(100 * m.mean()), "RMSE": rmse((mu - y)[m]),
                    "MAE": float(np.mean(np.abs(mu - y)[m])),
                    "persistence_RMSE": rmse((base["persistence"] - y)[m])}
                for k, m in condition_masks(data["w_raw"]).items() if m.any()}
            save_predictions(variant, data, mu, sigma, base)
        if "weather_scaled" in pred:
            res["weather"], pred_raw = weather_metrics(pred["weather_scaled"], data)
            save_weather_predictions(variant, data, pred_raw)
        results["variants"][variant] = res

    with open(OUT / "metrics_test.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    for variant, r in results["variants"].items():
        line = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                if k in ("dataset", "best_epoch", "RMSE", "MAE", "PICP_95_pct", "NLL_eq10", "sharpness_dB")}
        if "weather" in r:
            line["weather_RMSE_std"] = round(r["weather"]["RMSE_standardised"], 4)
        print(variant, line)
    for name, entry in results["datasets"].items():
        print(name, {b: round(v["RMSE"], 4) for b, v in entry["baselines"].items() if "RMSE" in v})
    print(f"Saved metrics and prediction tables to {OUT}")


if __name__ == "__main__":
    main()
