"""Evaluate a causal_hourly_v1 model against the baselines on identical test keys.

Reads results/<name>/test_predictions.npz (written once by
training/train_causal_hourly.py after training) and the baseline predictions in
results/causal_baseline_audit/aligned_keys_hourly.npz; refuses to continue
unless the (city, origin) keys and targets are identical.

Writes (new files only) to results/<name>/:
  metrics_test.json      model and baselines: pooled, per horizon, per city, by
                         precipitation class at the target hour; uncertainty calibration
  training_curves.png/pdf

Usage:
    python evaluation/evaluate_causal_model.py [causal_hourly_fig1heads_seed42]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from data_pipeline.cities import CITIES

H = config.HORIZON
Z = {50: 0.674, 80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}


def metrics(err, mask=None):
    if mask is None:
        mask = np.ones_like(err, dtype=bool)
    n = mask.sum(0)
    se, ae = np.where(mask, err ** 2, 0), np.where(mask, np.abs(err), 0)
    return {"n_values": int(n.sum()), "RMSE": float(np.sqrt(se.sum() / max(n.sum(), 1))),
            "MAE": float(ae.sum() / max(n.sum(), 1)),
            "RMSE_by_step": np.sqrt(se.sum(0) / np.maximum(n, 1)).tolist(),
            "MAE_by_step": (ae.sum(0) / np.maximum(n, 1)).tolist()}


def target_precip(city_index, origin_hours):
    """Observed precipitation at each target hour (outcome-based grouping, diagnostics only)."""
    out = np.empty((len(origin_hours), H))
    for c, city in enumerate(CITIES):
        p = pd.read_csv(config.PHYSICS_DIR / f"{city}_physics.csv", usecols=["valid_time", "precipitation_mm"],
                        parse_dates=["valid_time"])
        series = pd.Series(p.precipitation_mm.to_numpy(), index=p.valid_time.to_numpy().astype("datetime64[h]").astype(np.int64))
        rows = city_index == c
        out[rows] = series.reindex((origin_hours[rows][:, None] + np.arange(1, H + 1)).ravel()).to_numpy().reshape(-1, H)
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "causal_hourly_fig1heads_seed42"
    folder = ROOT / "results" / name
    out = folder / "metrics_test.json"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting.")
    pred = np.load(folder / "test_predictions.npz")
    base = np.load(ROOT / "results" / "causal_baseline_audit" / "aligned_keys_hourly.npz")
    same = (np.array_equal(pred["city_index"], base["city_index"]) and
            np.array_equal(pred["origin_hour_since_epoch"], base["origin_hour_since_epoch"]) and
            np.allclose(pred["y"], base["y"], atol=1e-5))
    if not same:
        raise SystemExit("Model and baseline test keys or targets differ; stopping.")

    y = pred["y"].astype(float)
    city = pred["city_index"]
    precip = target_precip(city, pred["origin_hour_since_epoch"])
    classes = {"dry (no precipitation)": precip == 0, "light (0-2.5 mm/h)": (precip > 0) & (precip < 2.5),
               "heavy (>= 2.5 mm/h)": precip >= 2.5}
    forecasts = {"model": pred["epl_mean"].astype(float),
                 "persistence": base["pred_persistence"].astype(float),
                 "recent_mean_24h": base["pred_recent_mean_24h"].astype(float),
                 "linear_extrapolation": base["pred_linear_extrapolation"].astype(float)}
    result = {"test_values": int(y.size), "test_origins": int(len(y)),
              "keys_identical_to_baselines": True,
              "forecasts": {k: {"overall": metrics(p - y),
                                "by_city": {c: metrics((p - y)[city == i]) for i, c in enumerate(CITIES)},
                                "by_precip_class_at_target_hour": {n: metrics(p - y, m) for n, m in classes.items()}}
                            for k, p in forecasts.items()}}
    pers = result["forecasts"]["persistence"]["overall"]
    mod = result["forecasts"]["model"]["overall"]
    result["model_vs_persistence"] = {
        "RMSE_change_dB": mod["RMSE"] - pers["RMSE"], "RMSE_change_pct": 100 * (mod["RMSE"] - pers["RMSE"]) / pers["RMSE"],
        "MAE_change_dB": mod["MAE"] - pers["MAE"], "MAE_change_pct": 100 * (mod["MAE"] - pers["MAE"]) / pers["MAE"],
        "RMSE_change_by_step_dB": (np.array(mod["RMSE_by_step"]) - np.array(pers["RMSE_by_step"])).tolist()}

    mu, sigma = pred["epl_mean"].astype(float), pred["epl_std"].astype(float)
    err = np.abs(y - mu)
    cal = {"nominal_coverage": {str(k): float(100 * (err <= z * sigma).mean()) for k, z in Z.items()},
           "mean_95_width_dB": float((2 * 1.96 * sigma).mean()),
           "NLL_eq10": float((0.5 * (np.log(sigma ** 2) + ((y - mu) / sigma) ** 2)).mean())}
    groups = dict(classes)
    for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 10), (10, 100)):
        groups[f"true EPL {lo}-{hi} dB"] = (y >= lo) & (y < hi)
    for k in range(H):
        groups[f"horizon +{k + 1} h"] = np.zeros_like(y, dtype=bool)
        groups[f"horizon +{k + 1} h"][:, k] = True
    cal["coverage_95_by_group"] = {g: {"n": int(m.sum()), "coverage_pct": float(100 * (err[m] <= 1.96 * sigma[m]).mean()),
                                       "mean_width_dB": float((2 * 1.96 * sigma[m]).mean())}
                                   for g, m in groups.items() if m.sum() > 0}
    result["uncertainty"] = cal

    w = pred["weather_scaled"].astype(float) - pred["weather_true_scaled"].astype(float)
    result["weather_RMSE_standardised"] = float(np.sqrt((w ** 2).mean()))
    with open(out, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    log = pd.read_csv(folder / "training_log.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, col, title in ((axes[0], "loss", "Total loss"), (axes[1], "epl_loss", "EPL loss (Gaussian NLL)")):
        ax.plot(log.epoch + 1, log[col], marker="o", ms=3, label="training", color="#2a78d6")
        ax.plot(log.epoch + 1, log[f"val_{col}"], marker="o", ms=3, label="validation", color="#eb6834")
        ax.axvline(int(log.epoch[log.val_loss.idxmin()]) + 1, color="#888888", ls="--", lw=1)
        ax.set_xlabel("Epoch"); ax.set_title(title, fontsize=9); ax.grid(color="#e5e5e5")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{name}: training curves (dashed: best validation epoch)", fontsize=9)
    fig.tight_layout()
    fig.savefig(folder / "training_curves.png", dpi=200)
    fig.savefig(folder / "training_curves.pdf")
    plt.close(fig)

    for k, r in result["forecasts"].items():
        o = r["overall"]
        print(f"{k:22s} RMSE {o['RMSE']:.4f} MAE {o['MAE']:.4f} by hour {np.round(o['RMSE_by_step'], 3).tolist()}")
    print("vs persistence:", {k: round(v, 4) if not isinstance(v, list) else np.round(v, 3).tolist()
                              for k, v in result["model_vs_persistence"].items()})
    print("coverage:", {k: round(v, 1) for k, v in cal["nominal_coverage"].items()})


if __name__ == "__main__":
    main()
