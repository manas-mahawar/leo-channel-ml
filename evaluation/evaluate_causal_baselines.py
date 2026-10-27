"""Baselines under the strictly causal protocol causal_hourly_v1 (no model, no training).

1. causal_hourly_v1, test origins of the six training cities:
     persistence          y_hat(t+k) = y(t)
     linear extrapolation y_hat(t+k) = y(t) + k (y(t) - y(t-1)),  k = 1..5 hours
     recent mean          y_hat(t+k) = mean(y(t-23..t))
   All three on the same (city, origin, horizon) keys; RMSE/MAE pooled, per
   horizon, per city, by precipitation class at the target hour.
2. Leak quantification on the minute protocol of data_pipeline/minute_full_source.py:
   the same minute anchors and the same (linearly interpolated) minute targets,
   scored with (a) the existing leaky inputs (interpolated EPL at the anchor,
   which uses hour h+1) and (b) causal inputs (hourly values up to hour h only).
   The minute targets are interpolated estimates, not observations.

Writes only to a new folder (default results/causal_baseline_audit/; refuses to
overwrite): config.json, baseline_metrics.json, aligned_keys_hourly.npz,
aligned_keys_minute.npz.

Usage:
    python evaluation/evaluate_causal_baselines.py [output_folder]
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from data_pipeline.causal_forecast_source import (CausalHourlySource, causal_minute_proxy_baselines,
                                                  fit_feature_scaler)
from data_pipeline.cities import CITIES

H = config.HORIZON
CLASSES = {"dry (no precipitation)": lambda p: p == 0, "light (0-2.5 mm/h)": lambda p: (p > 0) & (p < 2.5),
           "heavy (>= 2.5 mm/h)": lambda p: p >= 2.5}


def metrics(err, mask=None):
    if mask is None:
        mask = np.ones_like(err, dtype=bool)
    n = mask.sum(0)
    se, ae = np.where(mask, err ** 2, 0), np.where(mask, np.abs(err), 0)
    return {"n_values": int(n.sum()), "RMSE": float(np.sqrt(se.sum() / max(n.sum(), 1))),
            "MAE": float(ae.sum() / max(n.sum(), 1)),
            "RMSE_by_step": np.sqrt(se.sum(0) / np.maximum(n, 1)).tolist(),
            "MAE_by_step": (ae.sum(0) / np.maximum(n, 1)).tolist()}


def evaluate_hourly(sources, split="test"):
    """Baselines on identical keys; returns metrics plus the keys and predictions used."""
    keys, y, prec, preds = [], [], [], {"persistence": [], "linear_extrapolation": [], "recent_mean_24h": []}
    for c, src in enumerate(sources):
        o = src.splits[split]
        w = src.windows(o)
        keys.append(np.stack([np.full(len(o), c), w["origin_time"].astype("datetime64[h]").astype(np.int64)], axis=1))
        y.append(w["y"]); prec.append(w["precip_target"])
        k = np.arange(1, H + 1)
        preds["persistence"].append(np.repeat(w["y_now"][:, None], H, axis=1))
        preds["linear_extrapolation"].append(w["y_now"][:, None] + k * (w["y_now"] - w["y_prev"])[:, None])
        preds["recent_mean_24h"].append(np.repeat(w["y_last24_mean"][:, None], H, axis=1))
    keys, y, prec = np.concatenate(keys), np.concatenate(y), np.concatenate(prec)
    preds = {b: np.concatenate(p) for b, p in preds.items()}
    city = keys[:, 0]
    result = {b: {"overall": metrics(p - y),
                  "by_city": {sources[c].city: metrics((p - y)[city == c]) for c in range(len(sources))},
                  "by_precip_class_at_target_hour": {n: metrics(p - y, rule(prec)) for n, rule in CLASSES.items()}}
              for b, p in preds.items()}
    return result, keys, y, preds


def minute_leak_comparison(fraction=1.0):
    """Leaky vs causal inputs on identical minute anchors and interpolated targets."""
    from data_pipeline.minute_full_source import CityMinuteSource
    out, keys = {}, []
    errs = {"leaky_persistence": [], "leaky_linear_extrapolation": [],
            "causal_persistence": [], "causal_linear_extrapolation": []}
    for c, city in enumerate(CITIES):
        src = CityMinuteSource(city)
        a = src.splits["test"]
        w = src.windows(a)                                        # existing source: interpolated values
        y = w["epl"]
        steps = np.arange(1, H + 1)
        causal = causal_minute_proxy_baselines(src.epl[:, 0], a)
        errs["leaky_persistence"].append(np.repeat(w["epl_now"][:, None], H, axis=1) - y)
        errs["leaky_linear_extrapolation"].append(w["epl_now"][:, None] + steps * (w["epl_now"] - w["epl_prev"])[:, None] - y)
        errs["causal_persistence"].append(causal["persistence"] - y)
        errs["causal_linear_extrapolation"].append(causal["linear_extrapolation"] - y)
        keys.append(np.stack([np.full(len(a), c), a], axis=1))
    keys = np.concatenate(keys)
    for name, e in errs.items():
        e = np.concatenate(e)
        out[name] = {"overall": metrics(e),
                     "by_city": {city: metrics(e[keys[:, 0] == c]) for c, city in enumerate(CITIES)}}
    return out, keys


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "causal_baseline_audit"
    if folder.exists():
        raise SystemExit(f"{folder} exists; choose a new folder (nothing is overwritten).")
    folder.mkdir(parents=True)

    sources = []
    for city, c in CITIES.items():
        table = pd.read_csv(config.PHYSICS_DIR / f"{city}_physics.csv", parse_dates=["valid_time"])
        sources.append(CausalHourlySource(table, city, c["latitude"], c["longitude"]))
    scaler = fit_feature_scaler(sources)
    hourly, keys, y, preds = evaluate_hourly(sources)
    minute, minute_keys = minute_leak_comparison()
    existing = json.load(open(config.RESULTS_DIR / "metrics_test.json", encoding="utf-8"))["datasets"]["paper"]

    cfg = {**CausalHourlySource.metadata(),
           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "created_by": "evaluation/evaluate_causal_baselines.py",
           "cities": list(CITIES), "features": config.FEATURES, "target": config.EPL_COLUMN,
           "split": "chronological 70/15/15 of each city's usable hourly origins; the next split's first input row "
                    "is after the previous split's last target row",
           "split_boundaries": {s.city: s.split_boundaries() for s in sources},
           "excluded_origins": {s.city: s.excluded for s in sources},
           "scaling": {"policy": "global StandardScaler fitted on training input rows only (not used by the baselines)",
                       "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
           "random_seed": config.SEED, "randomness": "none (all origins used, deterministic order)",
           "baselines": {"persistence": "y_hat(t+k) = y(t)", "linear_extrapolation": "y_hat(t+k) = y(t) + k (y(t) - y(t-1))",
                         "recent_mean_24h": "y_hat(t+k) = mean(y(t-23..t))", "k": "1..5 hours"},
           "minute_leak_comparison": "test anchors of data_pipeline/minute_full_source.py (5-minute stride); targets are "
                                     "linearly interpolated minute values (estimates, not observations); leaky inputs use "
                                     "interpolated EPL at the anchor (needs hour h+1), causal inputs hold hour h",
           "limitations": ["ERA5 is retrospective reanalysis: causal in time, not operationally available in real time",
                           "labels are ITU-R model outputs from reanalysis weather, not link measurements",
                           "no measured minute data exist; minute values are interpolated or held hourly values"]}
    with open(folder / "config.json", "w", encoding="utf-8") as file:
        json.dump(cfg, file, indent=2)
    with open(folder / "baseline_metrics.json", "w", encoding="utf-8") as file:
        json.dump({"causal_hourly_v1_test": hourly, "test_origins": int(len(keys)),
                   "minute_leak_comparison_test": minute, "minute_test_anchors": int(len(minute_keys)),
                   "existing_hourly_protocol_for_context (stratified 8,916 test windows, read from results/paper)":
                       {b: {"RMSE": v["RMSE"], "MAE": v["MAE"]} for b, v in existing["baselines"].items() if "MAE" in v}},
                  file, indent=2)
    np.savez_compressed(folder / "aligned_keys_hourly.npz", city_index=keys[:, 0], origin_hour_since_epoch=keys[:, 1],
                        cities=np.array(list(CITIES)), y=y.astype(np.float32),
                        **{f"pred_{b}": p.astype(np.float32) for b, p in preds.items()})
    np.savez_compressed(folder / "aligned_keys_minute.npz", city_index=minute_keys[:, 0].astype(np.int8),
                        anchor_minute=minute_keys[:, 1].astype(np.int32), cities=np.array(list(CITIES)))
    print("test origins:", len(keys))
    for b, r in hourly.items():
        print(f"{b:22s} RMSE {r['overall']['RMSE']:.4f} MAE {r['overall']['MAE']:.4f} by hour "
              f"{np.round(r['overall']['RMSE_by_step'], 3).tolist()}")
    print("minute anchors:", len(minute_keys))
    for b, r in minute.items():
        print(f"{b:28s} RMSE {r['overall']['RMSE']:.4f} MAE {r['overall']['MAE']:.4f} by minute "
              f"{np.round(r['overall']['RMSE_by_step'], 4).tolist()}")


if __name__ == "__main__":
    main()
