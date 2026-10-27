"""Streaming evaluation for the full-series 1-minute reconstruction.

Every requested test anchor is scored exactly once, batch by batch; only
running sums are kept (no full prediction arrays). The model and the two
baselines are scored on the same anchors, target minutes and horizons:

  persistence:           EPL_hat(t+k) = EPL(t)
  linear extrapolation:  EPL_hat(t+k) = EPL(t) + k * (EPL(t) - EPL(t-1)),   k = 1..5 minutes
  model:                 Gaussian mean mu_k (and sigma_k for coverage / width / NLL)

EPL(t) is the anchor minute's value, available at prediction time (it equals
the sum of the three instantaneous attenuation inputs at t).

Caveat (documented in docs/MINUTE_REPLICATION_IMPLEMENTATION.md): minutes are
linear interpolations of hourly values, so a minute between hours h and h+1
already contains the hour-(h+1) value. Inputs therefore carry information from
up to one hour ahead, and linear extrapolation is exact unless the five target
minutes cross an hour boundary.

Usage (evaluate a saved minute model on a test fraction):
    python evaluation/evaluate_minute_full.py results/minute_full_smoke --test-fraction 0.01
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from data_pipeline.minute_full_loader import MinuteBatches, MinuteFullData

H = config.HORIZON
CONDITIONS = {"dry (no precipitation)": lambda p: p == 0, "light (0-2.5 mm/h)": lambda p: (p > 0) & (p < 2.5),
              "heavy (>= 2.5 mm/h)": lambda p: p >= 2.5}


class Accumulator:
    """Running sums of squared / absolute errors per horizon (and optional groups)."""

    def __init__(self):
        self.n, self.se, self.ae = np.zeros(H), np.zeros(H), np.zeros(H)

    def add(self, err, mask=None):
        if mask is None:
            mask = np.ones_like(err, dtype=bool)
        self.n += mask.sum(0)
        self.se += np.where(mask, err ** 2, 0).sum(0)
        self.ae += np.where(mask, np.abs(err), 0).sum(0)

    def result(self):
        n = np.maximum(self.n, 1)
        return {"n_values": int(self.n.sum()), "RMSE": float(np.sqrt(self.se.sum() / max(self.n.sum(), 1))),
                "MAE": float(self.ae.sum() / max(self.n.sum(), 1)),
                "RMSE_by_step": np.sqrt(self.se / n).tolist(), "MAE_by_step": (self.ae / n).tolist()}


def evaluate(batches, model=None, keep_predictions=0):
    """Score the model (if given) and both baselines over every anchor of `batches` once."""
    names = ["persistence", "linear_extrapolation"] + (["model"] if model is not None else [])
    total = {k: Accumulator() for k in names}
    city = {k: {c: Accumulator() for c in batches.data.cities} for k in names}
    cond = {k: {c: Accumulator() for c in CONDITIONS} for k in names}
    w_se_std, w_se_raw, w_n = 0.0, np.zeros(len(config.WEATHER_TARGETS)), 0
    unc = {"inside_95": 0, "width_sum": 0.0, "nll_sum": 0.0, "n": 0}
    kept, seen, t0 = [], set(), time.time()
    precip_i = config.WEATHER_TARGETS.index("precipitation_mm")
    steps = np.arange(1, H + 1)

    for i in range(len(batches)):
        raw = batches.raw_batch(i)
        seen.update(zip(raw["city"].tolist(), raw["anchor"].tolist()))
        y = raw["epl"]
        preds = {"persistence": np.repeat(raw["epl_now"][:, None], H, axis=1),
                 "linear_extrapolation": raw["epl_now"][:, None] + (raw["epl_now"] - raw["epl_prev"])[:, None] * steps}
        if model is not None:
            X, targets = batches[i]
            out = model.predict_on_batch(X)
            log_var = np.clip(np.asarray(out["epl"][:, H:], float), *config.LOG_VAR_CLIP)
            mu, sigma = np.asarray(out["epl"][:, :H], float), np.exp(0.5 * log_var)
            preds["model"] = mu
            z = (y - mu) / sigma
            unc["inside_95"] += int((np.abs(z) <= 1.96).sum())
            unc["width_sum"] += float((2 * 1.96 * sigma).sum())
            unc["nll_sum"] += float((0.5 * (log_var + z ** 2)).sum())
            unc["n"] += y.size
            w_pred = np.asarray(out["weather"], float)
            w_se_std += float(((w_pred - targets["weather"]) ** 2).sum())
            w_raw = batches.data.w_scaler.inverse_transform(w_pred.reshape(-1, w_pred.shape[-1])).reshape(w_pred.shape)
            w_se_raw += ((w_raw - raw["weather"]) ** 2).sum((0, 1))
            w_n += len(y) * H
            if len(kept) * batches.batch_size < keep_predictions:
                kept.append({"city": raw["city"], "anchor": raw["anchor"], "y": y, "mu": mu, "sigma": sigma})
        precip = raw["weather"][:, :, precip_i]
        for k, p in preds.items():
            err = p - y
            total[k].add(err)
            for c_i in np.unique(raw["city"]):
                city[k][batches.data.cities[c_i]].add(err[raw["city"] == c_i])
            for c_name, rule in CONDITIONS.items():
                cond[k][c_name].add(err, rule(precip))

    result = {"anchors_scored": len(seen), "anchors_requested": len(batches.anchors),
              "seconds": round(time.time() - t0, 1),
              "metrics": {k: total[k].result() for k in names},
              "by_city": {k: {c: a.result() for c, a in city[k].items()} for k in names},
              "by_condition_at_target_minute": {k: {c: a.result() for c, a in cond[k].items()} for k in names}}
    if model is not None:
        result["uncertainty"] = {"PICP_95_pct": 100 * unc["inside_95"] / unc["n"],
                                 "mean_interval_width_dB": unc["width_sum"] / unc["n"],
                                 "NLL_eq10": unc["nll_sum"] / unc["n"]}
        result["weather"] = {"RMSE_standardised": float(np.sqrt(w_se_std / (w_n * len(config.WEATHER_TARGETS)))),
                             **{f"RMSE_{n}": float(np.sqrt(v / w_n)) for n, v in zip(config.WEATHER_TARGETS, w_se_raw)}}
    predictions = {k: np.concatenate([d[k] for d in kept])[:keep_predictions] for k in kept[0]} if kept else None
    return result, predictions


def main():
    import tensorflow as tf
    from train_paper_model import gaussian_nll

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("--test-fraction", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    exp = Path(args.exp_dir)
    cfg = json.load(open(exp / "run_config.json", encoding="utf-8"))
    out = exp / f"test_eval_fraction_{args.test_fraction:g}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting.")
    data = MinuteFullData()
    data.load_scalers(exp)
    model = tf.keras.models.load_model(ROOT / cfg["model_dir"] / "best.keras",
                                       custom_objects={"gaussian_nll": gaussian_nll}, compile=False)
    batches = MinuteBatches(data, "test", batch_size=args.batch_size, fraction=args.test_fraction, seed=cfg["seed"])
    result, _ = evaluate(batches, model)
    with open(out, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "training"))
    main()
