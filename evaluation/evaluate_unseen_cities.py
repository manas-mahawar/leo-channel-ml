"""Zero-shot evaluation on the ten unseen cities (paper Sect. 3.3, 4.2).

Paper protocol (Eqs. 12-13): the input vector of an unseen location is a
convex combination of the feature vectors of its K = 3 nearest training
cities at the same time index, with weights softmax(-d_k / tau) over the
Haversine distances d_k. The trained model is not retrained.

Our choices (paper does not specify them):
  * tau = config.TAU_KM (sensitivity to 250/500/1000 km is reported);
  * the latitude/longitude features are set to the target city's coordinates;
  * evaluation hours: every hour from the latest first-test-anchor of the
    training cities to the end of 2024 (the test period, never trained on);
  * "true EPL" is our ITU-R label computed from the city's own ERA5 weather.

Input modes reported:
  synthesized         Eq. 12-13 weighted sum (the paper's text)
  synthesized_median  unweighted median of the 3 neighbours (paper Fig. 2:
                      "Climatology Service K=3 nearest cities median")
  own_inputs          extra check, not in the paper: the city's real weather
                      and ITU-R features (separates model generalisation
                      from interpolation error)
plus persistence baselines (own EPL at t; weighted neighbour EPL at t).

Usage:
    python evaluation/evaluate_unseen_cities.py                 # main model
    python evaluation/evaluate_unseen_cities.py multitask_msl   # MSL-pressure model
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import config
from data_pipeline.build_paper_dataset import add_features, use_msl_pressure
from data_pipeline.cities import CITIES, UNSEEN_CITIES
from train_paper_model import dataset_dir, gaussian_nll

H = config.HORIZON
OUT = config.RESULTS_DIR


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def neighbour_weights(lat, lon, tau_km):
    """K nearest training cities and their softmax(-d/tau) weights (Eq. 13)."""
    d = {c: haversine_km(lat, lon, v["latitude"], v["longitude"]) for c, v in CITIES.items()}
    nearest = sorted(d, key=d.get)[:config.K_NEIGHBOURS]
    dist = np.array([d[c] for c in nearest])
    w = np.exp(-(dist - dist.min()) / tau_km)          # shift for numerical safety
    return nearest, dist, w / w.sum()


def load_features(city, coords, msl=False):
    df = pd.read_csv(config.PHYSICS_DIR / f"{city}_physics.csv", parse_dates=["valid_time"])
    if msl:
        years = config.YEARS if city in CITIES else config.UNSEEN_YEARS
        df = use_msl_pressure(df, city, years)
    df = add_features(df, coords["latitude"], coords["longitude"])
    return df.set_index("valid_time")


def windows(features, anchors_pos):
    past = anchors_pos[:, None] + np.arange(-config.SEQ_LEN + 1, 1)
    return features[past]


def score(city, mode, n, y, mu, sigma=None):
    row = {"city": city, "mode": mode, "hours": n,
           "true_EPL_mean_dB": float(y.mean()), "pred_EPL_mean_dB": float(mu.mean()),
           "abs_error_of_means_dB": float(abs(mu.mean() - y.mean())),
           "MAE_dB": float(np.mean(np.abs(mu - y))), "RMSE_dB": float(np.sqrt(np.mean((mu - y) ** 2)))}
    if sigma is not None:
        row["PICP_95_pct"] = float(100 * np.mean(np.abs((y - mu) / sigma) <= 1.96))
        row["sharpness_dB"] = float(sigma.mean())
    return row


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "multitask"
    if variant == "minute_multitask":
        raise SystemExit("The unseen-city protocol is defined on hourly data; not run for the minute model.")
    msl = variant == "multitask_msl"
    suffix = "" if variant == "multitask" else f"_{variant}"

    model = tf.keras.models.load_model(ROOT / "models" / "paper" / variant / "model.keras",
                                       custom_objects={"gaussian_nll": gaussian_nll})
    x_scaler = joblib.load(dataset_dir(variant) / "feature_scaler.pkl")

    test_meta = pd.read_csv(config.DATASET_DIR / "meta_test.csv", parse_dates=["anchor_time"])
    start = test_meta.groupby("city").anchor_time.min().max()
    train_feats = {c: load_features(c, v, msl) for c, v in CITIES.items()}
    met_cols = [f for f in config.FEATURES if f not in ("latitude", "longitude")]
    rows, per_tau, neighbours_all, series = [], [], {}, {}

    for city, coords in UNSEEN_CITIES.items():
        own = load_features(city, coords, msl)
        times = own.index
        pos = np.flatnonzero((times >= start) & (times <= times[-1] - pd.Timedelta(hours=H)))
        if pos.min() < config.SEQ_LEN - 1 + config.STD_WINDOW_H - 1:
            raise ValueError(f"{city}: not enough history before the first anchor.")
        y = own[config.EPL_COLUMN].to_numpy()[pos[:, None] + np.arange(1, H + 1)]

        def predict(table):
            X = windows(table[config.FEATURES].to_numpy(np.float32), pos)
            Xs = x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
            out = model.predict(Xs, batch_size=1024, verbose=0)["epl"]
            log_var = np.clip(out[:, H:], *config.LOG_VAR_CLIP)
            return out[:, :H].astype(float), np.exp(0.5 * log_var).astype(float)

        modes = {}
        for tau in (250.0, 500.0, 1000.0):
            nearest, dist, w = neighbour_weights(coords["latitude"], coords["longitude"], tau)
            synth = own.copy()
            synth[met_cols] = sum(wk * train_feats[c].loc[times, met_cols].to_numpy()
                                  for c, wk in zip(nearest, w))
            mu, sigma = predict(synth)
            per_tau.append({"city": city, "tau_km": tau, "MAE": float(np.mean(np.abs(mu - y))),
                            "abs_error_of_means": float(abs(mu.mean() - y.mean()))})
            if tau == config.TAU_KM:
                modes["synthesized"] = (mu, sigma)
                neighbours_all[city] = {c: {"distance_km": round(float(dk), 1), "weight": round(float(wk), 4)}
                                        for c, dk, wk in zip(nearest, dist, w)}
                epl_synth_t = sum(wk * train_feats[c].loc[times, config.EPL_COLUMN].to_numpy()
                                  for c, wk in zip(nearest, w))[pos]

        median = own.copy()
        median[met_cols] = np.median(np.stack([train_feats[c].loc[times, met_cols].to_numpy()
                                               for c in nearest]), axis=0)
        modes["synthesized_median"] = predict(median)
        modes["own_inputs"] = predict(own)

        for mode, (mu, sigma) in modes.items():
            rows.append(score(city, mode, len(pos), y, mu, sigma))
        own_persist = np.repeat(own[config.EPL_COLUMN].to_numpy()[pos][:, None], H, axis=1)
        rows.append(score(city, "persistence_own", len(pos), y, own_persist))
        rows.append(score(city, "persistence_synthesized", len(pos), y, np.repeat(epl_synth_t[:, None], H, axis=1)))
        series[city] = np.stack([y[:, 0], modes["synthesized"][0][:, 0], modes["own_inputs"][0][:, 0]])
        series[f"{city}_start"] = np.array([str(times[pos[0]])])
        print(city, neighbours_all[city])

    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"table8_unseen_cities{suffix}.csv", index=False)
    pd.DataFrame(per_tau).to_csv(OUT / f"unseen_tau_sensitivity{suffix}.csv", index=False)
    summary = table.groupby("mode")[["true_EPL_mean_dB", "pred_EPL_mean_dB", "abs_error_of_means_dB",
                                     "MAE_dB", "RMSE_dB", "PICP_95_pct", "sharpness_dB"]].mean()
    summary.to_csv(OUT / f"table8_unseen_summary{suffix}.csv")
    with open(OUT / "unseen_neighbours.json", "w", encoding="utf-8") as file:
        json.dump(neighbours_all, file, indent=2)
    np.savez(OUT / f"unseen_series{suffix}.npz", **series)   # hour+1 true / synthesized / own

    paper_style = table[table["mode"] == "synthesized"][["city", "true_EPL_mean_dB", "pred_EPL_mean_dB",
                                                          "abs_error_of_means_dB", "MAE_dB", "RMSE_dB", "PICP_95_pct"]]
    print(f"\n=== {variant}: Table 8 style (synthesized inputs, tau = {config.TAU_KM:.0f} km) ===")
    print(paper_style.round(3).to_string(index=False))
    print("\n=== Mean over the ten cities, by mode ===")
    print(summary.round(3).to_string())
    print("\n=== tau sensitivity (mean over cities) ===")
    print(pd.DataFrame(per_tau).groupby("tau_km")[["MAE", "abs_error_of_means"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
