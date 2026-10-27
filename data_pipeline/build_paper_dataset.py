"""Build the 60-step / 5-hour LSTM dataset for the six training cities.

Paper (Sect. 3.1): 10,000 hourly samples per city, stratified by month and
hour of day; chronological 70/15/15 split per location; inputs
standardised with training-set statistics only.

Our reading (documented in the manifest):
  * Each sampled hour t is the "anchor" of one window: inputs are the 14
    features at hours t-59..t, targets are EPL and the 5 weather variables
    at hours t+1..t+5. Windows come from the full, continuous hourly series.
  * mu features are the instantaneous ITU-R component at each hour (Eq. 8);
    sigma features are a trailing 24-h standard deviation (past only).
  * The sorted anchors are split 70/15/15; validation/test anchors that
    would share any hour with the previous split are dropped (purge gap).

Usage:
    python data_pipeline/build_paper_dataset.py          # surface pressure (main)
    python data_pipeline/build_paper_dataset.py --msl    # MSL pressure feature/target
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from data_pipeline.cities import CITIES

GAP_H = config.SEQ_LEN + config.HORIZON   # anchors closer than this share hours


def add_features(df, lat, lon):
    """Add the 14 paper features to an hourly physics table."""
    df = df.copy()
    df["latitude"] = lat           # Table 3 coordinates, not the ERA5 grid
    df["longitude"] = lon
    for name, column in [("gas", "gas_attenuation_dB"),
                         ("cloud", "cloud_attenuation_dB"),
                         ("rain", "rain_attenuation_dB")]:
        df[f"{name}_mean"] = df[column]
        df[f"{name}_std"] = df[column].rolling(config.STD_WINDOW_H).std()
    return df


def use_msl_pressure(df, city, years):
    """Replace the pressure feature/target with mean-sea-level pressure
    (paper Table 2). The ITU-R labels keep using surface pressure."""
    msl = pd.concat([pd.read_csv(config.MSL_DIR / f"{city}_{y}.csv", parse_dates=["valid_time"])
                     for y in years], ignore_index=True)
    if not msl["valid_time"].equals(df["valid_time"]):
        raise ValueError(f"{city}: MSL pressure hours do not match the physics table.")
    df = df.copy()
    df["pressure_hPa"] = msl["pressure_msl_hPa"].to_numpy()
    return df


def stratified_anchors(times, eligible, rng):
    """Draw SAMPLES_PER_CITY anchors, allocated to (month, hour) strata in
    proportion to stratum size (largest-remainder rounding)."""
    idx = np.flatnonzero(eligible)
    strata = pd.Series(idx).groupby([times.dt.month.values[idx],
                                     times.dt.hour.values[idx]])
    sizes = strata.size()
    quota = sizes / sizes.sum() * config.SAMPLES_PER_CITY
    count = np.floor(quota).astype(int)
    remainder = (quota - count).sort_values(ascending=False)
    count.loc[remainder.index[:config.SAMPLES_PER_CITY - count.sum()]] += 1

    chosen = []
    for key, members in strata:
        chosen.append(rng.choice(members.to_numpy(), size=count[key], replace=False))
    return np.sort(np.concatenate(chosen))


def split_anchors(anchors):
    """Chronological 70/15/15 split with a purge gap between splits."""
    n = len(anchors)
    n_train = int(round(n * config.TRAIN_FRAC))
    n_val = int(round(n * config.VAL_FRAC))
    train = anchors[:n_train]
    val = anchors[n_train:n_train + n_val]
    test = anchors[n_train + n_val:]

    val_kept = val[val - train[-1] >= GAP_H]
    test_kept = test[test - val[-1] >= GAP_H]
    purged = {"val": len(val) - len(val_kept), "test": len(test) - len(test_kept)}
    return {"train": train, "val": val_kept, "test": test_kept}, purged


def make_windows(df, anchors):
    feats = df[config.FEATURES].to_numpy(np.float32)
    weather = df[config.WEATHER_TARGETS].to_numpy(np.float32)
    epl = df[config.EPL_COLUMN].to_numpy(np.float32)

    past = anchors[:, None] + np.arange(-config.SEQ_LEN + 1, 1)
    future = anchors[:, None] + np.arange(1, config.HORIZON + 1)
    return {
        "X": feats[past],                 # (n, 60, 14)
        "weather": weather[future],       # (n, 5, 5)
        "epl": epl[future],               # (n, 5)
        "epl_now": epl[anchors],          # EPL at the anchor (persistence)
    }


def check_windows(df, anchors, w):
    """Feature/target alignment: last input step is the anchor hour, and
    targets are the next five hours."""
    last = w["X"][:, -1, :]
    parts = last[:, [config.FEATURES.index(c) for c in ("gas_mean", "cloud_mean", "rain_mean")]]
    if not np.allclose(parts.sum(axis=1), w["epl_now"], atol=1e-4):
        raise ValueError("Last input step is not aligned with the anchor EPL.")
    times = df["valid_time"].to_numpy()
    if not (times[anchors + 1] - times[anchors] == np.timedelta64(1, "h")).all():
        raise ValueError("First target hour is not anchor + 1 h.")
    if not np.isfinite(w["X"]).all():
        raise ValueError("Non-finite inputs in windows.")


def main():
    rng = np.random.default_rng(config.SEED)
    msl = "--msl" in sys.argv
    out_dir = config.DATASET_DIR_MSL if msl else config.DATASET_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = {s: [] for s in ("train", "val", "test")}
    city_info = {}

    for city_id, (city, coords) in enumerate(CITIES.items()):
        df = pd.read_csv(config.PHYSICS_DIR / f"{city}_physics.csv", parse_dates=["valid_time"])
        if df["valid_time"].duplicated().any() or not df["valid_time"].is_monotonic_increasing:
            raise ValueError(f"{city}: duplicate or unsorted timestamps.")
        if msl:
            df = use_msl_pressure(df, city, config.YEARS)
        df = add_features(df, coords["latitude"], coords["longitude"])

        # An anchor needs a full 60-h input window whose sigma features
        # have a full 24-h history, and 5 future hours.
        first = config.SEQ_LEN - 1 + config.STD_WINDOW_H - 1
        position = np.arange(len(df))
        eligible = (position >= first) & (position < len(df) - config.HORIZON)

        anchors = stratified_anchors(df["valid_time"], eligible, rng)
        if len(anchors) != config.SAMPLES_PER_CITY or len(np.unique(anchors)) != len(anchors):
            raise ValueError(f"{city}: sampling did not give unique {config.SAMPLES_PER_CITY} anchors.")

        # Keep the sampled hourly records (paper: dataset stored as CSV).
        df.iloc[anchors].to_csv(out_dir / f"samples_{city}.csv", index=False)

        splits, purged = split_anchors(anchors)
        info = {"purged": purged, "splits": {}}
        for split, idx in splits.items():
            w = make_windows(df, idx)
            check_windows(df, idx, w)
            w["city_id"] = np.full(len(idx), city_id, dtype=np.int8)
            w["anchor_time"] = df["valid_time"].to_numpy()[idx]
            parts[split].append(w)
            info["splits"][split] = {
                "n": int(len(idx)),
                "first_anchor": str(w["anchor_time"][0]),
                "last_anchor": str(w["anchor_time"][-1]),
                "epl_target_mean_dB": float(w["epl"].mean()),
                "epl_target_max_dB": float(w["epl"].max()),
            }
        sampled = df.iloc[anchors]
        info["samples_per_stratum_min_max"] = [
            int(sampled.groupby([sampled.valid_time.dt.month, sampled.valid_time.dt.hour]).size().min()),
            int(sampled.groupby([sampled.valid_time.dt.month, sampled.valid_time.dt.hour]).size().max()),
        ]
        city_info[city] = info
        print(city, {s: v["n"] for s, v in info["splits"].items()}, "purged", purged)

    data = {s: {k: np.concatenate([p[k] for p in parts[s]]) for k in parts[s][0]} for s in parts}

    # Scalers are fitted on training windows only.
    n_feat, n_weather = len(config.FEATURES), len(config.WEATHER_TARGETS)
    x_scaler = StandardScaler().fit(data["train"]["X"].reshape(-1, n_feat))
    w_scaler = StandardScaler().fit(data["train"]["weather"].reshape(-1, n_weather))
    joblib.dump(x_scaler, out_dir / "feature_scaler.pkl")
    joblib.dump(w_scaler, out_dir / "weather_scaler.pkl")

    for split, d in data.items():
        X = x_scaler.transform(d["X"].reshape(-1, n_feat)).reshape(d["X"].shape)
        W = w_scaler.transform(d["weather"].reshape(-1, n_weather)).reshape(d["weather"].shape)
        np.save(out_dir / f"X_{split}.npy", X.astype(np.float32))
        np.save(out_dir / f"weather_{split}.npy", W.astype(np.float32))
        np.save(out_dir / f"weather_raw_{split}.npy", d["weather"])
        np.save(out_dir / f"epl_{split}.npy", d["epl"])
        np.save(out_dir / f"epl_now_{split}.npy", d["epl_now"])
        pd.DataFrame({
            "city": np.array(list(CITIES))[d["city_id"]],
            "anchor_time": d["anchor_time"],
        }).to_csv(out_dir / f"meta_{split}.csv", index=False)

    manifest = {
        "paper": "El-Sayed et al., EURASIP JWCN 2026:102",
        "created_by": "data_pipeline/build_paper_dataset.py",
        "weather_source": str(config.WEATHER_DIR.relative_to(ROOT)),
        "physics_source": str(config.PHYSICS_DIR.relative_to(ROOT)),
        "settings": {
            "frequency_GHz": config.FREQUENCY_GHZ,
            "elevation_deg": config.ELEVATION_DEG,
            "polarization_tilt_deg": config.POLARIZATION_TILT_DEG,
            "samples_per_city": config.SAMPLES_PER_CITY,
            "seq_len_h": config.SEQ_LEN,
            "horizon_h": config.HORIZON,
            "split": [config.TRAIN_FRAC, config.VAL_FRAC, 1 - config.TRAIN_FRAC - config.VAL_FRAC],
            "purge_gap_h": GAP_H,
            "std_window_h": config.STD_WINDOW_H,
            "seed": config.SEED,
        },
        "features": config.FEATURES,
        "weather_targets": config.WEATHER_TARGETS,
        "shapes": {s: {k: list(np.shape(v)) for k, v in d.items() if k != "anchor_time"} for s, d in data.items()},
        "feature_scaler_mean": dict(zip(config.FEATURES, x_scaler.mean_.round(6).tolist())),
        "feature_scaler_std": dict(zip(config.FEATURES, x_scaler.scale_.round(6).tolist())),
        "cities": city_info,
        "assumptions": [
            "60 time steps of hourly data = 60 hours (paper text says '60 min').",
            "14 features from Table 2; Table 4's temporal features are not used.",
            "mu_* = instantaneous ITU-R component (Eq. 8); sigma_* = trailing 24-h std (paper gives no formula).",
            "Elevation fixed at 30 deg (paper: 'fixed elevation angle', value not given); constant feature scales to 0.",
            "Stratified sampling: proportional allocation over 288 (month, hour) strata, seed 42.",
            "Each sample is an anchor hour; its 60-h window and 5-h targets come from the continuous series.",
            "Chronological 70/15/15 split of sorted anchors per city, purge gap of 65 h between splits.",
            "Scalers fitted on training windows of all six cities together (paper Fig. 2 says 'normalized by location'; Sect. 3.1 says training-set statistics).",
            "EPL targets kept in dB (not scaled).",
        ],
        "pressure_feature": "mean sea level (Table 2)" if msl else "surface pressure (Sect. 3.1)",
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    for split, d in data.items():
        print(f"{split}: X {d['X'].shape}, weather {d['weather'].shape}, epl {d['epl'].shape}")
    print(f"Saved dataset and manifest to {out_dir}")


if __name__ == "__main__":
    main()
