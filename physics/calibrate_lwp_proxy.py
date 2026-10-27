"""Fit the cloud liquid-water-path (LWP) proxy used for P.840 labels.

The paper says LWP was "approximated using a simple proxy derived from
relative humidity and precipitation" but gives no formula. We use

    s     = clip((RH - RH_c) / (100 - RH_c), 0, 1)
    LWP   = a * s + b * precip_mm          [kg/m^2]

and fit a, b >= 0 (non-negative least squares) against real ERA5 total
column cloud liquid water (tclw) that we already downloaded from CDS for
Athens (2020-2024) and Bergen (2020-01..2021-02).

Only hours before LWP_CALIBRATION_END are used for fitting and for
choosing RH_c, so the 2023-2024 hours are an honest held-out check.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config


def load_era5_lwp():
    """Hourly ERA5 tclw at the Open-Meteo grid point of each city."""
    frames = []
    folders = sorted(p for p in config.ERA5_CDS_DIR.glob("*_*_*") if p.is_dir())
    for folder in folders:
        city = folder.name.split("_")[0]
        weather = pd.read_csv(config.WEATHER_DIR / f"{city}_2020.csv", nrows=1)
        lat, lon = weather.latitude[0], weather.longitude[0]

        ds = xr.open_dataset(folder / "data_stream-oper_stepType-instant.nc")
        point = ds.sel(latitude=lat, longitude=lon, method="nearest")
        frames.append(pd.DataFrame({
            "city": city,
            "valid_time": point.valid_time.values,
            "tclw_kg_m2": point.tclw.values,
        }))
    return pd.concat(frames, ignore_index=True)


def load_weather(cities):
    frames = []
    for city in cities:
        for year in config.YEARS:
            df = pd.read_csv(config.WEATHER_DIR / f"{city}_{year}.csv",
                             parse_dates=["valid_time"])
            frames.append(df.assign(city=city))
    return pd.concat(frames, ignore_index=True)


def design(df, rh_c):
    s = ((df["relative_humidity_percent"] - rh_c) / (100 - rh_c)).clip(0, 1)
    return np.column_stack([s, df["precipitation_mm"]])


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    era5 = load_era5_lwp()
    weather = load_weather(era5["city"].unique())
    data = era5.merge(weather, on=["city", "valid_time"], how="inner")
    if len(data) != len(era5):
        raise ValueError("Some ERA5 hours have no matching Open-Meteo hour.")

    fit = data[data["valid_time"] < config.LWP_CALIBRATION_END]
    held = data[data["valid_time"] >= config.LWP_CALIBRATION_END]
    y_fit = fit["tclw_kg_m2"].to_numpy()

    # Choose RH_c by training-period fit only.
    best = None
    for rh_c in range(40, 100, 5):
        coef, _ = nnls(design(fit, rh_c), y_fit)
        score = r2(y_fit, design(fit, rh_c) @ coef)
        if best is None or score > best[2]:
            best = (rh_c, coef, score)
    rh_c, (a, b), r2_fit = best

    pred_held = design(held, rh_c) @ np.array([a, b])
    y_held = held["tclw_kg_m2"].to_numpy()

    result = {
        "formula": "LWP = a*clip((RH-RH_c)/(100-RH_c),0,1) + b*precip_mm  [kg/m^2]",
        "rh_c_percent": rh_c,
        "a_kg_m2": float(a),
        "b_kg_m2_per_mm": float(b),
        "fit_period": f"< {config.LWP_CALIBRATION_END}",
        "fit_hours": int(len(fit)),
        "fit_cities": sorted(fit["city"].unique().tolist()),
        "fit_r2": float(r2_fit),
        "heldout_period": f">= {config.LWP_CALIBRATION_END}",
        "heldout_hours": int(len(held)),
        "heldout_cities": sorted(held["city"].unique().tolist()),
        "heldout_r2": float(r2(y_held, pred_held)),
        "heldout_corr": float(np.corrcoef(y_held, pred_held)[0, 1]),
        "heldout_mean_true": float(y_held.mean()),
        "heldout_mean_pred": float(pred_held.mean()),
        "note": "Our proxy, not the paper's (unpublished) formula.",
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.LWP_PROXY_FILE, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_held, pred_held, s=2, alpha=0.3)
    top = max(y_held.max(), pred_held.max())
    ax.plot([0, top], [0, top], "k--", lw=1)
    ax.set_xlabel("ERA5 tclw (kg/m$^2$)")
    ax.set_ylabel("RH + precipitation proxy (kg/m$^2$)")
    ax.set_title(f"LWP proxy, held-out hours (r = {result['heldout_corr']:.2f})")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "lwp_proxy_calibration.png", dpi=150)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
