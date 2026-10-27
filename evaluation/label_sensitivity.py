"""Sensitivity of the EPL label statistics to two choices the paper leaves open.

* Elevation angle (paper: "fixed ... representative of LEO", no value): 30
  (ours), 45, 60, 90 deg. Gas and cloud scale exactly with 1/sin(el); rain is
  recomputed with P.618 at each elevation.
* Rain input: ERA5 liquid rain (ours) or total precipitation (incl. snow).

Nothing is retrained and no dataset is changed. The statistics are compared
with values visible in the paper (references only): unseen-city mean true EPL
0.911 dB (Table 8) and a trained-city maximum of about 6.2-6.7 dB (Figs. 3-4).

Usage:
    python evaluation/label_sensitivity.py
Output: results/replication_audit/label_sensitivity.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
import physics.build_attenuation as ba
from data_pipeline.cities import CITIES, UNSEEN_CITIES

OUT = ROOT / "results" / "replication_audit"
UNSEEN_START = "2024-04-09"            # same period as our unseen-city evaluation


def rain_at(elevation, rate, lat, lon):
    """P.618 rain attenuation at another elevation (temporarily overrides the module constants)."""
    old = ba.EL, ba.SIN_EL
    ba.EL, ba.SIN_EL = elevation, np.sin(np.deg2rad(elevation))
    try:
        return ba.rain_attenuation(rate, lat, lon)["rain_attenuation_dB"].to_numpy()
    finally:
        ba.EL, ba.SIN_EL = old


def epl(phys, coords, elevation, rain_column):
    scale = np.sin(np.deg2rad(config.ELEVATION_DEG)) / np.sin(np.deg2rad(elevation))
    gas_cloud = (phys["gas_attenuation_dB"] + phys["cloud_attenuation_dB"]).to_numpy() * scale
    return gas_cloud + rain_at(elevation, phys[rain_column].to_numpy(), coords["latitude"], coords["longitude"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    load = lambda c: pd.read_csv(config.PHYSICS_DIR / f"{c}_physics.csv", parse_dates=["valid_time"])
    train = {c: load(c) for c in CITIES}
    unseen = {c: load(c) for c in UNSEEN_CITIES}
    rows = []
    for elevation in (30.0, 45.0, 60.0, 90.0):
        for rain_column in ("rain_mm", "precipitation_mm"):
            tr = np.concatenate([epl(p, CITIES[c], elevation, rain_column) for c, p in train.items()])
            un = [epl(p[p.valid_time >= UNSEEN_START], UNSEEN_CITIES[c], elevation, rain_column).mean()
                  for c, p in unseen.items()]
            rows.append({"elevation_deg": elevation, "rain_input": rain_column,
                         "train_cities_mean_dB": tr.mean(), "train_cities_p99_dB": np.percentile(tr, 99),
                         "train_cities_max_dB": tr.max(), "train_hours_above_6.7dB_pct": 100 * (tr > 6.7).mean(),
                         "unseen_cities_mean_of_period_means_dB": float(np.mean(un))})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "label_sensitivity.csv", index=False)
    print(table.round(3).to_string(index=False))
    print("\nPaper (reference): unseen-city mean true EPL 0.911 dB; trained-city max about 6.2-6.7 dB")


if __name__ == "__main__":
    main()
