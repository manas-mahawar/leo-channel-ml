"""Conditional calibration of the Gaussian EPL intervals (seed-42 models).

The overall 95 % coverage (PICP) can hide intervals that are too wide in dry
hours and too narrow in rain. This script reports, for each model:
  - empirical coverage of nominal 50/80/90/95/99 % intervals mu +- z sigma;
  - coverage and mean interval width (2 x 1.96 sigma) by weather class at the
    target hour and by true EPL level.
Grouping by the true weather / true EPL uses the outcome, so these groups
are diagnostics only; the model never sees them.

Usage:
    python evaluation/calibration_analysis.py
Output: results/paper/calibration_conditional.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

VARIANTS = ["multitask", "multitask_fig1heads", "multitask_msl", "no_itu", "epl_only"]
Z = {50: 0.674, 80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}


def main():
    rows = []
    for variant in VARIANTS:
        d = config.DATASET_DIR_MSL if variant == "multitask_msl" else config.DATASET_DIR
        y = np.load(d / "epl_test.npy").astype(float)
        precip = np.load(d / "weather_raw_test.npy")[:, :, config.WEATHER_TARGETS.index("precipitation_mm")]
        p = np.load(ROOT / "models" / "paper" / variant / "test_predictions.npz")
        mu, sd = p["epl_mean"].astype(float), p["epl_std"].astype(float)
        err = np.abs(y - mu)

        for nominal, z in Z.items():
            rows.append({"variant": variant, "group": "all", "subset": f"nominal {nominal} %",
                         "n": y.size, "coverage_pct": 100 * (err <= z * sd).mean(),
                         "mean_width_dB": (2 * z * sd).mean()})
        groups = {"no precipitation": precip == 0, "precipitation > 0": precip > 0,
                  "precipitation >= 2.5 mm/h": precip >= 2.5}
        for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 10), (10, 100)):
            groups[f"true EPL {lo}-{hi} dB"] = (y >= lo) & (y < hi)
        for name, m in groups.items():
            rows.append({"variant": variant, "group": "95 % interval", "subset": name, "n": int(m.sum()),
                         "coverage_pct": 100 * (err[m] <= 1.96 * sd[m]).mean(),
                         "mean_width_dB": (2 * 1.96 * sd[m]).mean(),
                         "mean_error_dB": (mu - y)[m].mean()})

    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "calibration_conditional.csv", index=False)
    print(table[table.variant.isin(["multitask", "multitask_fig1heads"])].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
