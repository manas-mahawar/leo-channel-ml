"""Dataset figures: attenuation components per city and sampling coverage.

Usage:
    python evaluation/plot_dataset.py
"""

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

FIG_DIR = config.RESULTS_DIR / "figures"
COMPONENTS = {"gas_attenuation_dB": ("Gas", "#2a78d6"),
              "cloud_attenuation_dB": ("Cloud", "#1baf7a"),
              "rain_attenuation_dB": ("Rain", "#eb6834")}
plt.rcParams.update({"axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    samples = {c: pd.read_csv(config.DATASET_DIR / f"samples_{c}.csv", parse_dates=["valid_time"])
               for c in CITIES}

    # Mean contribution of each component (stacked) per city
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bottom = np.zeros(len(CITIES))
    for column, (label, color) in COMPONENTS.items():
        means = np.array([samples[c][column].mean() for c in CITIES])
        ax.bar(list(CITIES), means, 0.6, bottom=bottom, color=color, label=label,
               edgecolor="white", linewidth=1)
        bottom += means
    ax.set_ylabel("Mean attenuation (dB)"); ax.legend(fontsize=8)
    ax.set_title("Mean EPL components, 10,000 sampled hours per city (39 GHz, 30 deg)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "dataset_components.png", dpi=300); fig.savefig(FIG_DIR / "dataset_components.pdf"); plt.close(fig)

    # EPL distribution per city (log y)
    fig, axes = plt.subplots(2, 3, figsize=(9, 5), sharex=True, sharey=True)
    for ax, city in zip(axes.ravel(), CITIES):
        ax.hist(samples[city][config.EPL_COLUMN], bins=np.linspace(0, 20, 81), color="#2a78d6")
        ax.set_yscale("log"); ax.set_title(city, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("EPL (dB)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Sampled hours (log)")
    fig.suptitle("EPL distribution of the sampled hours", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "dataset_epl_hist.png", dpi=300); fig.savefig(FIG_DIR / "dataset_epl_hist.pdf"); plt.close(fig)

    # Sampling coverage: samples per (month, hour) stratum, all cities
    allsamp = pd.concat(samples.values())
    grid = allsamp.groupby([allsamp.valid_time.dt.month, allsamp.valid_time.dt.hour]).size().unstack()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap="Blues", vmin=0)
    ax.set_xlabel("Hour of day (UTC)"); ax.set_ylabel("Month"); ax.grid(False)
    ax.set_yticks(range(12), range(1, 13))
    fig.colorbar(im, ax=ax, label="Samples (six cities)")
    ax.set_title("Stratified sampling coverage")
    fig.tight_layout(); fig.savefig(FIG_DIR / "dataset_sampling.png", dpi=300); fig.savefig(FIG_DIR / "dataset_sampling.pdf"); plt.close(fig)
    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
