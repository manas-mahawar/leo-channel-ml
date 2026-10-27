"""Figures 1 and 2: model architecture and system pipeline (as implemented).

Usage:
    python evaluation/make_diagrams.py
Outputs: results/paper/figures/fig1_architecture.{png,pdf},
         results/paper/figures/fig2_pipeline.{png,pdf}
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

FIG_DIR = config.RESULTS_DIR / "figures"
FILL = {"data": "#cde2fb", "model": "#b7d3f6", "head": "#d8f1e7", "out": "#f3f3f1", "eval": "#fde6d7"}
EDGE = "#3d3d3a"


def box(ax, x, y, w, h, text, kind, size=8.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                facecolor=FILL[kind], edgecolor=EDGE, linewidth=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color="#1f1f1d")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.0, shrinkA=0, shrinkB=0))


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure1():
    fig, ax = plt.subplots(figsize=(7, 7.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off")
    box(ax, 5, 10.2, 8.6, 0.9, "Input sequence: 60 time steps x 14 features\n"
        "lat, lon | T, RH, P, precip, cloud | mu/sigma gas, cloud, rain | elevation", "data")
    box(ax, 5, 8.7, 4.8, 0.8, "LSTM layer 1 (256 units)  + dropout 0.2", "model")
    box(ax, 5, 7.6, 4.8, 0.8, "LSTM layer 2 (256 units)  + dropout 0.2", "model")
    box(ax, 5, 6.5, 4.8, 0.8, "LSTM layer 3 (256 units)\nlast hidden state (256)", "model")
    for y1, y2 in ((9.75, 9.1), (8.3, 8.0), (7.2, 6.9)):
        arrow(ax, 5, y1, 5, y2)
    box(ax, 2.6, 4.9, 3.6, 0.9, "Weather head: Dense 256 -> 25\n(paper Fig. 1: 256 -> 256 -> 25)", "head", 8)
    box(ax, 7.4, 4.9, 3.6, 0.9, "EPL head: Dense 256 -> 10\n(paper Fig. 1: 256 -> 256 -> 10)", "head", 8)
    arrow(ax, 4.2, 6.1, 2.8, 5.35); arrow(ax, 5.8, 6.1, 7.2, 5.35)
    box(ax, 2.6, 3.5, 3.6, 0.8, "Reshape 5 x 5\n(5 h x T, RH, P, precip, cloud)", "head", 8)
    box(ax, 7.4, 3.5, 3.6, 0.8, "Split: mu (5) + log sigma^2 (5)", "head", 8)
    arrow(ax, 2.6, 4.45, 2.6, 3.9); arrow(ax, 7.4, 4.45, 7.4, 3.9)
    box(ax, 2.6, 2.2, 3.6, 0.9, "5-step weather forecast\nMSE loss, weight 1.0", "out", 8)
    box(ax, 7.4, 2.2, 3.6, 0.9, "5-step EPL forecast + sigma\nGaussian NLL (Eq. 10), weight 1.0", "out", 8)
    arrow(ax, 2.6, 3.1, 2.6, 2.65); arrow(ax, 7.4, 3.1, 7.4, 2.65)
    ax.text(5, 0.9, "Adam 1e-4, batch 128, <= 50 epochs, early stopping patience 6, seed 42.\n"
            "Main model: 1,337,123 parameters. The Fig. 1 hidden head layer (256, ReLU assumed)\n"
            "is tested separately as variant 'multitask_fig1heads'.",
            ha="center", va="center", fontsize=7.5, color="#52514e")
    ax.set_title("Fig. 1  Multi-task LSTM architecture (as implemented)", fontsize=10)
    save(fig, "fig1_architecture")


def figure2():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 24); ax.set_ylim(0, 12); ax.axis("off")
    cols = [
        ("Input data", "data", ["Open-Meteo ERA5\nhourly, 6 cities\n2020-2024",
                                "T, RH, surface P,\nprecip, rain,\ncloud cover",
                                "ERA5 CDS tclw\n(LWP proxy\ncalibration)",
                                "City coordinates\n(paper Table 3)"]),
        ("ITU-R labels", "model", ["P.676-12 gas\nAnnex 1 gamma x\nAnnex 2 heights",
                                   "P.840-8 cloud\nRH + precip\nLWP proxy",
                                   "Rain: P.838-3 +\nP.618-13, height\nfrom P.839-4",
                                   "EPL = gas +\ncloud + rain\n39 GHz, 30 deg"]),
        ("Dataset", "data", ["14 features\n(Table 2); sigma =\n24-h trailing std",
                             "10,000 anchors\nper city, stratified\nmonth x hour",
                             "60-h input,\n5-h targets",
                             "Chronological\n70/15/15, 65-h purge,\ntrain-only scaling"]),
        ("Model", "head", ["3 x LSTM-256\ndropout 0.2",
                           "Weather head\n5 x 5 (MSE)",
                           "EPL head\nmu, log sigma^2 (NLL)",
                           "Early stopping\nbest checkpoint"]),
        ("Evaluation", "eval", ["RMSE / MAE per\ncity and hour\n(Table 7)",
                                "PICP 95 %, NLL,\nsharpness,\ncalibration",
                                "Baselines:\npersistence,\nclimatology",
                                "Ablations\n(Tables 9, 10)"]),
        ("Unseen cities", "eval", ["10 cities, 2024;\nown ERA5 ->\ntrue EPL",
                                   "K = 3 nearest,\nsoftmax(-d / tau)\ntau = 500 km",
                                   "Also: median of 3,\nown inputs (extra)",
                                   "Table 8, Figs. 5-7"]),
    ]
    xs = [2, 6, 10, 14, 18, 22]
    for x, (title, kind, items) in zip(xs, cols):
        ax.add_patch(FancyBboxPatch((x - 1.85, 0.6), 3.7, 10.6, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor="white", edgecolor="#b5b4ad", linewidth=0.8))
        ax.text(x, 10.7, title, ha="center", va="center", fontsize=10, fontweight="bold")
        for i, text in enumerate(items):
            box(ax, x, 9.0 - i * 2.3, 3.4, 1.8, text, kind, 7.5)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        arrow(ax, x1 + 1.9, 5.9, x2 - 1.9, 5.9)
    ax.text(12, 0.15, "Paper Fig. 2 additionally shows '1-minute uniform sampling', a 60-min window and a "
            "5-minute stride; that reading is tested separately (dataset paper_minute).",
            ha="center", fontsize=7.5, color="#52514e")
    ax.set_title("Fig. 2  End-to-end pipeline (as implemented)", fontsize=11)
    save(fig, "fig2_pipeline")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure1()
    figure2()
    print(f"Saved Fig. 1 and Fig. 2 to {FIG_DIR}")
