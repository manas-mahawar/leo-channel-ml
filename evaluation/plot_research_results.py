"""Presentation figures for the causal hourly experiment (read-only; no training).

Inputs (all existing artefacts, never modified):
  results/causal_hourly_fig1heads_seed42/  training_log.csv, run_config.json,
                                           test_predictions.npz, metrics_test.json
  results/causal_baseline_audit/aligned_keys_hourly.npz   baselines on identical keys
  data/processed/physics_paper/<city>_physics.csv          labels and components
  results/replication_audit/minute_protocol_baselines.csv  minute-reading baselines (Fig. 13)

Outputs: results/figures/NN_<name>.png (200 dpi) and .pdf (vector), plus
figures_manifest.json (key numbers per figure). Refuses to write into an existing
results/figures/ unless --overwrite is given (it then replaces only its own files).
--export-docs copies selected PNGs to docs/assets/figures/causal/ for the README.

Every metric is recomputed from the saved predictions and checked against
metrics_test.json before plotting. No sampling is used: scatter and residual
plots bin all 195,000 values; the time-series periods are chosen by the fixed
rules in plot_timeseries().

Usage:
    python evaluation/plot_research_results.py [--overwrite] [--export-docs]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import config
from data_pipeline.cities import CITIES
from evaluate_causal_model import metrics, target_precip

RUN = ROOT / "results" / "causal_hourly_fig1heads_seed42"
BASE = ROOT / "results" / "causal_baseline_audit" / "aligned_keys_hourly.npz"
MINUTE = ROOT / "results" / "replication_audit" / "minute_protocol_baselines.csv"
OUT = ROOT / "results" / "figures"
DOCS = ROOT / "docs" / "assets" / "figures" / "causal"
DOC_FIGURES = ["05_horizon_rmse_mae", "06_baseline_comparison", "08_precipitation_classes",
               "09_calibration_reliability", "13_paper_context"]

H = config.HORIZON
CITY_NAMES = list(CITIES)
PAPER_RMSE = 0.026                       # paper Table 7, "Overall" (p. 13)

# Validated categorical palette (dataviz reference palette, light mode)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e5e5e5"
C_MODEL, C_PERS, C_MEAN, C_LIN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_GAS, C_CLOUD, C_RAIN = "#eda100", "#1baf7a", "#6250d6"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal blue: dry, light, heavy
METHODS = {"model": ("LSTM (seed 42)", C_MODEL), "persistence": ("Persistence", C_PERS),
           "recent_mean_24h": ("Recent 24-h mean", C_MEAN), "linear_extrapolation": ("Linear extrapolation", C_LIN)}
FOOT = "Test set: causal_hourly_v1, 6 cities, 2024-04-04 to 2024-12-31, 39,000 origins x 5 horizons (195,000 values)."

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def save(fig, name, manifest, info, foot=FOOT):
    if foot:
        fig.text(0.01, -0.01, foot, fontsize=7.5, color=INK2, ha="left", va="top")
    for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
        fig.savefig(OUT / f"{name}.{ext}", **kw)
    plt.close(fig)
    manifest[name] = info
    print(f"  {name}")


def bar_labels(ax, bars, fmt="{:.3f}", horizontal=False):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.text(v, b.get_y() + b.get_height() / 2, " " + fmt.format(v), va="center", fontsize=8, color=INK)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=7.5, color=INK)


# -----------------------------------------------------------------
# Data loading and consistency checks
# -----------------------------------------------------------------
def load():
    pred = np.load(RUN / "test_predictions.npz")
    base = np.load(BASE)
    if not (np.array_equal(pred["city_index"], base["city_index"]) and
            np.array_equal(pred["origin_hour_since_epoch"], base["origin_hour_since_epoch"]) and
            np.allclose(pred["y"], base["y"], atol=1e-5)):
        raise SystemExit("Prediction and baseline keys differ; stopping.")
    d = {"y": pred["y"].astype(float), "mu": pred["epl_mean"].astype(float), "sigma": pred["epl_std"].astype(float),
         "city": pred["city_index"], "origin": pred["origin_hour_since_epoch"],
         "pred": {"model": pred["epl_mean"].astype(float),
                  "persistence": base["pred_persistence"].astype(float),
                  "recent_mean_24h": base["pred_recent_mean_24h"].astype(float),
                  "linear_extrapolation": base["pred_linear_extrapolation"].astype(float)}}
    d["precip"] = target_precip(d["city"], d["origin"])
    d["classes"] = {"Dry": d["precip"] == 0, "Light rain\n(0-2.5 mm/h)": (d["precip"] > 0) & (d["precip"] < 2.5),
                    "Heavy rain\n(>= 2.5 mm/h)": d["precip"] >= 2.5}
    with open(RUN / "metrics_test.json", encoding="utf-8") as f:
        saved = json.load(f)
    for k, p in d["pred"].items():                     # recomputed metrics must equal the saved ones
        m = metrics(p - d["y"])
        s = saved["forecasts"][k]["overall"]
        assert abs(m["RMSE"] - s["RMSE"]) < 1e-9 and abs(m["MAE"] - s["MAE"]) < 1e-9, k
    d["saved"] = saved
    d["log"] = pd.read_csv(RUN / "training_log.csv")
    with open(RUN / "run_config.json", encoding="utf-8") as f:
        d["run"] = json.load(f)
    d["tables"] = {c: pd.read_csv(config.PHYSICS_DIR / f"{c}_physics.csv", parse_dates=["valid_time"],
                                  usecols=["valid_time", "gas_attenuation_dB", "cloud_attenuation_dB",
                                           "rain_attenuation_dB", "epl_dB"]) for c in CITY_NAMES}
    return d


# -----------------------------------------------------------------
# Figures
# -----------------------------------------------------------------
def plot_training(d, man):
    log = d["log"]
    ep = log.epoch + 1
    best = int(ep[log.val_loss.idxmin()])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title in ((axes[0], "loss", "Total loss (weather MSE + EPL NLL)"),
                           (axes[1], "epl_loss", "EPL loss (Gaussian NLL, Eq. 10)")):
        ax.plot(ep, log[col], color=C_MODEL, lw=2, marker="o", ms=4, label="Training")
        ax.plot(ep, log[f"val_{col}"], color=C_PERS, lw=2, marker="s", ms=4, label="Validation")
        ax.axvline(best, color=INK2, ls="--", lw=1)
        ax.annotate(f"selected checkpoint\n(epoch {best}, lowest val. loss)", xy=(best, ax.get_ylim()[1]),
                    xytext=(4, -4), textcoords="offset points", va="top", fontsize=8, color=INK2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (lower is better)"); ax.set_title(title, loc="left")
        ax.set_xticks(ep)
    axes[0].legend(loc="center right")
    fig.suptitle("Training history: validation loss is lowest at epoch 5, then rises (overfitting); "
                 "early stopping restored epoch 5", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "01_training_curves", man, {
        "epochs_run": int(len(log)), "best_epoch": best, "best_val_loss": float(log.val_loss.min()),
        "train_epl_loss_epoch5_to_last": [float(log.epl_loss[best - 1]), float(log.epl_loss.iloc[-1])],
        "val_epl_loss_epoch5_to_last": [float(log.val_epl_loss[best - 1]), float(log.val_epl_loss.iloc[-1])]},
        foot=f"Source: {RUN.relative_to(ROOT).as_posix()}/training_log.csv. Adam 1e-4, batch 128, patience 6.")


def plot_timeseries(d, man):
    """Period A: the 14-day block (non-overlapping, per city, +1 h) whose LSTM RMSE is the median of all
    blocks. Period B: the 7 days centred on the largest true EPL in the test set. Fixed rules, no tuning."""
    blocks = []
    for c in range(len(CITY_NAMES)):
        rows = np.where(d["city"] == c)[0]
        for s in range(0, len(rows) - 336 + 1, 336):
            r = rows[s:s + 336]
            blocks.append((rmse(d["mu"][r, 0] - d["y"][r, 0]), c, s))
    blocks.sort()
    med_rmse, ca, sa = blocks[(len(blocks) - 1) // 2]
    imax = int(np.argmax(d["y"][:, 0]))
    cb = int(d["city"][imax])
    rows_b = np.where(d["city"] == cb)[0]
    sb = max(0, min(int(np.searchsorted(rows_b, imax)) - 84, len(rows_b) - 168))
    periods = [(ca, sa, 336, f"A. Typical period: {CITY_NAMES[ca]} (14-day block with median LSTM error)"),
               (cb, sb, 168, f"B. Largest event in the test set: {CITY_NAMES[cb]} (7 days)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharey="row")
    info = {}
    for i, (c, s, n, title) in enumerate(periods):
        rows = np.where(d["city"] == c)[0][s:s + n]
        for j, k in enumerate((1, 5)):
            ax = axes[i, j]
            t = (d["origin"][rows] + k).astype("datetime64[h]")
            y, mu, sg = d["y"][rows, k - 1], d["mu"][rows, k - 1], d["sigma"][rows, k - 1]
            ax.fill_between(t, mu - 1.96 * sg, mu + 1.96 * sg, color=C_MODEL, alpha=0.18, lw=0, label="LSTM 95 % interval")
            ax.plot(t, y, color=INK, lw=1.6, label="True EPL (ITU-R label)")
            ax.plot(t, d["pred"]["persistence"][rows, k - 1], color=C_PERS, lw=1.1, ls="--", label="Persistence")
            ax.plot(t, mu, color=C_MODEL, lw=1.6, label="LSTM mean")
            ax.set_title(f"{title}\nforecast +{k} h  |  RMSE LSTM {rmse(mu - y):.2f} dB, "
                         f"persistence {rmse(d['pred']['persistence'][rows, k - 1] - y):.2f} dB",
                         loc="left", fontsize=9)
            ax.set_ylabel("EPL (dB)") if j == 0 else None
            ax.tick_params(axis="x", rotation=0)
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
            info[f"period_{'AB'[i]}_h{k}"] = {"city": CITY_NAMES[c], "start": str(t[0]), "end": str(t[-1]),
                                              "rmse_lstm": rmse(mu - y), "rmse_persistence":
                                              rmse(d["pred"]["persistence"][rows, k - 1] - y)}
    axes[0, 0].legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, "02_timeseries_examples", man, {**info, "selection": "A: median-RMSE 14-day block over all cities "
         "(+1 h); B: 7 days centred on the largest true EPL at +1 h", "median_block_rmse": med_rmse})


def plot_scatter(d, man):
    y, mu = d["y"].ravel(), d["mu"].ravel()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, lim, title in ((axes[0], 26.5, "Full range (all 195,000 values)"),
                           (axes[1], 7.0, "Zoom 0-7 dB (axis limit only; no values removed)")):
        hb = ax.hexbin(y, mu, gridsize=90, extent=(0, lim, 0, lim), bins="log", mincnt=1, cmap="Blues",
                       linewidths=0.1)
        ax.plot([0, lim], [0, lim], color=C_PERS, lw=1.5, ls="--", label="y = x (perfect forecast)")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
        ax.set_xlabel("True EPL (dB)"); ax.set_ylabel("LSTM predicted EPL (dB)")
        ax.set_title(title, loc="left", fontsize=10)
        fig.colorbar(hb, ax=ax, shrink=0.85, label="Values per hexagon (log)")
    axes[0].legend(loc="upper left")
    out = float(100 * ((y > 7) | (mu > 7)).mean())
    axes[1].text(6.9, 0.2, f"{out:.1f} % of values lie beyond 7 dB", ha="right", fontsize=8, color=INK2)
    fig.suptitle("Predicted vs true EPL, all horizons pooled: good agreement in clear sky, "
                 "large under-prediction of rain peaks", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "03_pred_vs_true_scatter", man, {
        "pearson_r": float(np.corrcoef(y, mu)[0, 1]), "max_true": float(y.max()), "max_pred": float(mu.max()),
        "pct_beyond_7dB": out})


def plot_residuals(d, man):
    r = (d["mu"] - d["y"]).ravel()
    mu = d["mu"].ravel()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ax.hist(r, bins=np.arange(-25, 6.01, 0.1), color=C_MODEL, edgecolor="none")
    ax.set_yscale("log"); ax.set_xlabel("Residual = predicted - true (dB)"); ax.set_ylabel("Count (log scale)")
    ax.axvline(0, color=INK2, lw=1)
    q = np.percentile(r, [1, 50, 99])
    ax.set_title("Residual distribution", loc="left")
    ax.text(0.02, 0.97, f"median {q[1]:+.3f} dB\n1st / 99th pct {q[0]:+.2f} / {q[2]:+.2f} dB\n"
            f"long negative tail = missed rain peaks", transform=ax.transAxes, va="top", fontsize=8, color=INK2)
    ax = axes[1]
    hb = ax.hexbin(mu, r, gridsize=80, bins="log", mincnt=1, cmap="Blues", linewidths=0.1)
    ax.axhline(0, color=C_PERS, lw=1.3, ls="--")
    ax.set_xlabel("LSTM predicted EPL (dB)"); ax.set_ylabel("Residual (dB)")
    ax.set_title("Residuals vs predicted EPL", loc="left")
    fig.colorbar(hb, ax=ax, shrink=0.85, label="Values per hexagon (log)")
    fig.tight_layout()
    save(fig, "04_residuals", man, {"residual_median": float(q[1]), "residual_p1": float(q[0]),
                                    "residual_p99": float(q[2]), "residual_mean": float(r.mean()),
                                    "residual_min": float(r.min()), "residual_max": float(r.max())})


def plot_horizon(d, man):
    hrs = np.arange(1, H + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    info = {}
    for ax, key, title in ((axes[0], "RMSE_by_step", "RMSE"), (axes[1], "MAE_by_step", "MAE")):
        vals = {k: metrics(d["pred"][k] - d["y"])[key] for k in ("model", "persistence")}
        for k, v in vals.items():
            label, col = METHODS[k]
            other = vals["persistence" if k == "model" else "model"]
            ax.plot(hrs, v, color=col, lw=2, marker="o" if k == "model" else "s", ms=6, label=label)
            for h, val, o in zip(hrs, v, other):   # the higher value is labelled above, the lower below
                ax.annotate(f"{val:.3f}", (h, val), textcoords="offset points", xytext=(0, 8 if val >= o else -15),
                            ha="center", fontsize=7.5, color=INK2)
            info[f"{k}_{title}"] = v
        ax.set_xticks(hrs, [f"+{h} h" for h in hrs]); ax.set_ylim(0, None)
        ax.set_xlabel("Forecast horizon"); ax.set_ylabel(f"{title} (dB)"); ax.set_title(f"{title} by horizon", loc="left")
    axes[0].legend(loc="lower right")
    fig.suptitle("The LSTM's RMSE advantage grows with horizon; its MAE is worse than persistence at +1 h and +2 h",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "05_horizon_rmse_mae", man, info)


def plot_baselines(d, man):
    keys = list(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), sharey=True)
    info = {}
    for ax, stat in ((axes[0], "RMSE"), (axes[1], "MAE")):
        vals = [metrics(d["pred"][k] - d["y"])[stat] for k in keys]
        bars = ax.barh(range(len(keys)), vals, color=[METHODS[k][1] for k in keys], height=0.6)
        bar_labels(ax, bars, horizontal=True)
        ax.set_yticks(range(len(keys)), [METHODS[k][0] for k in keys])
        ax.set_xlabel(f"{stat} (dB, lower is better)"); ax.set_title(f"Overall {stat}", loc="left")
        ax.set_xlim(0, max(vals) * 1.18); ax.grid(axis="y", visible=False)
        info[stat] = dict(zip(keys, vals))
    axes[0].invert_yaxis()                     # shared y axis: invert once so the LSTM is on top
    fig.suptitle("All methods scored on identical keys: LSTM has the lowest RMSE (-15.6 % vs persistence), "
                 "MAE only 3.2 % lower", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "06_baseline_comparison", man, info)


def grouped(ax, groups, series_vals, fmt="{:.3f}"):
    n = len(series_vals)
    w = 0.8 / n
    x = np.arange(len(groups))
    for i, (label, col, vals) in enumerate(series_vals):
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.92, color=col, label=label)
        bar_labels(ax, bars, fmt)
    ax.set_xticks(x, groups); ax.grid(axis="x", visible=False)


def plot_city(d, man):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    info = {}
    for ax, stat in ((axes[0], "RMSE"), (axes[1], "MAE")):
        sv = []
        for k in ("model", "persistence"):
            vals = [metrics((d["pred"][k] - d["y"])[d["city"] == c])[stat] for c in range(len(CITY_NAMES))]
            sv.append((METHODS[k][0], METHODS[k][1], vals))
            info[f"{k}_{stat}"] = dict(zip(CITY_NAMES, vals))
        grouped(ax, CITY_NAMES, sv)
        ax.set_ylabel(f"{stat} (dB)"); ax.set_title(f"{stat} by city", loc="left")
        ax.set_ylim(0, max(max(v) for _, _, v in sv) * 1.15)
    axes[0].legend(loc="upper left")
    fig.suptitle("LSTM RMSE is lower than persistence in every city; MAE is slightly worse in Athens and Warsaw",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "07_city_comparison", man, info, foot=FOOT + " 6,500 origins per city.")


def plot_precip(d, man):
    names = list(d["classes"])
    groups = [f"{n}\nn = {int(m.sum()):,}" for n, m in d["classes"].items()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    info = {}
    for ax, stat in ((axes[0], "RMSE"), (axes[1], "MAE")):
        sv = []
        for k in ("model", "persistence"):
            vals = [metrics(d["pred"][k] - d["y"], m)[stat] for m in d["classes"].values()]
            sv.append((METHODS[k][0], METHODS[k][1], vals))
            info[f"{k}_{stat}"] = dict(zip([n.split("\n")[0] for n in names], vals))
        grouped(ax, groups, sv)
        ax.set_ylabel(f"{stat} (dB)"); ax.set_title(f"{stat} by precipitation at the target hour", loc="left")
        ax.set_ylim(0, max(max(v) for _, _, v in sv) * 1.15)
    axes[0].legend(loc="upper left")
    fig.suptitle("Heavy rain is the weak point: the LSTM is worse than persistence there (RMSE 6.70 vs 5.85 dB)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "08_precipitation_classes", man, info, foot=FOOT + " Classes use ERA5 precipitation observed at the "
         "target hour (outcome-based, diagnostic only). All values included.")


def coverage(d, mask, z):
    err = np.abs(d["y"] - d["mu"])[mask]
    return float(100 * (err <= z * d["sigma"][mask]).mean())


def plot_reliability(d, man):
    from scipy.stats import norm
    nominal = np.array([50, 60, 70, 80, 90, 95, 99])
    zs = norm.ppf(0.5 + nominal / 200)
    groups = [("All values", np.ones_like(d["y"], dtype=bool), INK, "o")]
    groups += [(n.replace("\n", " "), m, col, mk) for (n, m), col, mk in zip(d["classes"].items(), RAMP, ("s", "^", "D"))]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 100], [0, 100], color=MUTED, ls="--", lw=1.2, label="Perfect calibration")
    info = {}
    for name, m, col, mk in groups:
        cov = [coverage(d, m, z) for z in zs]
        ax.plot(nominal, cov, color=col, lw=2, marker=mk, ms=6, label=f"{name} (n = {int(m.sum()):,})")
        info[name] = dict(zip(nominal.tolist(), cov))
    ax.set_xlim(45, 100); ax.set_ylim(0, 102); ax.set_aspect("auto")
    ax.set_xlabel("Nominal coverage of the Gaussian interval (%)"); ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Reliability of the predicted uncertainty", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8.5)
    ax.text(1.04, 0.45, "Above the diagonal: intervals too wide\n(under-confident)\n\n"
            "Below the diagonal: intervals too narrow\n(over-confident)", transform=ax.transAxes,
            fontsize=8.5, color=INK2, va="top")
    save(fig, "09_calibration_reliability", man, {**info, "z_values": zs.tolist()},
         foot="Interval: mu +- z * sigma, z from the normal quantile. 195,000 test values; classes as in Fig. 8.")


def plot_interval_groups(d, man):
    z = 1.959964
    bins = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 100)]
    epl_groups = [(f"{lo}-{hi} dB" if hi < 100 else f">= {lo} dB", (d["y"] >= lo) & (d["y"] < hi)) for lo, hi in bins]
    prec_groups = [(n, m) for n, m in d["classes"].items()]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7))
    info = {}
    for row, (gname, groups) in enumerate((("true EPL range", epl_groups), ("precipitation class", prec_groups))):
        labels = [f"{n}\nn = {int(m.sum()):,}" for n, m in groups]
        cov = [coverage(d, m, z) for _, m in groups]
        width = [float((2 * z * d["sigma"][m]).mean()) for _, m in groups]
        b = axes[row, 0].bar(labels, cov, color=C_MODEL, width=0.6)
        for bar, v in zip(b, cov):             # centred inside the bar, so labels never touch the 95 % line
            axes[row, 0].text(bar.get_x() + bar.get_width() / 2, v / 2, f"{v:.1f} %", ha="center", va="center",
                              fontsize=8, color="white", fontweight="bold")
        axes[row, 0].axhline(95, color=C_PERS, ls="--", lw=1.3, label="Nominal 95 %")
        axes[row, 0].legend(loc="upper right", bbox_to_anchor=(1.0, 1.12), fontsize=8)
        axes[row, 0].set_ylim(0, 112); axes[row, 0].set_ylabel("Empirical coverage (%)")
        axes[row, 0].set_title(f"95 % interval coverage by {gname}", loc="left", fontsize=10)
        b = axes[row, 1].bar(labels, width, color=C_MODEL, width=0.6)
        bar_labels(axes[row, 1], b, "{:.2f}")
        axes[row, 1].set_ylabel("Mean 95 % interval width (dB)")
        axes[row, 1].set_ylim(0, max(width) * 1.15)
        axes[row, 1].set_title(f"95 % interval width by {gname}", loc="left", fontsize=10)
        for ax in axes[row]:
            ax.grid(axis="x", visible=False)
        info[gname] = {n.replace("\n", " "): {"coverage_pct": c, "mean_width_dB": w}
                       for (n, _), c, w in zip(groups, cov, width)}
    fig.suptitle("Intervals widen with attenuation, but not enough: coverage collapses for EPL >= 5 dB and heavy rain",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "10_interval_width_coverage", man, info, foot=FOOT + " Groups use the true outcome (diagnostic only).")


def plot_epl_distribution(d, man):
    full = np.concatenate([t.epl_dB.to_numpy() for t in d["tables"].values()])
    test = d["y"].ravel()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    ax.hist(full, bins=np.arange(0, 26.01, 0.25), color=C_MODEL, edgecolor="white", linewidth=0.3)
    ax.set_yscale("log"); ax.set_xlabel("EPL (dB)"); ax.set_ylabel("Hours (log scale)")
    ax.set_title(f"All hourly labels, 6 cities, 2020-2024 (n = {full.size:,})", loc="left", fontsize=10)
    p99 = np.percentile(full, 99)
    ax.axvspan(6.2, 6.7, color=MUTED, alpha=0.35, lw=0)
    ax.text(6.9, ax.get_ylim()[1] * 0.4, "approx. largest true EPL visible\nin paper Figs. 3-4 (6.2-6.7 dB,\n"
            "read by eye; different dataset)", fontsize=7.5, color=INK2, va="top")
    ax.axvline(p99, color=C_PERS, lw=1.3, ls="--")
    ax.text(p99 + 0.3, ax.get_ylim()[1] * 0.004, f"dashed: our 99th percentile = {p99:.2f} dB",
            color=INK, fontsize=7.5, ha="left")
    ax = axes[1]
    xs = np.linspace(0, 26, 1041)
    for vals, label, col, ls in ((full, "All hourly labels 2020-2024", C_MODEL, "-"),
                                 (test, "Causal test targets (195,000)", C_PERS, "--")):
        s = np.sort(vals)
        ax.plot(xs, 100 * (1 - np.searchsorted(s, xs, side="right") / s.size), color=col, lw=2, ls=ls, label=label)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 100)
    ax.set_xlabel("EPL threshold (dB)"); ax.set_ylabel("Share of values above threshold (%, log)")
    ax.set_title("Exceedance: the extreme tail", loc="left", fontsize=10)
    info = {}
    for t in (2, 5, 6.7, 10, 20):
        pct = float(100 * (test > t).mean())
        info[f"test_pct_above_{t}dB"] = pct
        ax.plot(t, pct, "o", color=INK, ms=4)
        ax.annotate(f"> {t} dB: {pct:.2f} %" if pct >= 0.01 else f"> {t} dB: {pct:.3f} %", (t, pct),
                    textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=INK)
    ax.legend(loc="upper right")
    fig.suptitle("EPL labels are dominated by clear sky (median 0.91 dB) with a long rain tail up to 25.6 dB",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "11_epl_distribution", man, {**info, "full_p50": float(np.median(full)), "full_p99": float(p99),
         "full_max": float(full.max()), "test_max": float(test.max())},
         foot="Labels: ITU-R gas (P.676) + cloud (P.840, LWP proxy) + rain (P.838/P.618) at 39 GHz, 30 deg, "
              "from ERA5 hourly weather (data/processed/physics_paper/).")


def plot_components(d, man):
    t = d["tables"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw={"width_ratios": [1.4, 1, 1, 1]})
    ax = axes[0]
    bottom = np.zeros(len(CITY_NAMES))
    info = {}
    for col, name, color in (("gas_attenuation_dB", "Gas", C_GAS), ("cloud_attenuation_dB", "Cloud", C_CLOUD),
                             ("rain_attenuation_dB", "Rain", C_RAIN)):
        v = np.array([t[c][col].mean() for c in CITY_NAMES])
        ax.bar(CITY_NAMES, v, bottom=bottom, color=color, label=name, width=0.65, edgecolor="white", linewidth=1)
        for i, (b, val) in enumerate(zip(bottom, v)):
            if val < 0.08:                     # too thin for a readable label; value is in the manifest
                continue
            ax.text(i, b + val / 2, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if name == "Rain" else INK)
        bottom += v
        info[f"mean_{name.lower()}_dB"] = dict(zip(CITY_NAMES, v.tolist()))
    ax.set_ylabel("Mean attenuation (dB)"); ax.set_title("Mean EPL composition by city", loc="left", fontsize=10)
    ax.tick_params(axis="x", rotation=35); ax.grid(axis="x", visible=False); ax.legend(loc="upper left", fontsize=8)
    allv = {c: np.concatenate([t[x][c].to_numpy() for x in CITY_NAMES])
            for c in ("gas_attenuation_dB", "cloud_attenuation_dB", "rain_attenuation_dB")}
    for ax, col, name, color, rng in ((axes[1], "gas_attenuation_dB", "Gas", C_GAS, (0.4, 1.4)),
                                      (axes[2], "cloud_attenuation_dB", "Cloud", C_CLOUD, (0, 5.5)),
                                      (axes[3], "rain_attenuation_dB", "Rain (rainy hours only)", C_RAIN, (0, 22))):
        v = allv[col]
        if col.startswith("rain"):
            dry = float(100 * (v == 0).mean())
            v = v[v > 0]
            ax.text(0.97, 0.95, f"{dry:.1f} % of hours have\nzero rain attenuation\n(not shown)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
            info["pct_hours_zero_rain"] = dry
        ax.hist(v, bins=np.linspace(*rng, 61), color=color, edgecolor="white", linewidth=0.3)
        ax.set_yscale("log"); ax.set_xlabel(f"{name.split(' ')[0]} attenuation (dB)")
        ax.set_title(name, loc="left", fontsize=10); ax.set_ylabel("Hours (log)") if ax is axes[1] else None
        info[f"{name.split(' ')[0].lower()}_range_dB"] = [float(allv[col].min()), float(allv[col].max())]
    fig.suptitle("Gas sets a stable ~0.75 dB floor; rain is rare but causes almost all of the variability",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "12_attenuation_components", man, info,
         foot="All hourly labels, 6 cities, 2020-2024 (263,088 hours), from data/processed/physics_paper/<city>_physics.csv.")


def plot_paper_context(d, man):
    mb = pd.read_csv(MINUTE)
    minute = mb[mb.reading.str.startswith("minute")]
    pooled = {b: float(np.sqrt((g.rmse_avg ** 2).mean())) for b, g in minute.groupby("baseline")}
    ours = {k: metrics(d["pred"][k] - d["y"])["RMSE"] for k in METHODS}
    rows = [  # (group, label, value, colour, marker)
        ("Paper (El-Sayed et al. 2026, Table 7)", "Multi-task LSTM, reported", PAPER_RMSE, MUTED, "D"),
        ("Our labels, minute-interpolated reading\n(+1..+5 min ahead; interpolation leaks\nthe next hour, so not real forecasting)",
         "Linear extrapolation", pooled["linear_extrapolation"], C_LIN, "o"),
        ("", "Persistence", pooled["persistence"], C_PERS, "o"),
        ("Our labels, causal hourly protocol\n(+1..+5 h ahead; this work)", "LSTM (seed 42)", ours["model"], C_MODEL, "o"),
        ("", "Persistence", ours["persistence"], C_PERS, "o"),
        ("", "Recent 24-h mean", ours["recent_mean_24h"], C_MEAN, "o"),
        ("", "Linear extrapolation", ours["linear_extrapolation"], C_LIN, "o"),
    ]
    ypos = [0, 2.2, 2.9, 4.6, 5.3, 6.0, 6.7]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for (grp, label, v, col, mk), yv in zip(rows, ypos):
        ax.plot([0.005, v], [yv, yv], color=GRID, lw=1, zorder=1)
        ax.plot(v, yv, marker=mk, color=col, ms=9, zorder=3, markeredgecolor="white")
        ax.text(v * 1.12, yv, f"{v:.3f} dB", va="center", fontsize=9, color=INK)
        ax.text(0.0047, yv, label + "  ", va="center", ha="right", fontsize=9, color=INK)
    for grp, yv in zip([r[0] for r in rows], ypos):
        if grp:
            ax.text(0.0047, yv - 0.4, grp, va="bottom", ha="right", fontsize=8.5, color=INK2, fontweight="bold")
    for yb in (0.55, 3.4):                     # separators between the three task groups
        ax.axhline(yb, color=MUTED, lw=0.8)
    ax.set_xscale("log"); ax.set_xlim(0.005, 6); ax.set_ylim(7.1, -0.9)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("RMSE (dB, logarithmic scale)")
    ax.set_title("Different tasks, different labels: these values are NOT a controlled comparison", loc="left",
                 fontsize=11, fontweight="bold")
    fig.text(0.01, -0.02,
             "The paper's dataset is not available and its forecast step (minutes vs hours), rain-label method and "
             "overall-RMSE definition are ambiguous\n(docs/WHY_RESULTS_DIFFER_FROM_PAPER.md). Its 0.026 dB lies in the "
             "range of trivial baselines on a minute-interpolated series, not near hourly forecasting.\n"
             "Minute-reading values: pooled over 6 cities x 78,914 windows (results/replication_audit/"
             "minute_protocol_baselines.csv). Hourly values: 195,000 test values.",
             fontsize=7.5, color=INK2, va="top")
    save(fig, "13_paper_context", man, {"paper_reported": PAPER_RMSE, "minute_reading_pooled": pooled,
                                        "causal_hourly": ours}, foot=None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--overwrite", action="store_true", help="replace this script's own files in results/figures/")
    p.add_argument("--export-docs", action="store_true", help="copy key PNGs to docs/assets/figures/causal/")
    args = p.parse_args()
    if OUT.exists() and not args.overwrite:
        raise SystemExit(f"{OUT} exists; pass --overwrite to regenerate its figures.")
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading artefacts and checking metrics against metrics_test.json ...")
    d = load()
    man = {}
    for f in (plot_training, plot_timeseries, plot_scatter, plot_residuals, plot_horizon, plot_baselines,
              plot_city, plot_precip, plot_reliability, plot_interval_groups, plot_epl_distribution,
              plot_components, plot_paper_context):
        f(d, man)
    with open(OUT / "figures_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"source_run": str(RUN.relative_to(ROOT)), "sampling": "none; all test values used",
                   "figures": man}, f, indent=2, default=float)
    missing = [n for n in man for ext in ("png", "pdf") if not (OUT / f"{n}.{ext}").stat().st_size]
    if missing:
        raise SystemExit(f"Empty figure files: {missing}")
    print(f"{len(man)} figures written to {OUT.relative_to(ROOT)}")
    if args.export_docs:
        DOCS.mkdir(parents=True, exist_ok=True)
        for n in DOC_FIGURES:
            shutil.copy2(OUT / f"{n}.png", DOCS / f"{n}.png")
        print(f"Copied {len(DOC_FIGURES)} PNGs to {DOCS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
