"""All result figures (paper Figs. 3-7 and supporting plots), PNG + PDF.

Reads only saved outputs (results/paper/*.csv, *.json, *.npz and
models/paper/*/training_log.csv); run the evaluation scripts first.
Figures for experiments that have not finished are skipped.

Usage:
    python evaluation/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from data_pipeline.cities import CITIES, UNSEEN_CITIES

RES = config.RESULTS_DIR
FIG = RES / "figures"
MODELS = ROOT / "models" / "paper"
H = config.HORIZON
# Fixed categorical order (never cycled): cities, then model/baseline roles.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CITY_COLORS = dict(zip(CITIES, SLOTS))
ROLE = {"model": "#2a78d6", "persistence": "#eb6834", "climatology": "#1baf7a",
        "linear_extrapolation": "#eda100", "true": "#333333"}
VARIANT_COLORS = dict(zip(["multitask", "no_itu", "epl_only", "weather_only", "multitask_msl",
                           "minute_multitask", "multitask_fig1heads"], SLOTS))
REGIONS = {"Southern": ["Rome", "Madrid", "Marseille"], "Western": ["London", "Paris"],
           "Northern": ["Oslo", "Stockholm"], "Eastern": ["Berlin", "Bucharest"], "Alpine": ["Davos"]}
plt.rcParams.update({"axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.titlesize": 9.5})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=300)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("saved", name)


def predictions(variant):
    path = RES / f"test_predictions_{variant}.csv"
    return pd.read_csv(path, parse_dates=["anchor_time"]) if path.exists() else None


def metrics():
    return json.load(open(RES / "metrics_test.json", encoding="utf-8"))


# ------------------------------------------------------------------ Fig. 3
def fig3(p):
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    for city, color in CITY_COLORS.items():
        g = p[p.city == city]
        ax.scatter(g.true_epl_dB, g.pred_mean_dB, s=3, alpha=0.4, color=color, label=city, linewidths=0)
    top = max(p.true_epl_dB.max(), p.pred_mean_dB.max())
    ax.plot([0, top], [0, top], color="#333", lw=1, ls="--", label="y = x")
    ax.set_xlabel("True EPL (dB)"); ax.set_ylabel("Predicted EPL (dB)")
    ax.set_title("Fig. 3  Predicted vs true EPL, test set, hours 1-5")
    ax.legend(fontsize=7, markerscale=3)
    save(fig, "fig3_pred_vs_true")


# ------------------------------------------------------------------ Fig. 4
def fig4(p, city="Chilbolton", n=300):
    """Paper layout: one panel per forecast hour, consecutive test samples."""
    g = p[p.city == city]
    anchors = g.anchor_time.drop_duplicates().iloc[:n]
    fig, axes = plt.subplots(H, 1, figsize=(8, 10), sharex=True)
    for h, ax in enumerate(axes, start=1):
        s = g[(g.step == h) & g.anchor_time.isin(anchors)]
        x = np.arange(len(s))
        ax.fill_between(x, s.pred_mean_dB - 1.96 * s.pred_std_dB, s.pred_mean_dB + 1.96 * s.pred_std_dB,
                        color=ROLE["model"], alpha=0.18, lw=0, label="95% interval")
        ax.plot(x, s.true_epl_dB, color=ROLE["true"], lw=1, label="True")
        ax.plot(x, s.pred_mean_dB, color=ROLE["model"], lw=1.2, label="Predicted")
        ax.set_ylabel("EPL (dB)"); ax.set_title(f"Hour +{h}", fontsize=8.5)
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[-1].set_xlabel(f"Test sample ({city}, first {n} test windows in time order; windows are not hourly-consecutive)")
    fig.suptitle(f"Fig. 4  EPL forecasts with 95% intervals, hours +1 to +5 ({city})", fontsize=10)
    save(fig, "fig4_horizon_timeseries")


def fig4_cities(p, n=150):
    fig, axes = plt.subplots(len(CITIES), 1, figsize=(8, 11))
    for ax, (city, color) in zip(axes, CITY_COLORS.items()):
        s = p[(p.city == city) & (p.step == 1)].iloc[:n]
        t = s.anchor_time + pd.Timedelta(hours=1)
        ax.fill_between(t, s.pred_mean_dB - 1.96 * s.pred_std_dB, s.pred_mean_dB + 1.96 * s.pred_std_dB,
                        color=color, alpha=0.2, lw=0, label="95% interval")
        ax.plot(t, s.true_epl_dB, color=ROLE["true"], lw=1, label="True")
        ax.plot(t, s.pred_mean_dB, color=color, lw=1.4, label="Predicted")
        ax.set_ylabel("EPL (dB)"); ax.set_title(f"{city}: hour +1, first {n} test samples", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].legend(fontsize=7, ncol=3)
    save(fig, "fig4_city_timeseries_hour1")


def fig4b(p):
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for ax, (city, color) in zip(axes.ravel(), CITY_COLORS.items()):
        g = p[p.city == city]
        anchor = g.loc[g.true_epl_dB.idxmax(), "anchor_time"]
        s = g[g.anchor_time == anchor]
        ax.fill_between(s.step, s.pred_mean_dB - 1.96 * s.pred_std_dB, s.pred_mean_dB + 1.96 * s.pred_std_dB,
                        color=color, alpha=0.2, lw=0)
        ax.plot(s.step, s.true_epl_dB, "o-", color=ROLE["true"], lw=1, ms=4, label="True")
        ax.plot(s.step, s.pred_mean_dB, "s-", color=color, lw=1.5, ms=4, label="Predicted")
        ax.set_title(f"{city}, start {anchor:%Y-%m-%d %H}h", fontsize=8); ax.set_xlabel("Forecast hour")
        ax.set_xticks(range(1, H + 1))
    axes[0, 0].set_ylabel("EPL (dB)"); axes[1, 0].set_ylabel("EPL (dB)"); axes[0, 0].legend(fontsize=7)
    fig.suptitle("Largest-EPL test window per city: 5-h forecast with 95% interval", fontsize=9.5)
    save(fig, "fig4b_horizon_examples")


# ------------------------------------------------------------------ Figs. 5-7
def unseen_figures(suffix=""):
    path = RES / f"table8_unseen_cities{suffix}.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    cities = list(UNSEEN_CITIES)
    by = {m: t[t["mode"] == m].set_index("city") for m in t["mode"].unique()}
    tag = suffix.lstrip("_") or "multitask"

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    x = np.arange(len(cities)); w = 0.2
    series = [("True (ITU-R)", None, ROLE["true"]), ("Synthesized, weighted (Eq. 12)", "synthesized", SLOTS[0]),
              ("Synthesized, median (Fig. 2)", "synthesized_median", SLOTS[6]), ("Own inputs (extra)", "own_inputs", SLOTS[1])]
    for i, (label, mode, color) in enumerate(series):
        vals = by["synthesized"].loc[cities, "true_EPL_mean_dB"] if mode is None else by[mode].loc[cities, "pred_EPL_mean_dB"]
        axes[0].bar(x + (i - 1.5) * w, vals, w * 0.9, color=color, label=label)
    axes[0].set_ylabel("Mean EPL (dB)"); axes[0].set_xticks(x, cities)
    axes[0].set_title("Mean EPL over Apr-Dec 2024: ITU-R reference vs model predictions"); axes[0].legend(fontsize=7, ncol=2)
    for i, (label, mode, color) in enumerate(series[1:] + [("Persistence of own EPL", "persistence_own", SLOTS[3])]):
        axes[1].bar(x + (i - 1.5) * w, by[mode].loc[cities, "MAE_dB"], w * 0.9, color=color, label=label)
    axes[1].set_ylabel("Hourly MAE (dB)"); axes[1].set_xticks(x, cities)
    axes[1].set_title("Hour-by-hour absolute error, forecast hours 1-5"); axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Fig. 5  Ten unseen cities ({tag})", fontsize=10)
    save(fig, f"fig5_unseen_cities{suffix}")

    s = by["synthesized"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    names = list(REGIONS)
    ax.bar(np.arange(len(names)) - 0.2, [s.loc[REGIONS[r], "abs_error_of_means_dB"].mean() for r in names], 0.38,
           color=SLOTS[2], label="|mean pred - mean true| (paper's measure)")
    ax.bar(np.arange(len(names)) + 0.2, [s.loc[REGIONS[r], "MAE_dB"].mean() for r in names], 0.38,
           color=SLOTS[0], label="Hourly MAE")
    ax.set_xticks(np.arange(len(names)), names); ax.set_ylabel("Error (dB)")
    ax.set_title(f"Fig. 6  Unseen-city error by region, synthesized inputs ({tag})"); ax.legend(fontsize=7.5)
    save(fig, f"fig6_error_by_region{suffix}")

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sc = ax.scatter([UNSEEN_CITIES[c]["longitude"] for c in cities], [UNSEEN_CITIES[c]["latitude"] for c in cities],
                    c=s.loc[cities, "MAE_dB"], cmap="Blues", vmin=0, s=110, edgecolors="#333", linewidths=0.6,
                    label="Unseen city")
    ax.scatter([v["longitude"] for v in CITIES.values()], [v["latitude"] for v in CITIES.values()],
               marker="^", s=80, color=SLOTS[1], edgecolors="white", linewidths=1, label="Training city")
    for c in cities:
        ax.annotate(f"{c}\n{s.loc[c, 'MAE_dB']:.2f} dB", (UNSEEN_CITIES[c]["longitude"], UNSEEN_CITIES[c]["latitude"]),
                    fontsize=6.5, xytext=(5, 3), textcoords="offset points")
    for c, v in CITIES.items():
        ax.annotate(c, (v["longitude"], v["latitude"]), fontsize=6.5, color="#a14512", xytext=(5, -9),
                    textcoords="offset points")
    fig.colorbar(sc, ax=ax, label="Hourly MAE (dB), synthesized inputs")
    ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Latitude (deg)"); ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"Fig. 7  Zero-shot locations and hourly MAE ({tag})")
    save(fig, f"fig7_unseen_map{suffix}")

    npz = RES / f"unseen_series{suffix}.npz"
    if npz.exists():
        d = np.load(npz)
        fig, axes = plt.subplots(5, 2, figsize=(10, 10))
        for ax, city in zip(axes.ravel(), cities):
            n = 240
            t = pd.date_range(str(d[f"{city}_start"][0]), periods=n, freq="h") + pd.Timedelta(hours=1)
            ax.plot(t, d[city][0, :n], color=ROLE["true"], lw=1, label="True (ITU-R)")
            ax.plot(t, d[city][1, :n], color=SLOTS[0], lw=1.2, label="Synthesized inputs")
            ax.plot(t, d[city][2, :n], color=SLOTS[1], lw=1.2, label="Own inputs")
            ax.set_title(city, fontsize=8); ax.tick_params(axis="x", labelsize=6, rotation=30)
        axes[0, 0].legend(fontsize=6)
        fig.suptitle(f"Unseen cities: hour +1 EPL (dB), first 10 days of evaluation ({tag})", fontsize=9.5)
        save(fig, f"unseen_timeseries{suffix}")


# ------------------------------------------------------------------ supporting
def training_curves():
    runs = [v for v in VARIANT_COLORS if (MODELS / v / "training_log.csv").exists()]
    cols = 3
    rows = int(np.ceil(len(runs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows), squeeze=False)
    for ax, v in zip(axes.ravel(), runs):
        log = pd.read_csv(MODELS / v / "training_log.csv")
        ep = log.epoch + 1
        ax.plot(ep, log.loss, color=VARIANT_COLORS[v], lw=1.5, label="train")
        ax.plot(ep, log.val_loss, color=VARIANT_COLORS[v], lw=1.5, ls="--", label="validation")
        best = int(log.val_loss.idxmin())
        ax.axvline(best + 1, color="#999", lw=0.8, ls=":")
        ax.set_title(f"{v} (best epoch {best + 1})"); ax.set_xlabel("Epoch"); ax.set_ylabel("Total loss")
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(runs):]:
        ax.axis("off")
    fig.suptitle("Training and validation loss (dotted line: restored best epoch)", fontsize=10)
    save(fig, "training_curves")


def residuals(p):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    e = p.pred_mean_dB - p.true_epl_dB
    axes[0].hist(e, bins=200, color=ROLE["model"], label="Multi-task LSTM")
    axes[0].hist(p.persistence_dB - p.true_epl_dB, bins=200, histtype="step", color=ROLE["persistence"],
                 lw=1.2, label="Persistence")
    axes[0].set_yscale("log"); axes[0].set_xlabel("Prediction - truth (dB)"); axes[0].set_ylabel("Count (log)")
    axes[0].set_title("Error distribution, test set"); axes[0].legend(fontsize=7.5)
    axes[1].scatter(p.true_epl_dB, e, s=2, alpha=0.3, color=ROLE["model"], linewidths=0)
    axes[1].axhline(0, color="#333", lw=0.8)
    axes[1].set_xlabel("True EPL (dB)"); axes[1].set_ylabel("Prediction - truth (dB)")
    axes[1].set_title("Error vs EPL level (under-prediction of rain peaks)")
    save(fig, "residuals")


def error_vs_horizon(m):
    base = m["datasets"]["paper"]["baselines"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    steps = np.arange(1, H + 1)
    for name in ("multitask", "multitask_msl", "multitask_fig1heads", "no_itu", "epl_only"):
        if name in m["variants"] and "RMSE_by_step" in m["variants"][name]:
            ax.plot(steps, m["variants"][name]["RMSE_by_step"], "o-", color=VARIANT_COLORS[name], lw=2, ms=5,
                    label=f"LSTM: {name}")
    for name in ("persistence", "climatology"):
        ax.plot(steps, base[name]["RMSE_by_step"], "s--", color=ROLE[name], lw=1.5, ms=4, label=name)
    ax.set_xticks(steps); ax.set_xlabel("Forecast hour"); ax.set_ylabel("RMSE (dB)")
    ax.set_title("Test RMSE by forecast hour, six cities"); ax.legend(fontsize=7)
    save(fig, "error_vs_horizon")


def by_city(m):
    tables = {"LSTM (multitask)": RES / "table7_rmse_multitask.csv",
              "Persistence": RES / "table7_rmse_paper_persistence.csv",
              "Climatology": RES / "table7_rmse_paper_climatology.csv"}
    colors = [ROLE["model"], ROLE["persistence"], ROLE["climatology"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, label in ((axes[0], "average_RMSE", "RMSE (dB)"), (axes[1], "MAE", "MAE (dB)")):
        x = np.arange(len(CITIES) + 1)
        for i, ((name, path), c) in enumerate(zip(tables.items(), colors)):
            t = pd.read_csv(path)
            ax.bar(x + (i - 1) * 0.27, t[col], 0.25, color=c, label=name)
        ax.set_xticks(x, list(CITIES) + ["Overall"], rotation=30); ax.set_ylabel(label)
        ax.set_title(f"Test {label.split()[0]} by location, hours 1-5")
    axes[0].legend(fontsize=7)
    save(fig, "rmse_mae_by_city")

    t = pd.read_csv(tables["LSTM (multitask)"])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for city, color in CITY_COLORS.items():
        r = t[t.city == city].iloc[0]
        ax.plot(range(1, H + 1), [r[f"step_{h}"] for h in range(1, H + 1)], "o-", color=color, lw=1.6, ms=4, label=city)
    ax.set_xticks(range(1, H + 1)); ax.set_xlabel("Forecast hour"); ax.set_ylabel("RMSE (dB)")
    ax.set_title("Table 7 as a plot: LSTM RMSE per city and hour"); ax.legend(fontsize=7, ncol=2)
    save(fig, "rmse_by_city_and_hour")


def calibration(p, m):
    z = np.abs((p.true_epl_dB - p.pred_mean_dB) / p.pred_std_dB)
    nominal = np.linspace(0.05, 0.99, 40)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(nominal, [np.mean(z <= norm.ppf(0.5 + q / 2)) for q in nominal], color=ROLE["model"], lw=2,
                 label="Multi-task LSTM")
    axes[0].plot([0, 1], [0, 1], color="#333", lw=1, ls="--", label="Ideal")
    axes[0].set_xlabel("Nominal coverage"); axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("Interval calibration, test set"); axes[0].legend(fontsize=7.5)
    pc = m["variants"]["multitask"]["per_city_uncertainty"]
    x = np.arange(len(CITIES))
    axes[1].bar(x, [pc[c]["PICP_95_pct"] for c in CITIES], 0.6, color=[CITY_COLORS[c] for c in CITIES])
    axes[1].axhline(95, color="#333", lw=1, ls="--", label="Nominal 95%")
    axes[1].set_ylim(80, 100); axes[1].set_xticks(x, list(CITIES), rotation=30)
    axes[1].set_ylabel("PICP (%)"); axes[1].set_title("95% interval coverage by city"); axes[1].legend(fontsize=7.5)
    save(fig, "calibration")


def weather_head(m):
    wm = m["variants"]["multitask"]["weather"]
    wp = m["datasets"]["paper"]["baselines"]["weather_persistence"]
    units = {"temperature_C": "degC", "relative_humidity_percent": "%", "pressure_hPa": "hPa",
             "precipitation_mm": "mm", "cloud_cover_percent": "%"}
    fig, axes = plt.subplots(1, 5, figsize=(13, 3))
    for ax, (name, unit) in zip(axes, units.items()):
        ax.plot(range(1, H + 1), wm[f"RMSE_{name}_by_step"], "o-", color=ROLE["model"], lw=2, ms=4, label="LSTM")
        ax.plot(range(1, H + 1), wp[f"RMSE_{name}_by_step"], "s--", color=ROLE["persistence"], lw=1.5, ms=4,
                label="Persistence")
        ax.set_title(name.replace("_", " ")); ax.set_xlabel("Forecast hour"); ax.set_ylabel(f"RMSE ({unit})")
        ax.set_xticks(range(1, H + 1))
    axes[0].legend(fontsize=7)
    fig.suptitle("Weather head, test RMSE in physical units", fontsize=10)
    save(fig, "weather_head_rmse")


def baselines_and_conditions(m):
    v, b = m["variants"]["multitask"], m["datasets"]["paper"]["baselines"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    names = ["LSTM (multitask)", "Persistence", "Climatology"]
    rm = [v["RMSE"], b["persistence"]["RMSE"], b["climatology"]["RMSE"]]
    ma = [v["MAE"], b["persistence"]["MAE"], b["climatology"]["MAE"]]
    x = np.arange(3)
    axes[0].bar(x - 0.2, rm, 0.38, color=SLOTS[0], label="RMSE")
    axes[0].bar(x + 0.2, ma, 0.38, color=SLOTS[2], label="MAE")
    for i in range(3):
        axes[0].text(x[i] - 0.2, rm[i], f"{rm[i]:.3f}", ha="center", va="bottom", fontsize=7)
        axes[0].text(x[i] + 0.2, ma[i], f"{ma[i]:.3f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_xticks(x, names); axes[0].set_ylabel("dB"); axes[0].set_title("Model vs baselines, test set")
    axes[0].legend(fontsize=7.5)
    cond = v["by_condition"]
    keys = list(cond)
    x = np.arange(len(keys))
    axes[1].bar(x - 0.2, [cond[k]["RMSE"] for k in keys], 0.38, color=ROLE["model"], label="LSTM")
    axes[1].bar(x + 0.2, [cond[k]["persistence_RMSE"] for k in keys], 0.38, color=ROLE["persistence"],
                label="Persistence")
    axes[1].set_yscale("log"); axes[1].set_ylabel("RMSE (dB, log)")
    axes[1].set_xticks(x, [f"{k.split(' (')[0]}\n({cond[k]['share_pct']:.1f}% of hours)" for k in keys], fontsize=7)
    axes[1].set_title("Error by weather at the target hour"); axes[1].legend(fontsize=7.5)
    save(fig, "baselines_and_conditions")


def ablations(m):
    v = m["variants"]
    rows = [(n, v[n]) for n in ("multitask", "epl_only", "no_itu", "multitask_fig1heads", "multitask_msl")
            if n in v and "RMSE" in v[n]]
    if len(rows) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    x = np.arange(len(rows))
    colors = [VARIANT_COLORS[n] for n, _ in rows]
    for ax, key, label in ((axes[0], "RMSE", "EPL RMSE (dB)"), (axes[1], "MAE", "EPL MAE (dB)")):
        vals = [r[key] for _, r in rows]
        ax.bar(x, vals, 0.6, color=colors)
        for i, val in enumerate(vals):
            ax.text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, [n for n, _ in rows], rotation=30, ha="right"); ax.set_ylabel(label)
    axes[0].set_title("EPL accuracy by variant"); axes[1].set_title("EPL MAE by variant")
    wrows = [(n, v[n]) for n in ("multitask", "weather_only", "no_itu", "multitask_fig1heads", "multitask_msl")
             if n in v and "weather" in v[n]]
    axes[2].bar(np.arange(len(wrows)), [r["weather"]["RMSE_standardised"] for _, r in wrows], 0.6,
                color=[VARIANT_COLORS[n] for n, _ in wrows])
    axes[2].set_xticks(np.arange(len(wrows)), [n for n, _ in wrows], rotation=30, ha="right")
    axes[2].set_ylabel("Weather RMSE (standardised)"); axes[2].set_title("Weather accuracy by variant")
    fig.suptitle("Ablations (Tables 9 and 10) and extra variants, six-city test set", fontsize=10)
    save(fig, "ablations")


def seed_variation(m):
    """Every completed seed per variant (results/paper/seed_runs.csv from summarize_runs.py)."""
    path = config.RESULTS_DIR / "seed_runs.csv"
    if not path.exists():
        return
    runs = pd.read_csv(path)
    runs = runs[(runs.status == "complete") & runs.EPL_RMSE_dB.notna()]
    names = [n for n in ("multitask", "epl_only", "no_itu", "multitask_fig1heads") if (runs.variant == n).sum() > 1]
    if not names:
        return
    pers = m["datasets"]["paper"]["baselines"]["persistence"]["RMSE_by_step"][0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col, title in ((axes[0], "EPL_RMSE_dB", "Average over hours 1-5"),
                           (axes[1], "EPL_RMSE_hour1_dB", "Hour 1")):
        for i, n in enumerate(names):
            vals = runs.loc[runs.variant == n, col].to_numpy()
            ax.scatter(np.full(len(vals), i), vals, s=40, color=VARIANT_COLORS[n], zorder=3,
                       edgecolor="white", linewidth=1)
            ax.plot([i - 0.25, i + 0.25], [vals.mean()] * 2, color="#333333", linewidth=1.5)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_ylabel("Test EPL RMSE (dB)"); ax.set_title(title)
    axes[1].axhline(pers, color="#888888", linestyle="--", linewidth=1)
    axes[1].text(len(names) - 0.5, pers, "persistence", fontsize=7, va="bottom", ha="right", color="#555555")
    fig.suptitle(f"Run-to-run variation: one dot per seed (seeds {', '.join(map(str, sorted(runs.seed.unique())))}), "
                 "bar = mean", fontsize=10)
    save(fig, "seed_variation")


def minute_experiment(m):
    if "minute_multitask" not in m["variants"]:
        return
    v, b = m["variants"]["minute_multitask"], m["datasets"]["paper_minute"]["baselines"]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    steps = np.arange(1, H + 1)
    ax.plot(steps, v["RMSE_by_step"], "o-", color=ROLE["model"], lw=2, label=f"LSTM (overall {v['RMSE']:.4f})")
    ax.plot(steps, b["persistence"]["RMSE_by_step"], "s--", color=ROLE["persistence"],
            label=f"Persistence ({b['persistence']['RMSE']:.4f})")
    ax.plot(steps, b["linear_extrapolation"]["RMSE_by_step"], "^--", color=ROLE["linear_extrapolation"],
            label=f"Linear extrapolation ({b['linear_extrapolation']['RMSE']:.4f})")
    ax.plot(steps, [0.026, 0.026, 0.026, 0.027, 0.027], "k:", lw=1.5, label="Paper Table 7, overall (0.026)")
    ax.set_xticks(steps); ax.set_xlabel("Forecast step (minutes, 1-min interpolated data)")
    ax.set_ylabel("RMSE (dB)"); ax.set_title("Fig. 2 reading: 60-min input, 5-step horizon"); ax.legend(fontsize=7)
    save(fig, "minute_experiment")


def msl_comparison():
    a, b = RES / "table8_unseen_cities.csv", RES / "table8_unseen_cities_multitask_msl.csv"
    if not b.exists():
        return
    ta, tb = pd.read_csv(a), pd.read_csv(b)
    cities = list(UNSEEN_CITIES)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(cities))
    for i, (label, t, mode, color) in enumerate([
            ("Surface P, own inputs", ta, "own_inputs", SLOTS[1]), ("MSL P, own inputs", tb, "own_inputs", SLOTS[6]),
            ("Surface P, synthesized", ta, "synthesized", SLOTS[0]), ("MSL P, synthesized", tb, "synthesized", SLOTS[2])]):
        s = t[t["mode"] == mode].set_index("city")
        ax.bar(x + (i - 1.5) * 0.2, s.loc[cities, "MAE_dB"], 0.18, color=color, label=label)
    ax.set_xticks(x, cities); ax.set_ylabel("Hourly MAE (dB)")
    ax.set_title("Unseen cities: surface-pressure vs MSL-pressure input"); ax.legend(fontsize=7, ncol=2)
    save(fig, "msl_vs_surface_unseen")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    m = metrics()
    p = predictions("multitask")
    fig3(p); fig4(p); fig4_cities(p); fig4b(p)
    residuals(p); calibration(p, m)
    error_vs_horizon(m); by_city(m); weather_head(m); baselines_and_conditions(m)
    training_curves(); ablations(m); seed_variation(m); minute_experiment(m)
    unseen_figures(""); unseen_figures("_multitask_msl"); msl_comparison()


if __name__ == "__main__":
    main()
