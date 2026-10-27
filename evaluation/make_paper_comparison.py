"""Side-by-side comparison of published values and our measured results.

Writes to results/replication_audit/ (existing figures and tables are not touched):
  paper_vs_ours.md                     Tables 7-10, published reference vs measured
  fig6_regions_paper_grouping.png/pdf  Fig. 6 with the paper's regional grouping
                                       (inferred in replication_audit_checks.py:
                                       Berlin is in Western Europe)
Every published number is labelled "published (reference)"; nothing from the
paper is plotted as our result.

Usage (after evaluation/replication_audit_checks.py):
    python evaluation/make_paper_comparison.py
"""

import json
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
from replication_audit_checks import (PAPER_FIG5_ERR, PAPER_FIG6, PAPER_FIG6_GROUPS, PAPER_T7_AVG,
                                      PAPER_T8)

RES = config.RESULTS_DIR
OUT = ROOT / "results" / "replication_audit"


def fmt(v, d=3):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def table7(lines):
    seeds = pd.read_csv(RES / "seed_summary.csv").set_index("variant")
    hourly = {v: pd.read_csv(RES / f"table7_rmse_{v}.csv").set_index("city")["average_RMSE"]
              for v in ("multitask", "multitask_fig1heads", "paper_persistence")}
    minute = pd.read_csv(OUT / "minute_protocol_baselines.csv")
    minute = minute[minute.reading.str.startswith("minute")].pivot(index="city", columns="baseline", values="rmse_avg")
    lines += ["## Table 7: trained-city RMSE (dB), average over the five steps\n",
              "Hourly columns: our chronological test set (8,916 windows x 5 h, seed 42). Minute columns: "
              "trivial baselines (no model) on the full 2020-2024 series interpolated to 1 min, 5-min stride, "
              "last 15 % per city (473,484 windows), forecasting +1..+5 min.\n",
              "| City | Published (reference) | Fig. 1 heads, hourly | Main, hourly | Persistence, hourly | "
              "Persistence, minute protocol | Linear extrap., minute protocol |", "|---|---|---|---|---|---|---|"]
    for c in list(CITIES) + ["Overall"]:
        pub = PAPER_T7_AVG.get(c, 0.026 if c == "Overall" else None)
        mrow = minute.loc[c] if c in minute.index else None
        lines.append(f"| {c} | {fmt(pub)} | {fmt(hourly['multitask_fig1heads'][c])} | {fmt(hourly['multitask'][c])} | "
                     f"{fmt(hourly['paper_persistence'][c])} | {fmt(None if mrow is None else mrow['persistence'])} | "
                     f"{fmt(None if mrow is None else mrow['linear_extrapolation'])} |")
    checks = json.load(open(OUT / "audit_checks.json", encoding="utf-8"))["full_series_minute_protocol"]
    lines += [f"\nMinute-protocol pooled overall: persistence "
              f"{checks['pooled_overall_rmse']['minute steps (+1..+5 min) / persistence']:.3f}, linear extrapolation "
              f"{checks['pooled_overall_rmse']['minute steps (+1..+5 min) / linear_extrapolation']:.3f} dB. "
              f"Three-seed hourly means: Fig. 1 heads {seeds.loc['multitask_fig1heads', 'EPL_RMSE_dB_mean']:.3f} +- "
              f"{seeds.loc['multitask_fig1heads', 'EPL_RMSE_dB_std']:.3f}, main "
              f"{seeds.loc['multitask', 'EPL_RMSE_dB_mean']:.3f} +- {seeds.loc['multitask', 'EPL_RMSE_dB_std']:.3f} dB.\n"]


def table8(lines, ours):
    lines += ["## Table 8: unseen cities, mean EPL over the evaluation period (dB)\n",
              "Ours: main model (seed 42), synthesized inputs (Eq. 12-13, tau 500 km), every hour "
              "2024-04-09..2024-12-31. 'Abs. error' = |mean prediction - mean truth| (the paper's measure).\n",
              "| City | Published true | Published pred. | Published abs. error | Our true | Our pred. | Our abs. error |",
              "|---|---|---|---|---|---|---|"]
    for c, (t, p, e) in PAPER_T8.items():
        r = ours.loc[c]
        lines.append(f"| {c} | {t:.3f} | {p:.3f} | {e:.3f} | {r.true_EPL_mean_dB:.3f} | {r.pred_EPL_mean_dB:.3f} | "
                     f"{r.abs_error_of_means_dB:.3f} |")
    lines.append(f"| Average | 0.911 | 1.089 | 0.213 | {ours.true_EPL_mean_dB.mean():.3f} | "
                 f"{ours.pred_EPL_mean_dB.mean():.3f} | {ours.abs_error_of_means_dB.mean():.3f} |\n")


def tables9_10(lines):
    s = pd.read_csv(RES / "seed_summary.csv").set_index("variant")
    m = json.load(open(RES / "metrics_test.json", encoding="utf-8"))["variants"]
    lines += ["## Table 9: single-task vs multi-task\n",
              "| Configuration | Published EPL RMSE | Published weather RMSE | Our EPL RMSE (3 seeds) | "
              "Our weather RMSE, standardised |", "|---|---|---|---|---|",
              f"| EPL only | 0.015 | - | {s.loc['epl_only', 'EPL_RMSE_dB_mean']:.3f} +- {s.loc['epl_only', 'EPL_RMSE_dB_std']:.3f} | - |",
              f"| Weather only | - | 0.892 | - | {m['weather_only']['weather']['RMSE_standardised']:.3f} (1 seed) |",
              f"| Multi-task | 0.026 | 0.761 | {s.loc['multitask', 'EPL_RMSE_dB_mean']:.3f} +- {s.loc['multitask', 'EPL_RMSE_dB_std']:.3f} | "
              f"{s.loc['multitask', 'weather_RMSE_std_mean']:.3f} +- {s.loc['multitask', 'weather_RMSE_std_std']:.3f} |",
              "\nThe paper does not define how its weather RMSE is scaled, so only the direction can be compared.\n",
              "## Table 10: with vs without ITU-R-derived features\n",
              "| Configuration | Published RMSE | Published MAE | Our RMSE (3 seeds) | Our MAE (3 seeds) |",
              "|---|---|---|---|---|",
              f"| Full feature set | 0.026 | 0.009 | {s.loc['multitask', 'EPL_RMSE_dB_mean']:.3f} +- {s.loc['multitask', 'EPL_RMSE_dB_std']:.3f} | "
              f"{s.loc['multitask', 'EPL_MAE_dB_mean']:.3f} |",
              f"| Without ITU-R features | 0.028 | 0.012 | {s.loc['no_itu', 'EPL_RMSE_dB_mean']:.3f} +- {s.loc['no_itu', 'EPL_RMSE_dB_std']:.3f} | "
              f"{s.loc['no_itu', 'EPL_MAE_dB_mean']:.3f} |\n"]


def fig6(ours):
    regions = list(PAPER_FIG6_GROUPS)
    our_vals = [ours.loc[PAPER_FIG6_GROUPS[r], "abs_error_of_means_dB"].mean() for r in regions]
    pub_vals = [PAPER_FIG6[r] for r in regions]
    x = np.arange(len(regions))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(x - 0.2, pub_vals, 0.38, color="#bbbbbb", edgecolor="#555555", hatch="//",
           label="Published (reference, Fig. 6)")
    ax.bar(x + 0.2, our_vals, 0.38, color="#2a78d6", label="Ours: main model, synthesized inputs")
    for i, (p, o) in enumerate(zip(pub_vals, our_vals)):
        ax.text(i - 0.2, p, f"{p:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + 0.2, o, f"{o:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [f"{r}\n({', '.join(PAPER_FIG6_GROUPS[r])})" for r in regions], fontsize=7)
    ax.set_ylabel("Mean |error of period means| (dB)")
    ax.set_title("Fig. 6 layout with the paper's regional grouping (inferred: Berlin in Western Europe)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(axis="y", color="#e5e5e5")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_regions_paper_grouping.png", dpi=300)
    fig.savefig(OUT / "fig6_regions_paper_grouping.pdf")
    plt.close(fig)


def main():
    ours = pd.read_csv(RES / "table8_unseen_cities.csv")
    ours = ours[ours["mode"] == "synthesized"].set_index("city")
    lines = ["# Published reference values vs our measured results\n",
             "Generated by `evaluation/make_paper_comparison.py`. 'Published' columns are copied from "
             "El-Sayed et al. (2026) as references; all other numbers are measured in this repository.\n"]
    table7(lines); table8(lines, ours); tables9_10(lines)
    (OUT / "paper_vs_ours.md").write_text("\n".join(lines), encoding="utf-8")
    fig6(ours)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
