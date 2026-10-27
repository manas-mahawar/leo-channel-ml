"""Numerical checks for the replication gap audit (no training, read-only inputs).

1. Arithmetic consistency of the paper's published tables and figures
   (values typed from the paper; they are reference values, not ours).
2. How our overall RMSE depends on the aggregation definition.
3. Our EPL label distribution compared with the range visible in the
   paper's Figs. 3-4 (max about 6.7 dB).
4. Trivial baselines under the sampling protocol suggested by the paper's
   Fig. 2 and Fig. 4: the full 2020-2024 hourly series of each city linearly
   interpolated to 1 minute, 60-minute windows every 5 minutes, chronological
   70/15/15 split. Both horizon readings are scored: 5 one-minute steps and
   5 one-hour steps.

Usage:
    python evaluation/replication_audit_checks.py
Output: results/replication_audit/*.json, *.csv (new folder; nothing else is written)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from data_pipeline.cities import CITIES

OUT = ROOT / "results" / "replication_audit"
H = config.HORIZON

# ---- Published values (El-Sayed et al. 2026), reference only ----------------
PAPER_T7_AVG = {"Athens": 0.017, "Bergen": 0.148, "Budapest": 0.005, "Chilbolton": 0.005,
                "Vigo": 0.003, "Warsaw": 0.039}
PAPER_T7_HOUR1 = {"Athens": 0.015, "Bergen": 0.144, "Budapest": 0.004, "Chilbolton": 0.003,
                  "Vigo": 0.002, "Warsaw": 0.038}
PAPER_T7_OVERALL = 0.026
PAPER_T8 = {"Berlin": (0.963, 1.049, 0.087), "Oslo": (1.228, 1.066, 0.162), "Rome": (0.865, 1.036, 0.172),
            "Stockholm": (0.887, 1.048, 0.162), "Bucharest": (0.896, 1.036, 0.140),
            "London": (0.988, 1.150, 0.162), "Davos": (0.787, 1.080, 0.293), "Paris": (0.869, 1.142, 0.272),
            "Marseille": (0.820, 1.143, 0.323), "Madrid": (0.810, 1.145, 0.335)}
PAPER_T8_AVG = (0.911, 1.089, 0.213)
PAPER_FIG5_ERR = {"Paris": 0.272, "Rome": 0.172, "Berlin": 0.097, "London": 0.162, "Madrid": 0.335,
                  "Oslo": 0.162, "Davos": 0.293, "Marseille": 0.323, "Bucharest": 0.140, "Stockholm": 0.171}
PAPER_FIG6 = {"Southern Europe": 0.277, "Western Europe": 0.177, "Northern Europe": 0.167,
              "Eastern Europe": 0.140, "Alpine": 0.293}
PAPER_FIG6_GROUPS = {"Southern Europe": ["Rome", "Madrid", "Marseille"],    # inferred, see below
                     "Western Europe": ["London", "Paris", "Berlin"],
                     "Northern Europe": ["Oslo", "Stockholm"],
                     "Eastern Europe": ["Bucharest"], "Alpine": ["Davos"]}
PAPER_T9 = {"EPL only": 0.015, "Multi-task EPL": 0.026, "Weather only": 0.892, "Multi-task weather": 0.761,
            "time EPL only h": 1.45, "time multi-task h": 1.91, "stated improvements": "40%, 14.7%, +32%"}


def paper_checks():
    t7 = np.array(list(PAPER_T7_AVG.values()))
    t8 = np.array(list(PAPER_T8.values()))
    bergen_max_weight = PAPER_T7_OVERALL ** 2 / PAPER_T7_AVG["Bergen"] ** 2
    fig6 = {r: round(float(np.mean([PAPER_FIG5_ERR[c] for c in cs])), 4) for r, cs in PAPER_FIG6_GROUPS.items()}
    return {
        "table7": {
            "reported_overall": PAPER_T7_OVERALL,
            "mean_of_city_averages": round(float(t7.mean()), 4),
            "rms_of_city_averages_equal_weights": round(float(np.sqrt((t7 ** 2).mean())), 4),
            "median_of_city_averages": round(float(np.median(t7)), 4),
            "hour1_reported_overall": 0.026,
            "hour1_mean_of_cities": round(float(np.mean(list(PAPER_T7_HOUR1.values()))), 4),
            "hour1_rms_of_cities": round(float(np.sqrt(np.mean(np.square(list(PAPER_T7_HOUR1.values()))))), 4),
            "max_share_of_squared_error_weight_on_Bergen_for_pooled_rms_0.026": round(bergen_max_weight, 4),
            "note": "With equal city sample sizes (10,000 per city), neither the mean (0.036) nor the "
                    "pooled RMS (0.063) of the city values gives 0.026; a pooled RMS of 0.026 would need "
                    "Bergen to hold at most about 3 % of the test samples.",
        },
        "table8": {
            "reported_averages": PAPER_T8_AVG,
            "mean_true": round(float(t8[:, 0].mean()), 4), "mean_pred": round(float(t8[:, 1].mean()), 4),
            "mean_of_listed_abs_errors": round(float(t8[:, 2].mean()), 4),
            "abs_difference_of_means": round(float(abs(t8[:, 1].mean() - t8[:, 0].mean())), 4),
            "rows_where_listed_error_differs_from_abs(pred-true)_by_more_than_0.0015": {
                c: round(abs(p - t), 3) for c, (t, p, e) in PAPER_T8.items() if abs(abs(p - t) - e) > 0.0015},
            "figure5_and_7_values_differing_from_table8": {
                c: {"table8": PAPER_T8[c][2], "fig5": v} for c, v in PAPER_FIG5_ERR.items() if v != PAPER_T8[c][2]},
            "figure5_average_of_bars": round(float(np.mean(list(PAPER_FIG5_ERR.values()))), 4),
            "figure7_reported_average": 0.212, "figure7_reported_std": 0.082,
            "std_of_fig5_errors_population": round(float(np.std(list(PAPER_FIG5_ERR.values()))), 4),
            "std_of_fig5_errors_sample": round(float(np.std(list(PAPER_FIG5_ERR.values()), ddof=1)), 4),
        },
        "figure6": {"reported": PAPER_FIG6, "recomputed_from_fig5_with_inferred_groups": fig6,
                    "inferred_groups": PAPER_FIG6_GROUPS,
                    "note": "Grouping inferred by matching averages; Berlin must be in Western Europe."},
        "table9": {
            "reported": PAPER_T9,
            "multitask_vs_epl_only_relative_change_pct": round(100 * (0.026 - 0.015) / 0.015, 1),
            "epl_only_vs_multitask_relative_reduction_pct": round(100 * (0.026 - 0.015) / 0.026, 1),
            "weather_improvement_pct": round(100 * (0.892 - 0.761) / 0.892, 1),
            "training_time_increase_pct": round(100 * (1.91 - 1.45) / 1.45, 1),
            "note": "The table's own EPL numbers show the EPL-only model 42 % better than multi-task; the "
                    "'40 % improvement' cannot be a multi-task EPL gain as the text states.",
        },
        "table10": {"rmse_change_pct": round(100 * (0.028 - 0.026) / 0.026, 1),
                    "mae_change_pct": round(100 * (0.012 - 0.009) / 0.009, 1)},
        "uncertainty": {"sect4.1_calibration_pct": 99.03, "sect4.1_PICP_pct": 99.23, "table11_pct": 99.03,
                        "note": "Two different coverage values are reported for the trained cities."},
        "early_stopping": {"text_patience": 6, "table6": "Early stopping 23 epochs",
                           "note": "Table 6 gives an epoch count where the text gives a patience."},
    }


def our_aggregation():
    out = {}
    y = np.load(config.DATASET_DIR / "epl_test.npy").astype(float)
    meta = pd.read_csv(config.DATASET_DIR / "meta_test.csv")
    for v in ("multitask", "multitask_fig1heads"):
        mu = np.load(ROOT / "models" / "paper" / v / "test_predictions.npz")["epl_mean"].astype(float)
        e = mu - y
        city = {c: float(np.sqrt((e[(meta.city == c).to_numpy()] ** 2).mean())) for c in CITIES}
        out[v] = {"pooled_rmse_all_windows_and_steps": float(np.sqrt((e ** 2).mean())),
                  "mean_of_city_rmse": float(np.mean(list(city.values()))),
                  "rms_of_city_rmse": float(np.sqrt(np.mean(np.square(list(city.values()))))),
                  "mean_of_per_step_rmse": float(np.mean(np.sqrt((e ** 2).mean(0)))),
                  "city_rmse": city, "test_windows_per_city": meta.city.value_counts().to_dict()}
    return out


def label_distribution():
    y = np.load(config.DATASET_DIR / "epl_test.npy").astype(float)
    meta = pd.read_csv(config.DATASET_DIR / "meta_test.csv")
    mu = np.load(ROOT / "models" / "paper" / "multitask" / "test_predictions.npz")["epl_mean"].astype(float)
    small = y.max(1) <= 6.7
    rows = {}
    for c in CITIES:
        m = (meta.city == c).to_numpy()
        rows[c] = {"max_dB": float(y[m].max()), "p99_dB": float(np.percentile(y[m], 99)),
                   "share_targets_above_6.7dB_pct": float(100 * (y[m] > 6.7).mean())}
    se = (mu - y) ** 2
    return {"paper_visible_max_true_EPL_dB": "about 6.2 (Fig. 3) to 6.7 (Fig. 4)",
            "our_test_targets": rows,
            "our_overall_max_dB": float(y.max()),
            "share_windows_with_any_target_above_6.7dB_pct": float(100 * (~small).mean()),
            "share_of_squared_error_from_those_windows_pct": float(100 * se[~small].sum() / se.sum()),
            "diagnostic_rmse_main_model_on_windows_max_le_6.7dB": float(np.sqrt(se[small].mean())),
            "note": "Diagnostic only: removing windows by their true value uses the outcome and is not a "
                    "valid evaluation; it shows how much of our RMSE comes from the label tail."}


def minute_protocol_baselines():
    rows = []
    n_test_total = 0
    for city in CITIES:
        phys = pd.read_csv(config.PHYSICS_DIR / f"{city}_physics.csv", usecols=["valid_time", config.EPL_COLUMN])
        epl = phys[config.EPL_COLUMN].to_numpy(float)
        minutes = np.arange(len(epl)) * 60.0
        n_min = int(minutes[-1])
        last_needed = 300                                # hour-step reading needs +5 h
        anchors = np.arange(59, n_min - last_needed + 1, 5)
        test = anchors[int(round(0.85 * len(anchors))):]
        n_test_total += len(test)
        f = lambda m: np.interp(m, minutes, epl)
        now, prev = f(test), f(test - 1)
        for reading, steps in (("minute steps (+1..+5 min)", np.arange(1, H + 1)),
                               ("hour steps (+1..+5 h)", 60 * np.arange(1, H + 1))):
            y = f(test[:, None] + steps)
            pers = np.repeat(now[:, None], H, axis=1)
            lin = now[:, None] + (now - prev)[:, None] * steps
            for name, p in (("persistence", pers), ("linear_extrapolation", lin)):
                e = p - y
                rows.append({"city": city, "reading": reading, "baseline": name, "test_windows": len(test),
                             "rmse_avg": float(np.sqrt((e ** 2).mean())),
                             **{f"rmse_step{k + 1}": float(np.sqrt((e[:, k] ** 2).mean())) for k in range(H)}})
    table = pd.DataFrame(rows)
    table["paper_table7_avg"] = table.city.map(PAPER_T7_AVG)
    overall = (table.assign(se=table.rmse_avg ** 2 * table.test_windows)
               .groupby(["reading", "baseline"])
               .apply(lambda g: np.sqrt(g.se.sum() / g.test_windows.sum()), include_groups=False))
    return table, {"total_test_windows_six_cities": int(n_test_total),
                   "paper_fig4_samples_axis": "about 470,000",
                   "pooled_overall_rmse": {f"{r} / {b}": float(v) for (r, b), v in overall.items()}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"paper_consistency": paper_checks(), "our_aggregation": our_aggregation(),
              "label_distribution": label_distribution()}
    table, summary = minute_protocol_baselines()
    report["full_series_minute_protocol"] = summary
    table.to_csv(OUT / "minute_protocol_baselines.csv", index=False)
    with open(OUT / "audit_checks.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2)[:6000])
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
