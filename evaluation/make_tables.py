"""Paper-style Tables 7-11 from our saved results, plus paper reference values.

Our numbers are read from results/paper/metrics_test.json and the table CSVs
written by the evaluation scripts. Paper values are copied from El-Sayed et
al. (2026), Tables 7-11, and are shown only as references.

Usage:
    python evaluation/make_tables.py
Outputs: results/paper/tables/*.csv and results/paper/tables.md
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from data_pipeline.cities import UNSEEN_CITIES

RES = config.RESULTS_DIR
OUT = RES / "tables"

PAPER_T7 = {"Athens": [0.015, 0.016, 0.017, 0.018, 0.020, 0.017], "Bergen": [0.144, 0.146, 0.150, 0.150, 0.152, 0.148],
            "Budapest": [0.004, 0.004, 0.004, 0.005, 0.006, 0.005], "Chilbolton": [0.003, 0.004, 0.005, 0.005, 0.006, 0.005],
            "Vigo": [0.002, 0.003, 0.003, 0.003, 0.004, 0.003], "Warsaw": [0.038, 0.038, 0.039, 0.039, 0.040, 0.039],
            "Overall": [0.026, 0.026, 0.026, 0.027, 0.027, 0.026]}
PAPER_T8 = {"Berlin": (0.963, 1.049, 0.087), "Oslo": (1.228, 1.066, 0.162), "Rome": (0.865, 1.036, 0.172),
            "Stockholm": (0.887, 1.048, 0.162), "Bucharest": (0.896, 1.036, 0.140), "London": (0.988, 1.150, 0.162),
            "Davos": (0.787, 1.080, 0.293), "Paris": (0.869, 1.142, 0.272), "Marseille": (0.820, 1.143, 0.323),
            "Madrid": (0.810, 1.145, 0.335)}


def md(df, digits=3):
    def cell(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.{digits}f}"
        return "" if v is None or (isinstance(v, str) and v == "nan") else str(v)

    df = df.map(cell)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def table7(name):
    t = pd.read_csv(RES / f"table7_rmse_{name}.csv")
    t.columns = ["City", "Hour 1", "Hour 2", "Hour 3", "Hour 4", "Hour 5", "Average", "MAE"]
    t["Paper average"] = t["City"].map(lambda c: PAPER_T7[c][-1])
    return t


def table8(suffix=""):
    t = pd.read_csv(RES / f"table8_unseen_cities{suffix}.csv")
    rows = []
    for city in UNSEEN_CITIES:
        g = t[t.city == city].set_index("mode")
        rows.append({
            "City": city,
            "True EPL": g.loc["synthesized", "true_EPL_mean_dB"],
            "Pred. (weighted)": g.loc["synthesized", "pred_EPL_mean_dB"],
            "Abs. error (weighted)": g.loc["synthesized", "abs_error_of_means_dB"],
            "Abs. error (median)": g.loc["synthesized_median", "abs_error_of_means_dB"],
            "Hourly MAE (weighted)": g.loc["synthesized", "MAE_dB"],
            "Hourly MAE (own inputs)": g.loc["own_inputs", "MAE_dB"],
            "Hourly MAE (persistence)": g.loc["persistence_own", "MAE_dB"],
            "Paper true": PAPER_T8[city][0], "Paper pred.": PAPER_T8[city][1], "Paper abs. error": PAPER_T8[city][2],
        })
    df = pd.DataFrame(rows)
    mean = df.drop(columns="City").mean()
    return pd.concat([df, pd.DataFrame([{"City": "Average", **mean.to_dict()}])], ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = json.load(open(RES / "metrics_test.json", encoding="utf-8"))
    v, base = m["variants"], m["datasets"]["paper"]["baselines"]
    sections = ["# Result tables\n",
                "All values in dB unless stated. 'Paper' columns are copied from El-Sayed et al. (2026) "
                "for reference; every other number comes from our runs.\n"]

    t7 = table7("multitask")
    t7.to_csv(OUT / "table7_multitask.csv", index=False)
    sections += ["## Table 7: test RMSE per city and forecast hour (multi-task LSTM)\n", md(t7), ""]
    for b in ("persistence", "climatology"):
        tb = table7(f"paper_{b}")
        tb.to_csv(OUT / f"table7_{b}.csv", index=False)
        sections += [f"### Table 7 layout, {b} baseline\n", md(tb.drop(columns="Paper average")), ""]
    if "minute_multitask" in v:
        tm = table7("minute_multitask").rename(columns={f"Hour {i}": f"Min {i}" for i in range(1, 6)})
        tm.to_csv(OUT / "table7_minute_multitask.csv", index=False)
        sections += ["### Table 7 layout, Fig. 2 reading (1-min interpolated data, steps = minutes)\n", md(tm, 4), ""]
        for b in ("persistence", "linear_extrapolation"):
            tb = table7(f"paper_minute_{b}").rename(columns={f"Hour {i}": f"Min {i}" for i in range(1, 6)})
            sections += [f"#### {b}, same minute test set\n", md(tb.drop(columns="Paper average"), 4), ""]

    for suffix, label in (("", "multitask"), ("_multitask_msl", "multitask_msl")):
        if (RES / f"table8_unseen_cities{suffix}.csv").exists():
            t8 = table8(suffix)
            t8.to_csv(OUT / f"table8_{label}.csv", index=False)
            sections += [f"## Table 8: ten unseen cities ({label})\n",
                         "Means over every hour Apr-Dec 2024 (6,385 forecasts per city). "
                         "'Abs. error' = |mean predicted - mean true| (the paper's measure); "
                         "'Hourly MAE' = mean |error| over hours and horizons.\n", md(t8), ""]

    rows9 = []
    for name, label in (("epl_only", "EPL only"), ("weather_only", "Weather only"), ("multitask", "Multi-task")):
        if name in v:
            r = v[name]
            rows9.append({"Configuration": label, "EPL RMSE": r.get("RMSE"), "EPL MAE": r.get("MAE"),
                          "Weather RMSE (standardised)": r.get("weather", {}).get("RMSE_standardised"),
                          "Training time (h)": r["train_hours"], "Epochs (best)": f"{r['epochs_run']} ({r['best_epoch']})"})
    t9 = pd.DataFrame(rows9)
    t9["Paper EPL RMSE"] = t9["Configuration"].map({"EPL only": 0.015, "Multi-task": 0.026})
    t9["Paper weather RMSE"] = t9["Configuration"].map({"Weather only": 0.892, "Multi-task": 0.761})
    t9.to_csv(OUT / "table9_single_vs_multitask.csv", index=False)
    sections += ["## Table 9: single-task vs multi-task\n", md(t9),
                 "\nTraining times include any standby pauses of the laptop and are not comparable across runs.\n"]

    rows10 = []
    for name, label in (("multitask", "Full feature set (14)"), ("no_itu", "Without ITU-R-derived features (8)")):
        if name in v:
            rows10.append({"Model configuration": label, "Average RMSE": v[name]["RMSE"], "Average MAE": v[name]["MAE"],
                           "PICP 95 %": v[name]["PICP_95_pct"]})
    t10 = pd.DataFrame(rows10)
    t10["Paper RMSE"] = [0.026, 0.028][:len(t10)]
    t10["Paper MAE"] = [0.009, 0.012][:len(t10)]
    t10.to_csv(OUT / "table10_itu_features.csv", index=False)
    sections += ["## Table 10: removing the ITU-R-derived features\n", md(t10), ""]

    mt = v["multitask"]
    unseen = pd.read_csv(RES / "table8_unseen_summary.csv", index_col=0)
    t11 = pd.DataFrame([
        {"Metric": "Trained RMSE (dB)", "Bai et al. MLP [11]": "0.75", "Bai et al. LSTM [12]": "~0.65",
         "Paper (reported)": "0.026", "This replication": f"{mt['RMSE']:.3f}"},
        {"Metric": "Unseen-city error (dB)", "Bai et al. MLP [11]": "Not tested", "Bai et al. LSTM [12]": "Not tested",
         "Paper (reported)": "0.213",
         "This replication": f"{unseen.loc['synthesized', 'abs_error_of_means_dB']:.3f} (error of means); "
                             f"{unseen.loc['synthesized', 'MAE_dB']:.3f} (hourly MAE)"},
        {"Metric": "Prediction horizon", "Bai et al. MLP [11]": "1 min", "Bai et al. LSTM [12]": "1 min",
         "Paper (reported)": "5 h", "This replication": "5 h (hourly steps)"},
        {"Metric": "Geographic coverage", "Bai et al. MLP [11]": "Single-site", "Bai et al. LSTM [12]": "Single-site",
         "Paper (reported)": "10 cities", "This replication": "6 training + 10 unseen cities"},
        {"Metric": "Uncertainty quantification", "Bai et al. MLP [11]": "No", "Bai et al. LSTM [12]": "No",
         "Paper (reported)": "99.03 %", "This replication": f"PICP {mt['PICP_95_pct']:.1f} %"},
        {"Metric": "Training cities", "Bai et al. MLP [11]": "1", "Bai et al. LSTM [12]": "1",
         "Paper (reported)": "6", "This replication": "6"},
        {"Metric": "Test cities", "Bai et al. MLP [11]": "1", "Bai et al. LSTM [12]": "1",
         "Paper (reported)": "10", "This replication": "6 (held-out period) + 10 unseen"},
    ])
    t11.to_csv(OUT / "table11_prior_work.csv", index=False)
    sections += ["## Table 11: qualitative comparison with prior work\n", md(t11),
                 "\nBai et al. values are as quoted in the paper (different datasets and targets; not re-run here).\n"]

    rows = [{"Model / baseline": "Persistence", "Dataset": "paper", "RMSE": base["persistence"]["RMSE"],
             "MAE": base["persistence"]["MAE"]},
            {"Model / baseline": "Climatology", "Dataset": "paper", "RMSE": base["climatology"]["RMSE"],
             "MAE": base["climatology"]["MAE"]}]
    for name, r in v.items():
        rows.append({"Model / baseline": name, "Dataset": r["dataset"], "RMSE": r.get("RMSE"), "MAE": r.get("MAE"),
                     "PICP 95 %": r.get("PICP_95_pct"), "NLL (Eq. 10)": r.get("NLL_eq10"),
                     "Sharpness": r.get("sharpness_dB"),
                     "Weather RMSE (std.)": r.get("weather", {}).get("RMSE_standardised"),
                     "Best epoch": str(r["best_epoch"])})
    if "paper_minute" in m["datasets"]:
        for b, r in m["datasets"]["paper_minute"]["baselines"].items():
            if "MAE" in r:
                rows.append({"Model / baseline": b, "Dataset": "paper_minute", "RMSE": r["RMSE"], "MAE": r["MAE"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "all_models_summary.csv", index=False)
    sections += ["## All trained variants and baselines (test sets)\n", md(summary, 4), ""]

    if (RES / "seed_summary.csv").exists():   # written by evaluation/summarize_runs.py
        seeds = pd.read_csv(RES / "seed_summary.csv")
        seeds = seeds[seeds.n_runs > 1][["variant", "n_runs", "seeds", "EPL_RMSE_dB_mean", "EPL_RMSE_dB_std",
                                         "EPL_RMSE_dB_min", "EPL_RMSE_dB_max", "EPL_MAE_dB_mean",
                                         "EPL_RMSE_hour1_dB_mean", "PICP_95_pct_mean"]]
        seeds.to_csv(OUT / "repeated_seeds.csv", index=False)
        pairs = pd.read_csv(RES / "seed_comparisons.csv")
        sections += ["## Repeated seeds (test set, EPL in dB)\n",
                     "Mean and sample standard deviation over the listed seeds; seed 42 is the original run.\n",
                     md(seeds, 4), "", "### Paired differences (A - B, same seed; negative = A better)\n",
                     md(pairs, 4), ""]

    (RES / "tables.md").write_text("\n".join(sections), encoding="utf-8")
    print("\n".join(sections))


if __name__ == "__main__":
    main()
