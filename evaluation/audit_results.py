"""Independent audit of the saved results (read-only except for the report).

Recomputes every test metric from the saved predictions and the dataset
labels with its own code (not the evaluation script's functions), then
checks that metrics_test.json, the result tables, tables.md and the figures
agree with that recomputation and with each other.

Usage:
    python evaluation/audit_results.py
Output: results/paper/audit_report.json (one entry per check, PASS/FAIL)
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

RES = config.RESULTS_DIR
MODELS = ROOT / "models" / "paper"
VARIANTS = ["multitask", "no_itu", "epl_only", "weather_only", "multitask_msl",
            "minute_multitask", "multitask_fig1heads"]
FILES = ["best.keras", "model.keras", "run_config.json", "training_log.csv", "history.json",
         "val_predictions.npz", "test_predictions.npz"]
TOL = 1e-6
checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print("PASS" if ok else "FAIL", name, detail)


def close(a, b, tol=TOL):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def dataset_of(variant):
    return {"multitask_msl": config.DATASET_DIR_MSL,
            "minute_multitask": config.DATASET_DIR_MINUTE}.get(variant, config.DATASET_DIR)


def recompute(variant):
    """Metrics from first principles: errors in dB, sigma from the saved std."""
    d = dataset_of(variant)
    y = np.load(d / "epl_test.npy").astype(np.float64)
    p = np.load(MODELS / variant / "test_predictions.npz")
    out = {}
    if "epl_mean" in p:
        mu, sd = p["epl_mean"].astype(np.float64), p["epl_std"].astype(np.float64)
        err = mu - y
        out["RMSE"] = np.sqrt((err ** 2).mean())
        out["MAE"] = np.abs(err).mean()
        out["RMSE_by_step"] = np.sqrt((err ** 2).mean(axis=0))
        out["PICP_95_pct"] = 100 * (np.abs(err) <= 1.96 * sd).mean()
        out["NLL_eq10"] = (0.5 * (2 * np.log(sd) + (err / sd) ** 2)).mean()
        out["sharpness_dB"] = sd.mean()
        meta = pd.read_csv(d / "meta_test.csv")
        out["per_city_RMSE"] = {c: np.sqrt((err[(meta.city == c).to_numpy()] ** 2).mean())
                                for c in meta.city.unique()}
    if "weather_scaled" in p:
        w = p["weather_scaled"].astype(np.float64)
        out["weather_RMSE_std"] = np.sqrt(((w - np.load(d / "weather_test.npy")) ** 2).mean())
    return out


def main():
    metrics = json.load(open(RES / "metrics_test.json", encoding="utf-8"))
    recomputed = {}

    # 1. Every run is complete and internally consistent
    for v in VARIANTS:
        missing = [f for f in FILES if not (MODELS / v / f).exists()]
        check(f"{v}: all run files present", not missing, f"missing {missing}" if missing else "")
        cfg = json.load(open(MODELS / v / "run_config.json", encoding="utf-8"))
        log = pd.read_csv(MODELS / v / "training_log.csv")
        best = int(log.epoch[log.val_loss.idxmin()]) + 1
        check(f"{v}: run_config best_epoch/epochs match training_log",
              best == cfg["best_epoch"] and len(log) == cfg["epochs_run"] and not log.epoch.duplicated().any(),
              f"best {best}, epochs {len(log)}, seed {cfg['settings']['SEED']}")
        check(f"{v}: early stopping consistent (patience {config.PATIENCE} or max epochs)",
              len(log) - best == config.PATIENCE or len(log) == config.MAX_EPOCHS)
        p = np.load(MODELS / v / "test_predictions.npz")
        n = len(np.load(dataset_of(v) / "epl_test.npy", mmap_mode="r"))
        check(f"{v}: predictions finite, one row per test window",
              all(np.isfinite(p[k]).all() and len(p[k]) == n for k in p.keys()))

    # 2. Independent recomputation equals metrics_test.json
    for v in VARIANTS:
        r = recomputed[v] = recompute(v)
        m = metrics["variants"][v]
        keys = [k for k in ("RMSE", "MAE", "PICP_95_pct", "NLL_eq10", "sharpness_dB") if k in r]
        bad = [k for k in keys if not close(r[k], m[k], 1e-4 if k == "PICP_95_pct" else TOL)]
        if "RMSE_by_step" in r and not np.allclose(r["RMSE_by_step"], m["RMSE_by_step"], atol=TOL):
            bad.append("RMSE_by_step")
        if "weather_RMSE_std" in r and not close(r["weather_RMSE_std"], m["weather"]["RMSE_standardised"]):
            bad.append("weather")
        check(f"{v}: recomputed metrics equal metrics_test.json", not bad, f"differ: {bad}" if bad else "")
        if "per_city_RMSE" in r:
            t7 = pd.read_csv(RES / f"table7_rmse_{v}.csv").set_index("city")
            ok = all(close(t7.loc[c, "average_RMSE"], x) for c, x in r["per_city_RMSE"].items())
            check(f"{v}: per-city RMSE equals table7_rmse_{v}.csv", ok)

    # 3. Baselines recomputed from the labels
    for name, d in (("paper", config.DATASET_DIR), ("paper_minute", config.DATASET_DIR_MINUTE)):
        y = np.load(d / "epl_test.npy").astype(np.float64)
        now = np.load(d / "epl_now_test.npy").astype(np.float64)
        pers = np.sqrt(((now[:, None] - y) ** 2).mean())
        check(f"{name}: persistence RMSE recomputed", close(pers, metrics["datasets"][name]["baselines"]["persistence"]["RMSE"]),
              f"{pers:.4f}")
        if name == "paper_minute":
            prev = np.load(d / "epl_prev_test.npy").astype(np.float64)
            lin = now[:, None] + (now - prev)[:, None] * np.arange(1, config.HORIZON + 1)
            lin_rmse = np.sqrt(((lin - y) ** 2).mean())
            check("paper_minute: linear-extrapolation RMSE recomputed",
                  close(lin_rmse, metrics["datasets"][name]["baselines"]["linear_extrapolation"]["RMSE"]), f"{lin_rmse:.4f}")

    # 4. Units and scaling: inverse-transformed weather targets equal the raw values
    for d in (config.DATASET_DIR, config.DATASET_DIR_MSL, config.DATASET_DIR_MINUTE):
        sc = joblib.load(d / "weather_scaler.pkl")
        w, raw = np.load(d / "weather_test.npy"), np.load(d / "weather_raw_test.npy")
        inv = sc.inverse_transform(w.reshape(-1, w.shape[-1])).reshape(w.shape)
        check(f"{d.name}: weather inverse transform equals raw targets", np.allclose(inv, raw, atol=1e-3),
              f"max diff {np.abs(inv - raw).max():.2e}")
        tr = np.load(d / "weather_train.npy")
        check(f"{d.name}: weather scaler fitted on training targets (mean 0, std 1)",
              np.allclose(tr.reshape(-1, tr.shape[-1]).mean(0), 0, atol=1e-3)
              and np.allclose(tr.reshape(-1, tr.shape[-1]).std(0), 1, atol=1e-3))

    # 5. Result tables agree with the recomputation
    t9 = pd.read_csv(RES / "tables" / "table9_single_vs_multitask.csv").set_index("Configuration")
    check("Table 9 equals recomputation",
          close(t9.loc["EPL only", "EPL RMSE"], recomputed["epl_only"]["RMSE"])
          and close(t9.loc["Multi-task", "EPL RMSE"], recomputed["multitask"]["RMSE"])
          and close(t9.loc["Weather only", "Weather RMSE (standardised)"], recomputed["weather_only"]["weather_RMSE_std"]))
    t10 = pd.read_csv(RES / "tables" / "table10_itu_features.csv")
    check("Table 10 equals recomputation",
          close(t10["Average RMSE"][0], recomputed["multitask"]["RMSE"])
          and close(t10["Average RMSE"][1], recomputed["no_itu"]["RMSE"]))
    t7 = pd.read_csv(RES / "tables" / "table7_multitask.csv")
    check("Table 7 overall equals recomputation",
          any(close(x, recomputed["multitask"]["RMSE"], 5e-4) for x in t7.select_dtypes("number").to_numpy().ravel()))
    summary = pd.read_csv(RES / "tables" / "all_models_summary.csv")
    rows = summary[summary["Model / baseline"].isin(VARIANTS) & summary["RMSE"].notna()]
    check("all_models_summary equals recomputation",
          all(close(r.RMSE, recomputed[r["Model / baseline"]]["RMSE"]) for _, r in rows.iterrows()),
          f"{len(rows)} rows")
    md = (RES / "tables.md").read_text(encoding="utf-8")
    want = [f"{recomputed[v]['RMSE']:.3f}" for v in ("multitask", "epl_only", "no_itu")]
    check("tables.md shows the recomputed RMSEs", all(s in md for s in want), str(want))

    # 6. Unseen cities: summaries are the means of the per-city rows
    for suffix in ("", "_multitask_msl", "_multitask_fig1heads"):
        f = RES / f"table8_unseen_cities{suffix}.csv"
        if f.exists():
            per = pd.read_csv(f).groupby("mode")[["abs_error_of_means_dB", "MAE_dB", "RMSE_dB"]].mean()
            summ = pd.read_csv(RES / f"table8_unseen_summary{suffix}.csv", index_col=0)
            ok = np.allclose(per.to_numpy(), summ.loc[per.index, per.columns].to_numpy(), atol=1e-9)
            check(f"unseen{suffix or '_multitask'}: summary = mean of per-city rows", ok)

    # 7. Repeated-seed runs: complete, seed recorded, metrics equal seed_runs.csv
    if (RES / "seed_runs.csv").exists():
        seed_runs = pd.read_csv(RES / "seed_runs.csv").set_index("run")
        for run_dir in sorted((MODELS / "seeds").glob("*")):
            missing = [f for f in FILES if not (run_dir / f).exists()]
            cfg = json.load(open(run_dir / "run_config.json", encoding="utf-8")) if not missing else {}
            check(f"seeds/{run_dir.name}: complete, seed recorded in folder name and config",
                  not missing and run_dir.name.endswith(f"_seed{cfg.get('seed')}")
                  and cfg["settings"]["SEED"] == cfg["seed"], f"missing {missing}" if missing else "")
            if not missing:
                p = np.load(run_dir / "test_predictions.npz")
                y = np.load(config.DATASET_DIR / "epl_test.npy").astype(np.float64)
                if "epl_mean" in p:
                    r = np.sqrt(((p["epl_mean"].astype(np.float64) - y) ** 2).mean())
                    check(f"seeds/{run_dir.name}: RMSE equals seed_runs.csv",
                          close(r, seed_runs.loc[run_dir.name, "EPL_RMSE_dB"]), f"{r:.4f}")

    # 8. Figures: every PNG has a PDF of the same run, and result figures are newer than the metrics
    figs = RES / "figures"
    metrics_time = (RES / "metrics_test.json").stat().st_mtime
    for png in sorted(figs.glob("*.png")):
        pdf = png.with_suffix(".pdf")
        same_run = pdf.exists() and abs(pdf.stat().st_mtime - png.stat().st_mtime) < 120
        check(f"figure {png.stem}: PDF from the same run", same_run)
        if not png.stem.startswith(("dataset_", "fig1_", "fig2_")):
            check(f"figure {png.stem}: newer than metrics_test.json", png.stat().st_mtime >= metrics_time)

    report = {"passed": sum(c["status"] == "PASS" for c in checks),
              "failed": sum(c["status"] == "FAIL" for c in checks), "checks": checks}
    with open(RES / "audit_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"\n{report['passed']} passed, {report['failed']} failed -> {RES / 'audit_report.json'}")
    sys.exit(1 if report["failed"] else 0)


if __name__ == "__main__":
    main()
