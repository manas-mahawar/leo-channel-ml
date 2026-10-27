"""Machine-readable summary of every training run, including repeated seeds.

For each run (models/paper/<variant>/ = seed 42, models/paper/seeds/<variant>_seed<N>/)
the test metrics are computed from its saved predictions. A run counts as
complete only when its run_config.json exists (written at the end of training).
Runs listed in training/queue_seeds.txt but not finished are reported as
"running" (folder exists) or "queued".

Usage:
    python evaluation/summarize_runs.py
Outputs (results/paper/): experiments_summary.json, seed_runs.csv, seed_summary.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

MODELS = ROOT / "models" / "paper"
OUT = config.RESULTS_DIR
DATASETS = {"multitask_msl": config.DATASET_DIR_MSL, "minute_multitask": config.DATASET_DIR_MINUTE}
# (variant A, variant B, metric): A - B per seed; negative = A better
COMPARISONS = [("multitask_fig1heads", "multitask", "EPL_RMSE_dB"),
               ("multitask", "epl_only", "EPL_RMSE_dB"),
               ("no_itu", "multitask", "EPL_RMSE_dB"),
               ("multitask_fig1heads", "multitask", "EPL_RMSE_hour1_dB")]


def test_metrics(run_dir, variant):
    d = DATASETS.get(variant, config.DATASET_DIR)
    p = np.load(run_dir / "test_predictions.npz")
    out = {}
    if "epl_mean" in p:
        y = np.load(d / "epl_test.npy").astype(float)
        mu, sd = p["epl_mean"].astype(float), p["epl_std"].astype(float)
        err = mu - y
        out.update({"EPL_RMSE_dB": float(np.sqrt((err ** 2).mean())), "EPL_MAE_dB": float(np.abs(err).mean()),
                    "EPL_RMSE_hour1_dB": float(np.sqrt((err[:, 0] ** 2).mean())),
                    "PICP_95_pct": float(100 * (np.abs(err) <= 1.96 * sd).mean()),
                    "NLL_eq10": float((0.5 * (2 * np.log(sd) + (err / sd) ** 2)).mean()),
                    "sharpness_dB": float(sd.mean())})
    if "weather_scaled" in p:
        w = np.load(d / "weather_test.npy")
        out["weather_RMSE_std"] = float(np.sqrt(((p["weather_scaled"] - w) ** 2).mean()))
    return out


def planned_seed_runs():
    queue = ROOT / "training" / "queue_seeds.txt"
    runs = []
    if queue.exists():
        for line in queue.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and not parts[0].startswith("#"):
                runs.append((parts[0], int(parts[1])))
    return runs


def main():
    run_dirs = [p for p in sorted(MODELS.iterdir()) if (p / "run_config.json").exists()]
    run_dirs += [p for p in sorted((MODELS / "seeds").glob("*")) if p.is_dir()]
    rows = []
    for run_dir in run_dirs:
        if not (run_dir / "run_config.json").exists():
            continue
        cfg = json.load(open(run_dir / "run_config.json", encoding="utf-8"))
        row = {"run": run_dir.name, "variant": cfg["variant"], "seed": cfg["settings"]["SEED"],
               "status": "complete", "dataset": cfg.get("dataset", str(config.DATASET_DIR.relative_to(ROOT))),
               "path": str(run_dir.relative_to(ROOT)), "params": cfg["params"],
               "epochs_run": cfg["epochs_run"], "best_epoch": cfg["best_epoch"],
               "train_hours_wallclock": cfg["train_hours"]}
        row.update(test_metrics(run_dir, cfg["variant"]))
        rows.append(row)
    done = {(r["variant"], r["seed"]) for r in rows}
    for variant, seed in planned_seed_runs():
        if (variant, seed) not in done:
            folder = MODELS / "seeds" / f"{variant}_seed{seed}"
            rows.append({"run": folder.name, "variant": variant, "seed": seed,
                         "status": "running" if folder.exists() else "queued", "path": str(folder.relative_to(ROOT))})

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "seed_runs.csv", index=False)

    complete = table[table.status == "complete"]
    metrics = [c for c in ("EPL_RMSE_dB", "EPL_MAE_dB", "EPL_RMSE_hour1_dB", "PICP_95_pct", "weather_RMSE_std")
               if c in complete]
    agg = complete.groupby("variant")[metrics].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.insert(0, "seeds", complete.groupby("variant").seed.apply(lambda s: " ".join(map(str, sorted(s)))))
    agg.insert(0, "n_runs", complete.groupby("variant").size())
    agg.to_csv(OUT / "seed_summary.csv")

    # Paired comparisons: same seed, same data, only the variant differs
    pairs = []
    for a, b, metric in COMPARISONS:
        ra = complete[complete.variant == a].set_index("seed")[metric]
        rb = complete[complete.variant == b].set_index("seed")[metric]
        seeds = sorted(set(ra.index) & set(rb.index))
        if seeds:
            diff = (ra[seeds] - rb[seeds]).to_numpy()
            pairs.append({"comparison": f"{a} - {b}", "metric": metric, "seeds": " ".join(map(str, seeds)),
                          "differences": " ".join(f"{d:+.4f}" for d in diff),
                          "mean_difference": float(diff.mean()),
                          "std_difference": float(diff.std(ddof=1)) if len(diff) > 1 else None,
                          "same_sign_all_seeds": bool(np.all(diff > 0) or np.all(diff < 0))})
    pd.DataFrame(pairs).to_csv(OUT / "seed_comparisons.csv", index=False)

    summary = {"note": "Metrics computed from saved test predictions; 'std' is the sample standard "
                       "deviation over completed seeds (NaN for a single run). Paper values are "
                       "references only and are not included here.",
               "runs": json.loads(table.to_json(orient="records")),
               "by_variant": json.loads(agg.reset_index().to_json(orient="records")),
               "paired_comparisons": pairs}
    with open(OUT / "experiments_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(agg[["n_runs", "seeds"] + [c for c in agg.columns if c.startswith("EPL_RMSE_dB")]].round(4).to_string())
    print(f"\n{(table.status == 'complete').sum()} complete, {(table.status != 'complete').sum()} not finished")


if __name__ == "__main__":
    main()
