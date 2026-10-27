"""Copy a small, curated snapshot of the generated results into docs/assets/.

results/ is not version-controlled (it is large and fully regenerable). This
snapshot keeps the key figures and summary tables visible in the repository
for readers who do not run the pipeline. Re-run after regenerating results.

Usage:
    python evaluation/export_docs_assets.py
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

RES = config.RESULTS_DIR
OUT = ROOT / "docs" / "assets"
FIGURES = ["fig1_architecture", "fig2_pipeline", "fig3_pred_vs_true", "fig4_horizon_timeseries",
           "fig5_unseen_cities", "fig7_unseen_map", "ablations", "seed_variation", "minute_experiment",
           "msl_vs_surface_unseen", "calibration", "baselines_and_conditions", "error_vs_horizon"]
FILES = ["tables.md", "experiments_summary.json", "seed_summary.csv", "seed_comparisons.csv",
         "calibration_conditional.csv", "audit_report.json", "table8_unseen_summary.csv",
         "table8_unseen_summary_multitask_msl.csv", "minute_diagnostics.json"]


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        shutil.copy2(RES / "figures" / f"{name}.png", OUT / "figures" / f"{name}.png")
    for name in FILES:
        shutil.copy2(RES / name, OUT / name)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"Copied {len(FIGURES)} figures and {len(FILES)} result files to {OUT} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
