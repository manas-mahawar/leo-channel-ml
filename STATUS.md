# Project status

This file is a dated log; each section carries the date of its own checks
(2026-10-26 and 2026-10-27; the latest entry is from 2026-10-27). The current
final status (causal hourly main result, 41 tests, documentation) is in
[docs/PROJECT_FINAL_STATUS.md](docs/PROJECT_FINAL_STATUS.md).

## State: complete (partial, independent replication; not an exact reproduction)

All planned experiments are finished and audited; documentation, tests and
GitHub preparation are done. Git history and publication state are recorded
by Git itself (`git log`), not in this file.

### Runs (15, all complete, none resumed)

* Seed 42: multitask, no_itu, epl_only, weather_only, multitask_msl,
  minute_multitask, multitask_fig1heads (`models/paper/<variant>/`).
* Seeds 1 and 2: multitask, multitask_fig1heads, epl_only, no_itu
  (`models/paper/seeds/<variant>_seed<N>/`).

### Seed-level summary (test EPL RMSE, dB, seeds 42/1/2)

| Variant | Mean +- std | Range |
|---|---|---|
| multitask_fig1heads (best; selected on validation too) | 0.916 +- 0.006 | 0.910-0.920 |
| multitask | 0.940 +- 0.004 | 0.936-0.943 |
| no_itu | 0.940 +- 0.006 | 0.933-0.945 |
| epl_only | 0.946 +- 0.007 | 0.942-0.954 |
| persistence (reference baseline) | 1.042 | - |
| paper (published reference, not ours) | 0.026 | - |

Machine-readable: `results/paper/experiments_summary.json`, `seed_summary.csv`,
`seed_comparisons.csv` (snapshot in `docs/assets/`).

### Verified checks (2026-10-26)

* `evaluation/audit_results.py`: 127/127 PASS.
* `python -m unittest discover -s tests -v`: 12/12 PASS.
* Seed statistics recomputed independently: match.
* `evaluation/minute_diagnostics.py` reproduces every documented minute-experiment number.
* Documentation links and quoted paths resolve; key numbers in the report match the result files.
* Python syntax check of all pipeline scripts and parse check of `run_pipeline.ps1`: pass.
* `requirements.txt` equals the installed versions; `pip check` clean.
* Model artefacts unchanged since the start of the session (MD5 list in `backups/2026-10-26/`).
* Earlier: determinism (2-epoch retrain = original losses to 1e-7); main
  unseen-city evaluation re-run reproduced exactly; saved predictions equal
  fresh predictions from `best.keras`.

Not verified: a full `run_pipeline.ps1` run from an empty folder, or a fresh
environment on another machine.

### Remaining scientific limitations

Authors' dataset unavailable; our own weak cloud-water proxy; heavy rain
badly under-predicted and intervals under-cover rain; three seeds for four
variants and one for the rest; unseen cities and the minute experiment use
seed 42 only; labels are ITU-R model outputs, not measurements.

### Optional next steps (not required for completion)

* More seeds; repeated seeds for weather_only / multitask_msl / minute_multitask:
  add lines `<variant> <seed>` to `training/queue_seeds.txt`, then
  `.venv\Scripts\python training\run_queue.py training\queue_seeds.txt`,
  then `.\run_pipeline.ps1 -EvaluateOnly`.
* Unseen-city evaluation of the seed-1/2 models (script currently takes seed-42 models).
* Your decisions: choose a code licence, review the old prototype file
  modifications, then stage and commit (see `GITHUB_CHECKLIST.md`).

## Replication gap audit (2026-10-26)

Done: `docs/REPLICATION_GAP_AUDIT.md`, `docs/REPLICATION_EXPERIMENT_PLAN.md`,
`docs/REPLICATION_RESULTS.md`; new read-only analysis scripts
`evaluation/replication_audit_checks.py`, `label_sensitivity.py`,
`make_paper_comparison.py` (outputs in `results/replication_audit/`);
4 new tests (16 total). No implementation bug found; no model retrained;
no existing output overwritten. Paper-faithful architecture:
`multitask_fig1heads`. Only remaining experiment that could change the
comparison: the full-series minute reconstruction (plan stage 7, about
55-70 h CPU; GPU justified for that stage only).

## Where things are

`README.md` (overview), `REPLICATION.md` (full method and results),
`docs/FINAL_RESEARCH_REPORT.md`, `docs/PROJECT_INTERVIEW_NOTES.md`,
`docs/RESUME_BULLETS.md`, `docs/REFERENCES.md`, `GITHUB_CHECKLIST.md`.

## Rain label diagnostic (2026-10-27)

`docs/RAIN_LABEL_DIAGNOSTIC.md`: rain, gas and cloud labels reproduce
independent ITU-R references (P.618 to 1e-15 dB, `itur` P.618 exactly, P.840
K_l exactly, gas within 0.005 dB); **no coding error found**. The label tail
gap with the paper is attributed to unspecified choices (hourly rate as
R_0.01, P.618 time percentage, elevation, rain-rate definition). Labels and
hourly results unchanged (primary baseline). New: `evaluation/rain_label_diagnostic.py`,
`data_pipeline/minute_full_source.py` (lazy full-series minute windows,
473,220 test windows, not trained), 4 tests (20 total).

## Minute reconstruction pipeline (2026-10-27)

See `docs/MINUTE_REPLICATION_IMPLEMENTATION.md`. Lazy loader, train-only
scalers, `minute_full_fig1heads` variant, streaming evaluation; smoke test
(1 % of anchors, 1 epoch, 4 min 46 s) passed end to end
(`results/minute_full_smoke/`, `models/minute_full_smoke/`). Fixed a
split-boundary leak in the unused prototype source (two hourly rows shared).
Full run not started: about 64 h on mains / 125 h on battery CPU; GPU
justified, needs approval. 28 tests pass; audit 127/127; hourly results unchanged.

## Causal forecasting investigation (2026-10-27)

See `docs/CAUSAL_FORECASTING_INVESTIGATION.md`. The paper's text states a
5-hour horizon on hourly data; the minute protocol's tiny errors come from
interpolation using the next hour (removing it raises persistence 10.6x and
linear extrapolation 37x on identical anchors and targets). New strictly
causal protocol `causal_hourly_v1` (all hourly origins, 39,000 test
origins): persistence 1.018 dB, linear extrapolation 2.519 dB, 24-h mean
1.126 dB (`results/causal_baseline_audit/`). No model trained.

## Causal hourly model, seed 42 (2026-10-27)

See `docs/CAUSAL_HOURLY_MODEL_EXPERIMENT.md`. `multitask_fig1heads` on
`causal_hourly_v1` (11 epochs, best 5, 3.1 h): test RMSE 0.859 dB vs
persistence 1.018 dB (-15.6 %), MAE 0.335 vs 0.346 dB (-3.2 %) on identical
39,000 origins; worse than persistence in heavy rain and in MAE at +1/+2 h.
Outputs: `results/causal_hourly_fig1heads_seed42/`, `models/causal_hourly_fig1heads_seed42/`.
