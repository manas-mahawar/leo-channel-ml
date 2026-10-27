# Finalization checklist

Use this list before submitting or demonstrating the project. Run all
commands from the repository root in Windows PowerShell. The virtual
environment `.venv` exists in this workspace.

Status as of 2026-10-27 (see `docs/PROJECT_FINAL_STATUS.md` for evidence):

| Status | Meaning |
|---|---|
| **VERIFIED** | Run or checked, and passing |
| **PRESENT** | The file exists, but its behaviour was not re-checked |
| **N/A** | Not available |
| **NOT RUN** | Intentionally not run (expensive, or would overwrite outputs) |
| **MANUAL** | Needs your action |

## Quick commands (safe; nothing is trained or deleted)

```powershell
git status --short                                         # review your changes (read-only)
.venv\Scripts\python -m unittest discover -s tests         # 41 tests, about 10 s
.venv\Scripts\python evaluation\plot_research_results.py --overwrite --export-docs
                                                           # re-checks the saved metrics and redraws the figures
Get-Content results\causal_hourly_fig1heads_seed42\metrics_test.json -TotalCount 30
```

`plot_research_results.py --overwrite` replaces only its own files in
`results\figures\` and the five copies in `docs\assets\figures\causal\`.

## 1. Repository integrity

| Item | Status | How to check |
|---|---|---|
| Nothing staged or committed by the assistant | VERIFIED | `git diff --cached --stat` prints nothing |
| Your earlier uncommitted edits (10 modified prototype files, `.gitignore`) still present | VERIFIED | `git status --short` shows them as ` M` |
| No secrets in the repository docs | VERIFIED | Docs mention `.cdsapirc` only as a file name; `.env` and `.cdsapirc` are git-ignored |
| A licence chosen | MANUAL | See `GITHUB_CHECKLIST.md` |
| Commit or push | MANUAL | Your decision; the assistant did not stage, commit or push |

## 2. Environment and dependencies

| Item | Status | How to check |
|---|---|---|
| `.venv` with Python 3.12 exists | VERIFIED | `.venv\Scripts\python --version` |
| `requirements.txt` pins all imported packages | PRESENT | Was checked against the installed versions on 2026-10-26 (`STATUS.md`); not re-checked |
| Fresh install on another machine | NOT RUN | `py -3.12 -m venv .venv` then `.venv\Scripts\python -m pip install -r requirements.txt` |

## 3. Data availability and provenance

| Item | Status | Notes |
|---|---|---|
| Hourly weather (`data\processed\weather\open_meteo_v2\`) | PRESENT | Open-Meteo ERA5 archive (CC BY 4.0); git-ignored |
| ITU-R label tables (`data\processed\physics_paper\<city>_physics.csv`) | VERIFIED | The test targets were recomputed from them (`results\why_results_differ\units_check.json`) |
| Data in a fresh clone | N/A | `data\` is git-ignored. Rebuild with `.\run_pipeline.ps1 -Download` (long; NOT RUN) |
| Authors' original dataset | N/A | Only "on request"; not used |

## 4. Model and saved artefacts

| Item | Status | Notes |
|---|---|---|
| `models\causal_hourly_fig1heads_seed42\best.keras`, `model.keras` | PRESENT | Unchanged (checksum snapshot); not reloaded in this session |
| `results\causal_hourly_fig1heads_seed42\test_predictions.npz`, `metrics_test.json`, `run_config.json`, `training_log.csv` | VERIFIED | Metrics recomputed from the predictions match to 1e-9 |
| Retraining | NOT RUN | About 3 h on CPU; `training\train_causal_hourly.py` refuses to overwrite the finished run |

## 5. Evaluation and metrics

| Item | Status | Value |
|---|---|---|
| LSTM RMSE / MAE (causal hourly, 195,000 values) | VERIFIED | 0.8592 / 0.3350 dB |
| Persistence RMSE / MAE | VERIFIED | 1.0181 / 0.3461 dB |
| Improvement over persistence | VERIFIED | RMSE 15.6 %, MAE 3.2 % |
| Heavy-rain RMSE, LSTM vs persistence | VERIFIED | 6.704 vs 5.845 dB |
| 95 % coverage, overall / heavy rain | VERIFIED | 97.4 % / 46.0 % |
| Earlier-protocol audit (`evaluation\audit_results.py`) | VERIFIED | 127/127 PASS, run through a read-only wrapper. Running the script directly rewrites `results\paper\audit_report.json` |

## 6. Visualization outputs

| Item | Status | How to check |
|---|---|---|
| 13 PNG + 13 PDF in `results\figures\` | VERIFIED | Opened all PNGs; checked PDF headers and trailers |
| `results\figures\figures_manifest.json` lists 13 figures | VERIFIED | Valid JSON |
| 5 PNG copies in `docs\assets\figures\causal\` | VERIFIED | Identical to `results\figures\` |
| Figures readable (no overlapping labels) | VERIFIED | Inspected visually |

## 7. Documentation and links

| Item | Status | Notes |
|---|---|---|
| `README.md` | VERIFIED | Metrics match; all relative links and images resolve |
| `docs\VISUAL_RESULTS_GUIDE.md`, `docs\PROJECT_EXPLAINED_FROM_SCRATCH.md`, `docs\PRESENTATION_GUIDE.md`, `docs\PROJECT_FINAL_STATUS.md` | VERIFIED | Link check passed; numbers cross-checked |
| Older reports (`docs\FINAL_RESEARCH_REPORT.md`, `REPLICATION.md`, `STATUS.md`) | PRESENT | They describe the earlier stratified protocol and are labelled as such in the README |

## 8. Tests

| Item | Status | Result |
|---|---|---|
| `.venv\Scripts\python -m unittest discover -s tests` | VERIFIED | 41 tests, OK |
| Syntax of the changed scripts (`python -m py_compile ...`) | VERIFIED | OK |

## 9. Reproducibility

| Level | Status |
|---|---|
| Saved metrics recomputed from saved predictions | VERIFIED |
| Figures regenerated from saved artefacts, byte-identical PNGs | VERIFIED (2026-10-27) |
| Training reproducible from scratch | NOT RUN (deterministic settings; an earlier 2-epoch retrain of another variant matched to 1e-7) |
| Data download and label rebuild from an empty folder | NOT RUN |

## 10. Scientific claims and limitations (say these out loud)

* [ ] Labels are **reconstructed** (ITU-R on ERA5), not measured satellite
      data.
* [ ] This is **not an exact reproduction** of the paper; 0.026 dB is **not
      comparable**.
* [ ] **One seed** only; no significance claim.
* [ ] **Heavy rain**: the LSTM is worse than persistence (6.70 vs 5.85 dB).
* [ ] **Calibration**: 97.4 % overall, but 46 % in heavy rain.
* [ ] **MAE gain**: only 3.2 %, and worse than persistence at +1 h and +2 h.
* [ ] The causal results (39,000 origins) are kept separate from the earlier
      stratified results (8,916 windows).

## 11. Presentation preparation

* [ ] Read `docs\PROJECT_EXPLAINED_FROM_SCRATCH.md`, then
      `docs\PRESENTATION_GUIDE.md`.
* [ ] Build the slides from the figures in `results\figures\`. Do **not** use
      the older `fig1_architecture.png` or `fig2_pipeline.png` unchanged.
* [ ] Rehearse to 9 minutes; practise the five-minute version.
* [ ] Practise the "acknowledge uncertainty" answers.

## 12. Submission and demonstration readiness

* [ ] Laptop plugged in (standby pauses Python processes).
* [ ] Demo command ready: the tests plus `plot_research_results.py`. This
      runs in about 20 seconds, needs no internet and does no training.
* [ ] Decide whether to commit (MANUAL). Check that no `.env` or credential
      files are staged.
* [ ] If you share the repository: note that `data\`, `models\` and the
      prediction files in `results\` are git-ignored. Readers see
      `docs\assets\` plus the small result files listed in
      `GITHUB_CHECKLIST.md` (metrics, the 13 figures, audit outputs).
