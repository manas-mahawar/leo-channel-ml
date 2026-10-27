# GitHub readiness checklist

Prepared 2026-10-27, after the causal hourly experiment. On that date the
curated files in Sect. 2 were staged with the commands in Sect. 9, for the
first publication commit. For the current state, ask Git (`git status`,
`git log`, `git remote -v`); this file does not track it.

## 1. Summary

* **Ready for the first commit.**
  * The commands in Sect. 9 stage 139 files (7.3 MB). With the 16 files
    already committed, the repository will hold 155 files.
  * `git add -A` would instead add the prototypes in Sect. 4 as well; do not
    use it.
  * The largest file is 0.69 MB, so a normal push is safe and Git LFS is not
    needed.
* **No credentials** in the files to publish or in the Git history (Sect. 6).
  Your commit e-mail address becomes public when you push (Sect. 11).
* **Nothing was deleted or moved.** The only configuration changes are
  `.gitignore` (Sect. 5) and a one-line `.gitattributes` (Sect. 5), plus
  wording updates in several documents (Sect. 8).
* **Still your decisions:** a licence, the earlier prototype files, and the
  commit e-mail (Sect. 4 and 11).

## 2. Files to commit

Git ignores everything that should stay local, so the folder-level commands
in Sect. 9 add only the files below.

| Group | Files |
|---|---|
| Root | `.gitignore`, `.gitattributes`, `README.md`, `REPLICATION.md`, `STATUS.md`, `GITHUB_CHECKLIST.md`, `requirements.txt`, `config.py`, `run_pipeline.ps1` |
| `data_pipeline/` | `cities.py` (tracked; its uncommitted change adds the ten unseen cities and **is needed** by the pipeline), `download_openmeteo.py`, `download_pressure_msl.py`, `build_paper_dataset.py`, `build_minute_dataset.py`, `causal_forecast_source.py`, `minute_full_source.py`, `minute_full_loader.py` |
| `physics/` | `build_attenuation.py`, `calibrate_lwp_proxy.py`, `lwp_proxy_calibration.json` (fitted coefficients; needed to rebuild labels without a CDS account) |
| `training/` | `train_causal_hourly.py`, `train_paper_model.py`, `train_minute_full.py`, `run_queue.py`, `queue.txt`, `queue_seeds.txt` |
| `evaluation/` | all 20 scripts |
| `tests/` | `test_pipeline.py`, `test_causal.py` (41 tests, including the leakage tests) |
| `docs/` | all 17 documents and `docs/assets/` (curated figures and tables, 2.8 MB) |
| `results/figures/` | the 13 figures as PNG + PDF, and `figures_manifest.json` (2.9 MB) |
| `results/causal_hourly_fig1heads_seed42/` | `metrics_test.json` (source of the main result), `run_config.json`, `training_log.csv`, `training_curves.png/.pdf` |
| `results/causal_baseline_audit/` | `baseline_metrics.json`, `config.json` |
| `results/why_results_differ/` | the 5 small outputs behind `docs/WHY_RESULTS_DIFFER_FROM_PAPER.md` |
| `results/replication_audit/` | audit checks, label sensitivity, minute-protocol baselines, rain-label diagnostic, paper comparison (about 190 kB) |

Every relative link in these Markdown files resolves to a file in this list
(checked; see Sect. 7).

## 3. Kept locally (git-ignored)

| Path | Size | Why it stays local |
|---|---|---|
| `.venv/` | 2.3 GB | local environment; rebuild from `requirements.txt` |
| `data/` (raw 32 MB, processed 912 MB) | 943 MB | downloaded or generated; Open-Meteo / CDS terms; regenerable |
| `models/` (`*.keras` checkpoints, predictions) | 613 MB | 16-18 MB per checkpoint; regenerable by training |
| `results/` except the files in Sect. 2 | about 87 MB | prediction arrays (`*.npz`, up to 10.8 MB), per-window CSVs (about 5 MB each), scalers; regenerable |
| `backups/` | 103 MB | local safety copies cited in `REPLICATION.md`; keep them |
| `training/*.npy`, `training/*.pkl` | 3.9 MB | arrays and scalers of the early prototype |
| `__pycache__/`, `*.pyc`, `*.log` | small | caches and logs |
| `.env`, `.env.*`, `.cdsapirc` | none present | credential files; the CDS key is in your home folder, outside the repository |

If you want to share checkpoints or predictions, attach them to a GitHub
Release or use another file host rather than the repository. Git LFS would
also work, but it has storage and bandwidth quotas and adds a tool that
cloners must install, which is not worth it for optional files.

## 4. Your decisions

1. **Licence (unresolved).** No licence has been chosen. Without a `LICENSE`
   file, others may read the code but have no permission to reuse it. MIT is
   common for portfolio code; the choice is yours.
2. **Earlier tracked prototypes with uncommitted edits:**
   * Files: `data_pipeline/build_dataset.py`, `create_features.py`,
     `create_sequences.py`, `download_era5.py`, `prepare_era5_weather.py`,
     and `physics/cloud_attenuation.py`, `gaseous_attenuation.py`,
     `rain_attenuation.py`.
   * They are not used by the current pipeline.
   * Commit, keep local, or revert them yourself after reviewing
     `git diff -- data_pipeline physics`.
3. **Untracked prototypes (Athens January experiments, ERA5 CDS download):**
   * Files: `data_pipeline/download_era5_group.py`, `merge_physics.py`,
     `run_batch.py`, `physics/check_itur_api.py`,
     `models/multitask_january_proxy.py`, `training/evaluate_*_baseline.py`,
     `training/inspect_*.py`, `training/prepare_january_*.py`.
   * Nothing imports them, and the README does not refer to them.
   * Committing them shows the project history. Leaving them out gives a
     smaller repository.
   * The commands in Sect. 9 leave them out.
4. **Recommended to leave out:**

   | File | What it is | Referenced by |
   |---|---|---|
   | `compiler_test.cpp` | C++ "hello world" for testing a compiler | only this checklist |
   | `check_progress.py` | counts the prototype's monthly ERA5 CSVs | only this checklist |
   | `validate_openmeteo.py` | checks the first Open-Meteo download (`open_meteo/`) | only this checklist |
   | `models/multitask_january_proxy_history.json` | prototype training history | only the prototype script |

   They were not deleted. Delete them or add them to `.gitignore` if you
   prefer.
5. **Outdated diagrams (no change made):**
   * `docs/assets/figures/fig1_architecture.png` and `fig2_pipeline.png`
     show the earlier protocol.
   * They are kept because the historical documents cite them.
   * Do not use them for the causal result.
6. **Duplicate figures (kept on purpose):**
   * Five PNGs in `docs/assets/figures/causal/` are byte copies of
     `results/figures/` (about 0.6 MB).
   * The README embeds them, and `plot_research_results.py --export-docs`
     regenerates them.
7. **Empty folders:** `experiments/` and `notebooks/` are empty. Git ignores
   empty folders, so they do not affect GitHub.

## 5. `.gitignore` changes made in this audit

* `results/` became `results/*`. Git cannot re-include files inside an
  ignored directory, so this change is needed for the allow-list below.
* Appended an allow-list:

  ```
  !results/figures/
  !results/causal_hourly_fig1heads_seed42/
  !results/causal_baseline_audit/
  !results/why_results_differ/
  !results/replication_audit/
  !results/causal_hourly_fig1heads_seed42/training_log.csv
  !results/why_results_differ/*.csv
  !results/replication_audit/*.csv
  ```

  The global `*.npz` and `*.pkl` rules still exclude predictions and scalers
  in those folders (checked with `git check-ignore`).
* Your earlier, uncommitted `.gitignore` lines (backups, `models/paper/`,
  `*.npz`, `docs/assets/*.csv`, caches, logs) are unchanged.

Without this change, a clone would get neither the source of the main result
(`metrics_test.json`) nor 8 of the 13 figures, although the README and the
documentation cite them.

### `.gitattributes` (one line)

```
results/causal_hourly_fig1heads_seed42/training_log.csv -whitespace
```

* **Why:** Keras's `CSVLogger` opens its file without `newline=""`, so on
  Windows every line ends in `\r\r\n`.
  * Git therefore stores the file byte-for-byte as non-text.
  * `git diff --check` reports the extra `\r` as trailing whitespace (12
    warnings).
* **Effect:** the attribute turns off whitespace checks for this one file
  only. Its bytes and logged values are unchanged.
* **Tested** in a separate scratch repository with the real file:
  * 12 warnings without the attribute, 0 with it;
  * 0 warnings after a commit (`git show --check`);
  * the stored file is byte-identical to the original.

## 6. Secrets and privacy audit

* **Credential files** (`.env*`, `.cdsapirc`, `*.pem`, `*.key`, `id_rsa*`,
  `.netrc`, `*credential*`, `*secret*`): none in the repository. A
  `.cdsapirc` exists only in your home folder and is never copied.
* **Content scan** of all files to publish and all tracked files (patterns:
  API key, secret, token, password, bearer, CDS key format). The only matches
  are:
  * `import cdsapi` / `cdsapi.Client()`, which read the key from the home
    folder at run time;
  * mentions of the file name `.cdsapirc` in documentation.

  No values were found.
* **Git history** (2 commits, 18 files): no credential files or key
  assignments. History rewriting is not needed.
* **Personal data:** no absolute paths, user names or e-mail addresses in the
  files to publish, including the result JSON files and the figure manifest.
  The commit author name and e-mail come from your Git configuration.
* **No `.env.example`:** the code reads no environment variables, so none is
  needed.

## 7. Validation (run 2026-10-27, results as observed)

| Check | Command or method | Result |
|---|---|---|
| Unit and leakage tests | `.venv\Scripts\python -m unittest discover -s tests` | 41 tests, OK |
| Tests in a simulated clone (only the files to publish, copied to a temp folder) | same command there | 41 tests, OK: 25 passed, 16 skipped ("dataset / physics tables / results not built"), 0 failures |
| Python syntax | `ast.parse` on all 66 `.py` files Git could publish (prototypes included) | 0 errors |
| Markdown links and images | relative links in the 23 Markdown files to publish | 44 links, all point to files that will be committed |
| Figures | 13 PNG + PDF pairs: signatures, PDF end marker, manifest | 13 / 13 valid, manifest has 13 entries, all publishable |
| Results unchanged | MD5 of 635 protected files (results, metrics, predictions, figures, checkpoints, datasets) before and after | identical |
| User edits unchanged | per-file `git diff` before and after | identical except the `.gitignore` allow-list |
| Staging | `git diff --cached --name-only` compared with the Sect. 9 plan | 139 files, identical to the plan; prototypes unstaged |
| Whitespace | `git diff --check`, `git diff --cached --check` | no warnings (after `.gitattributes`) |
| No training or evaluation | | none run; only tests and read-only checks |

## 8. Other changes in this audit

These are wording updates only, so that the documents match the new
`.gitignore`:

* `README.md`: the "Verify" and "Reproduce" notes, and the repository-layout
  table.
* `REPLICATION.md`: the "Version control" note.
* `docs/PROJECT_FINAL_STATUS.md`: unresolved issue 4.
* `docs/FINALIZATION_CHECKLIST.md`: the sharing note.
* `docs/PROJECT_INTERVIEW_NOTES.md`: the historical 30-second pitch said
  "I replicated", which contradicted the note at the top of that file. It now
  says "independently implemented"; the numbers are unchanged.
* `README.md`, "Reproduce from scratch":
  * the committed output folders trip the scripts' overwrite guards in a
    fresh clone, so the section shows how to move them aside first;
  * the causal experiment needs the earlier pipeline first.
* `README.md`: the sentence about older prototype scripts now describes what
  is actually in the repository.
* `STATUS.md`: the header date was inconsistent with the sections, and the
  file said nothing had been committed. Both statements were removed, and Git
  is the record of commits.
* `REPLICATION.md`: the title now reads "Partial, independent replication …
  (not an exact reproduction)". A note says which referenced outputs and
  backups are not on GitHub.

## 9. Staging commands (run on 2026-10-27; repeat only after changes)

Review:

```powershell
git status
git diff --cached -- .gitignore data_pipeline/cities.py   # changes to committed files
git diff -- data_pipeline physics                          # your prototype edits (Sect. 4.2), not staged
```

Stage (leaves out the prototypes and the files in Sect. 4.4):

```powershell
git add .gitignore .gitattributes README.md REPLICATION.md STATUS.md GITHUB_CHECKLIST.md requirements.txt config.py run_pipeline.ps1
git add data_pipeline/cities.py data_pipeline/download_openmeteo.py data_pipeline/download_pressure_msl.py data_pipeline/build_paper_dataset.py data_pipeline/build_minute_dataset.py data_pipeline/causal_forecast_source.py data_pipeline/minute_full_source.py data_pipeline/minute_full_loader.py
git add physics/build_attenuation.py physics/calibrate_lwp_proxy.py physics/lwp_proxy_calibration.json
git add training/train_causal_hourly.py training/train_paper_model.py training/train_minute_full.py training/run_queue.py training/queue.txt training/queue_seeds.txt
git add evaluation tests docs results
```

Check before committing:

```powershell
git status --short
git diff --cached --stat
git diff --cached --check                     # must print nothing
git diff --cached --name-only | Select-String '\.env|cdsapirc|\.keras|\.h5|\.npz|\.npy|\.pkl|\.joblib|id_rsa|\.pem|\.key'   # must print nothing
```

The commit and push commands are in Sect. 11.

## 10. Licensing and data provenance

* **Weather data:** Open-Meteo, CC BY 4.0. Attribution is given in
  `README.md` and `docs/REFERENCES.md`. Only derived figures and metrics are
  published, not the data.
* **ERA5 / Copernicus:** raw CDS files are not redistributed. Re-fitting the
  LWP proxy needs a free CDS account.
* **The paper:** CC BY-NC-ND 4.0. It is cited; its PDF and figures are not
  included, and all figures here are our own.
* **`itur`:** MIT licence, installed from PyPI and not vendored.
* **Labels:** the labels are computed with ITU-R models, not measured on a
  satellite link. The project is an independent implementation, not an exact
  reproduction of the paper.

## 11. Remaining steps before the first push

- [ ] **Licence:** choose one. If you add a `LICENSE` file, stage it with
  `git add LICENSE`. Without one, others may read but not reuse the code.
- [ ] **Commit e-mail:**
  * Your Git e-mail is a personal address, and the two existing commits
    already contain it. Pushing makes it public.
  * If GitHub's "Block command line pushes that expose my email" setting is
    on, the push will be rejected.
  * Options:
    * accept it;
    * for new commits, set a GitHub noreply address with
      `git config user.email "<id>+<user>@users.noreply.github.com"`;
    * rewriting the two existing commits would change history, so decide
      that separately.
- [ ] **Review the staged changes:** `git diff --cached --stat`, then
  `git diff --cached`.
- [ ] **Run the pre-commit checks** in Sect. 9 (both must print nothing).
- [ ] **Commit:**
  `git commit -m "Add causal hourly LEO EPL forecasting pipeline, results and documentation"`
- [ ] **Create an empty repository on GitHub** (website, or
  `gh repo create leo-channel-ml --private` / `--public`). Do not add a
  README, `.gitignore` or licence there, so that the first push does not
  conflict.
- [ ] **Push:**

  ```powershell
  git remote add origin https://github.com/<your-user>/leo-channel-ml.git
  git push -u origin main
  ```

  This skips `gh repo create … --source . --push`, which would add the remote
  and push in one step.
- [ ] **After the push:**
  * check on GitHub that the README's Mermaid diagram and its images render;
  * check that `results/figures/` shows 13 PNG + 13 PDF files;
  * check that no `data/`, `models/` or `backups/` folder appears.
- [ ] **Left local on purpose:** the 8 modified prototype files and the 15
  untracked files in Sect. 4 stay unstaged until you decide otherwise.
