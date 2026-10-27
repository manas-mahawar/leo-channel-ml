# Full replication pipeline (Windows PowerShell, from the repository root).
#
#   .\run_pipeline.ps1                # labels -> datasets -> training -> evaluation -> figures/tables -> audit
#   .\run_pipeline.ps1 -Download      # also download missing weather files first (Open-Meteo, no account needed)
#   .\run_pipeline.ps1 -EvaluateOnly  # only re-evaluate existing checkpoints, regenerate tables/figures, audit
#
# Every step writes only its own derived outputs. Dataset rebuilds are
# deterministic (bit-identical on this machine). Training skips finished runs
# (run_config.json present) and resumes an interrupted one, so completed
# checkpoints are never overwritten. Keep the laptop plugged in with the lid
# open while models train.

param([switch]$Download, [switch]$EvaluateOnly)
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"

function Step($name, $arguments) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & $py @arguments
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $name" }
}

if (-not $EvaluateOnly) {
    if ($Download) {
        foreach ($c in "Athens","Bergen","Budapest","Chilbolton","Vigo","Warsaw") {
            foreach ($y in 2020..2024) {
                Step "Weather $c $y" @("data_pipeline\download_openmeteo.py", $c, $y, "data\processed\weather\open_meteo_v2") } }
        foreach ($c in "Berlin","Oslo","Rome","Stockholm","Bucharest","London","Davos","Paris","Marseille","Madrid") {
            foreach ($y in 2023, 2024) { Step "Weather $c $y" @("data_pipeline\download_openmeteo.py", $c, $y) } }
        Step "MSL pressure" @("data_pipeline\download_pressure_msl.py")
    }

    # The LWP proxy fit needs ERA5 cloud-water files from the Copernicus CDS
    # (account required). Without them, the committed coefficients in
    # physics\lwp_proxy_calibration.json are used unchanged.
    if (Test-Path "data\raw\era5") {
        Step "LWP proxy calibration"      @("physics\calibrate_lwp_proxy.py")
    } else {
        Write-Host "`n=== LWP proxy calibration: skipped (no data\raw\era5); using physics\lwp_proxy_calibration.json ===" -ForegroundColor Yellow
    }
    Step "ITU-R labels, training cities"  @("physics\build_attenuation.py")
    Step "ITU-R labels, unseen cities"    @("physics\build_attenuation.py", "--unseen")
    Step "Dataset (main)"                 @("data_pipeline\build_paper_dataset.py")
    Step "Dataset (MSL pressure)"         @("data_pipeline\build_paper_dataset.py", "--msl")
    Step "Dataset (1-minute reading)"     @("data_pipeline\build_minute_dataset.py")
    Step "Dataset figures"                @("evaluation\plot_dataset.py")
    Step "Train all variants (queue)"     @("training\run_queue.py")
    Step "Train repeated seeds (queue)"   @("training\run_queue.py", "training\queue_seeds.txt")
}

Step "Evaluate (six-city test sets)"  @("evaluation\evaluate_paper_model.py")
Step "Unseen cities (main model)"     @("evaluation\evaluate_unseen_cities.py")
Step "Unseen cities (MSL model)"      @("evaluation\evaluate_unseen_cities.py", "multitask_msl")
Step "Unseen cities (Fig. 1 heads)"   @("evaluation\evaluate_unseen_cities.py", "multitask_fig1heads")
Step "Run and seed summary"           @("evaluation\summarize_runs.py")
Step "Calibration by condition"       @("evaluation\calibration_analysis.py")
Step "Minute-experiment diagnostics"  @("evaluation\minute_diagnostics.py")
Step "Figures 1-2"                    @("evaluation\make_diagrams.py")
Step "Figures 3-7 and supporting"     @("evaluation\make_figures.py")
Step "Tables 7-11"                    @("evaluation\make_tables.py")
Step "Independent audit"              @("evaluation\audit_results.py")
Step "Replication gap checks"         @("evaluation\replication_audit_checks.py")
Step "Label sensitivity"              @("evaluation\label_sensitivity.py")
Step "Published vs measured"          @("evaluation\make_paper_comparison.py")
Step "Unit tests"                     @("-m", "unittest", "discover", "-s", "tests", "-v")
Write-Host "`nDone. Results in results\paper\ (tables.md, figures\, metrics_test.json, audit_report.json)." -ForegroundColor Green
