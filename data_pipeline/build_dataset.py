import argparse
import calendar
import subprocess
import sys
import zipfile
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_ROOT / "data_pipeline"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "era5"
MONTHLY_DIR = PROJECT_ROOT / "data" / "processed" / "weather" / "monthly"
COMBINED_DIR = PROJECT_ROOT / "data" / "processed" / "weather"
def run_month(city, year, month):
    month = f"{month:02d}"
    year = str(year)
    archive = RAW_DIR / f"{city}_{year}_{month}.zip"
    extract_dir = RAW_DIR / f"{city}_{year}_{month}"
    monthly_csv = MONTHLY_DIR / f"{city}_{year}_{month}.csv"
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download ERA5 if the archive is not already present.
    if not archive.exists() or archive.stat().st_size == 0:
        print(f"\nDownloading {city} {year}-{month}...")
        subprocess.run([sys.executable, str(PIPELINE_DIR / "download_era5.py"), city, year, month,], cwd=PROJECT_ROOT, check=True,)
    else:
        print(f"\nArchive already exists: {archive.name}")

    # Step 2: Verify and extract the archive.
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"Invalid or missing ZIP archive: {archive}")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError(f"Corrupt ZIP archive: {archive}")
        z.extractall(extract_dir)

    # Step 3: Prepare the weather CSV if not already prepared.
    if monthly_csv.exists() and monthly_csv.stat().st_size > 0:
        print(f"Monthly CSV already exists; skipping: {monthly_csv.name}")
    else:
        print(f"Preparing {city} {year}-{month}...")
        subprocess.run([sys.executable, str(PIPELINE_DIR / "prepare_era5_weather.py"), str(extract_dir), str(monthly_csv),], cwd=PROJECT_ROOT,check=True,)

    # Validate the monthly output.
    df = pd.read_csv(monthly_csv)
    expected_rows = 24 * calendar.monthrange(int(year), int(month))[1]

    if len(df) != expected_rows:
        raise RuntimeError(f"{monthly_csv.name}: expected {expected_rows} rows, " f"found {len(df)}")
    print(f"Validated {monthly_csv.name}: {len(df)} hourly rows")


def combine_city(city):
    files = sorted(MONTHLY_DIR.glob(f"{city}_????_??.csv"))

    if not files:
        print(f"No monthly CSV files found for {city}.")
        return

    frames = [pd.read_csv(file, parse_dates=["valid_time"]) for file in files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("valid_time").drop_duplicates(subset=["valid_time", "latitude", "longitude"])
    combined = combined.reset_index(drop=True)

    output = COMBINED_DIR / f"{city}_weather.csv"
    combined.to_csv(output, index=False)

    print(f"\nCombined dataset: {output}")
    print(f"Rows: {len(combined)}")
    print(f"Date range: {combined['valid_time'].min()} to "
          f"{combined['valid_time'].max()}")


def main():
    parser = argparse.ArgumentParser(
        description="Download, prepare, and combine monthly ERA5 weather data."
    )
    parser.add_argument("--city", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-month", type=int, default=12)
    args = parser.parse_args()

    # Import the existing city configuration.
    sys.path.insert(0, str(PIPELINE_DIR))
    from cities import CITIES

    if args.city not in CITIES:
        raise ValueError(f"Unknown city '{args.city}'. Available: {', '.join(CITIES)}")

    if not (
        1 <= args.start_month <= 12 and 1 <= args.end_month <= 12 and args.start_month <= args.end_month):
        raise ValueError("Months must be between 1 and 12, in ascending order.")

    for month in range(args.start_month, args.end_month + 1):
        run_month(args.city, args.year, month)
    combine_city(args.city)

if __name__ == "__main__":
    main()