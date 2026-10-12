import argparse
import calendar
import os
import subprocess
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_ROOT / "data_pipeline"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "era5"
MONTHLY_DIR = PROJECT_ROOT / "data" / "processed" / "weather" / "monthly"
COMBINED_DIR = PROJECT_ROOT / "data" / "processed" / "weather"

# Columns written by prepare_era5_weather.py, in order.
REQUIRED_COLUMNS = ["valid_time", "latitude", "longitude", "temperature_C", "relative_humidity_percent", "pressure_hPa", "precipitation_mm", "cloud_cover_percent", "tclw",]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]


def remove_partial(path):
    """Best-effort removal of a temporary file; never raises, so the original error is kept."""
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        print(f"Warning: could not remove temporary file {path}: {error}")


def expected_hours(year, month):
    days = calendar.monthrange(year, month)[1]
    return pd.date_range(f"{year}-{month:02d}-01", periods=24 * days, freq="h")


def validate_monthly_csv(path, year, month):
    """Return the parsed monthly DataFrame, or raise ValueError explaining why it is invalid."""
    df = pd.read_csv(path)
    name = Path(path).name

    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"{name}: unexpected columns {list(df.columns)}")

    df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")

    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")
        non_numeric = df[column][values.isna() & df[column].notna()]
        if not non_numeric.empty:
            raise ValueError(f"{name}: non-numeric values in column '{column}', e.g. {non_numeric.iloc[0]!r}")
        non_finite = values[np.isinf(values)]
        if not non_finite.empty:
            raise ValueError(f"{name}: non-finite values in column '{column}', e.g. {non_finite.iloc[0]}")
        df[column] = values

    nan_counts = df[REQUIRED_COLUMNS].isna().sum()
    if nan_counts.any():
        raise ValueError(f"{name}: missing/unparseable values {nan_counts[nan_counts > 0].to_dict()}")

    duplicates = df["valid_time"][df["valid_time"].duplicated()]
    if not duplicates.empty:
        raise ValueError(f"{name}: {len(duplicates)} duplicate timestamps, e.g. {duplicates.iloc[0]}")

    expected = expected_hours(year, month)
    actual = pd.DatetimeIndex(df["valid_time"])
    missing = expected.difference(actual)
    unexpected = actual.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(f"{name}: {len(missing)} missing and {len(unexpected)} unexpected timestamps "
                         f"(first missing: {missing[:1].tolist()}, first unexpected: {unexpected[:1].tolist()})")

    grid_points = df[["latitude", "longitude"]].drop_duplicates()
    if len(grid_points) != 1:
        raise ValueError(f"{name}: expected 1 grid point, found {len(grid_points)}: {grid_points.values.tolist()}")

    return df


def run_month(city, year, month):
    month = f"{month:02d}"
    year = str(year)
    archive = RAW_DIR / f"{city}_{year}_{month}.zip"
    extract_dir = RAW_DIR / f"{city}_{year}_{month}"
    monthly_csv = MONTHLY_DIR / f"{city}_{year}_{month}.csv"
    partial_csv = MONTHLY_DIR / f"{city}_{year}_{month}.partial.csv"
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

    # Step 0: Reuse an existing monthly CSV only if it passes validation.
    if monthly_csv.exists():
        try:
            df = validate_monthly_csv(monthly_csv, int(year), int(month))
            print(f"\nValidated existing {monthly_csv.name}: {len(df)} hourly rows")
            return
        except ValueError as error:
            print(f"\nExisting monthly CSV is invalid; rebuilding. Reason: {error}")

    # Step 1: Download ERA5 if the archive is not already present.
    if not archive.exists() or archive.stat().st_size == 0:
        print(f"\nDownloading {city} {year}-{month}...")
        subprocess.run([sys.executable, str(PIPELINE_DIR / "download_era5.py"), city, year, month,], cwd=PROJECT_ROOT, check=True,)
    else:
        print(f"\nArchive already exists: {archive.name}")

    # Step 2: Verify and extract the archive.
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"Invalid or missing ZIP archive: {archive}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError(f"Corrupt ZIP archive: {archive}")
        z.extractall(extract_dir)

    # Step 3: Prepare into a temporary file, validate it, then atomically replace the final CSV.
    print(f"Preparing {city} {year}-{month}...")
    try:
        subprocess.run([sys.executable, str(PIPELINE_DIR / "prepare_era5_weather.py"), str(extract_dir), str(partial_csv),], cwd=PROJECT_ROOT,check=True,)
        df = validate_monthly_csv(partial_csv, int(year), int(month))
        os.replace(partial_csv, monthly_csv)
    except (subprocess.CalledProcessError, ValueError) as error:
        raise RuntimeError(f"Failed to build valid {monthly_csv.name}: {error}") from error
    finally:
        # No-op after a successful replace; otherwise the previous monthly CSV is left untouched.
        remove_partial(partial_csv)

    print(f"Validated {monthly_csv.name}: {len(df)} hourly rows")


def combine_city(city):
    files = sorted(MONTHLY_DIR.glob(f"{city}_????_??.csv"))

    if not files:
        print(f"No monthly CSV files found for {city}.")
        return

    # Re-validate every month on disk, not only the months requested in this run.
    frames = []
    for file in files:
        year, month = (int(part) for part in file.stem.split("_")[-2:])
        frames.append(validate_monthly_csv(file, year, month))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("valid_time").reset_index(drop=True)

    grid_points = combined[["latitude", "longitude"]].drop_duplicates()
    if len(grid_points) != 1:
        raise RuntimeError(f"{city}: months use different grid points: {grid_points.values.tolist()}")

    if combined["valid_time"].duplicated().any():
        raise RuntimeError(f"{city}: duplicate timestamps across monthly files")

    # The combined timeline must be continuous, so a missing month is an error.
    timeline = pd.date_range(combined["valid_time"].min(), combined["valid_time"].max(), freq="h")
    if not pd.DatetimeIndex(combined["valid_time"]).equals(timeline):
        missing = timeline.difference(combined["valid_time"])
        raise RuntimeError(f"{city}: combined timeline has {len(missing)} missing hours, "
                           f"first missing: {missing[:1].tolist()}")

    combined.insert(0, "city", city)

    output = COMBINED_DIR / f"{city}_weather.csv"
    partial_output = COMBINED_DIR / f"{city}_weather.partial.csv"
    try:
        combined.to_csv(partial_output, index=False)
        os.replace(partial_output, output)
    finally:
        # No-op after a successful replace; otherwise the previous combined CSV is left untouched.
        remove_partial(partial_output)

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