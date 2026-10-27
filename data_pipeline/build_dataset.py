
import argparse
import calendar
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_ROOT / "data_pipeline"

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "era5"
MONTHLY_DIR = PROJECT_ROOT / "data" / "processed" / "weather" / "monthly"
WEATHER_DIR = PROJECT_ROOT / "data" / "processed" / "weather"

WEATHER_COLUMNS = [
    "valid_time",
    "latitude",
    "longitude",
    "temperature_C",
    "relative_humidity_percent",
    "pressure_hPa",
    "precipitation_mm",
    "cloud_cover_percent",
    "tclw",
]

NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "temperature_C",
    "relative_humidity_percent",
    "pressure_hPa",
    "precipitation_mm",
    "cloud_cover_percent",
    "tclw",
]


def validate_monthly_csv(path, year, month):
    """Validate one complete month of hourly weather data."""

    path = Path(path)
    name = path.name

    if not path.exists():
        raise FileNotFoundError(f"Monthly CSV not found: {path}")

    df = pd.read_csv(path)

    missing_columns = set(WEATHER_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{name}: missing columns: {sorted(missing_columns)}"
        )

    df = df[WEATHER_COLUMNS].copy()

    # Validate timestamps.
    df["valid_time"] = pd.to_datetime(
        df["valid_time"], errors="raise"
    )

    if df["valid_time"].isna().any():
        raise ValueError(f"{name}: missing timestamps found.")

    duplicates = df.loc[
        df["valid_time"].duplicated(keep=False), "valid_time"
    ]

    if not duplicates.empty:
        raise ValueError(
            f"{name}: {len(duplicates)} rows have duplicate timestamps; "
            f"example: {duplicates.iloc[0]}"
        )

    # Validate numeric columns.
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df[NUMERIC_COLUMNS].isna().any().any():
        raise ValueError(f"{name}: missing numeric values found.")

    if not np.isfinite(df[NUMERIC_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: non-finite numeric values found.")

    # Exactly one grid point must represent the city.
    if df["latitude"].nunique() != 1 or df["longitude"].nunique() != 1:
        raise ValueError(
            f"{name}: expected one ERA5 grid point, but found "
            f"{df['latitude'].nunique()} latitudes and "
            f"{df['longitude'].nunique()} longitudes."
        )

    # Verify the full month is present at hourly frequency.
    days_in_month = calendar.monthrange(year, month)[1]
    expected_count = days_in_month * 24

    expected_times = pd.date_range(
        start=f"{year}-{month:02d}-01 00:00:00",
        periods=expected_count,
        freq="h",
    )

    df = df.sort_values("valid_time").reset_index(drop=True)

    if len(df) != expected_count:
        raise ValueError(
            f"{name}: expected {expected_count} hourly rows, "
            f"found {len(df)}."
        )

    if not df["valid_time"].reset_index(drop=True).equals(
        pd.Series(expected_times, name="valid_time")
    ):
        raise ValueError(
            f"{name}: timestamps do not match the complete expected "
            f"hourly timeline for {year}-{month:02d}."
        )

    return df


def run_month(city, year, month):
    """Download, prepare, validate, and save one city-month."""

    year = int(year)
    month = int(month)

    if not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12.")

    year_str = str(year)
    month_str = f"{month:02d}"

    archive = RAW_DIR / f"{city}_{year_str}_{month_str}.zip"
    extract_dir = RAW_DIR / f"{city}_{year_str}_{month_str}"

    monthly_file = MONTHLY_DIR / f"{city}_{year_str}_{month_str}.csv"
    partial_csv = MONTHLY_DIR / (
        f"{city}_{year_str}_{month_str}.partial.csv"
    )

    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse an existing monthly CSV only after full validation.
    if monthly_file.exists():
        try:
            validate_monthly_csv(monthly_file, year, month)
            print(f"Validated monthly CSV already exists: {monthly_file}")
            return
        except (ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
            print(f"Existing monthly CSV failed validation: {exc}")
            print("Rebuilding this city-month.")

    print(f"\nDownloading {city} {year_str}-{month_str}...")

    # download_era5.py skips valid archives and replaces invalid ones.
    if not archive.exists() or not zipfile.is_zipfile(archive):
        subprocess.run(
            [
                sys.executable,
                str(PIPELINE_DIR / "download_era5.py"),
                city,
                year_str,
                month_str,
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    if not archive.exists():
        raise FileNotFoundError(
            f"ERA5 archive was not created: {archive}"
        )

    # Validate the archive before extracting it.
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"ERA5 archive is not a valid ZIP: {archive}")

    with zipfile.ZipFile(archive) as zf:
        corrupt_member = zf.testzip()
        if corrupt_member is not None:
            raise ValueError(
                f"Corrupt member in {archive}: {corrupt_member}"
            )

        zf.extractall(extract_dir)

    print(f"Preparing {city} {year_str}-{month_str}...")

    partial_csv.unlink(missing_ok=True)

    try:
        # The third argument tells the preparation script which city's
        # nearest ERA5 grid point to select.
        subprocess.run(
            [
                sys.executable,
                str(PIPELINE_DIR / "prepare_era5_weather.py"),
                str(extract_dir),
                str(partial_csv),
                city,
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        df = validate_monthly_csv(partial_csv, year, month)

        # Publish the CSV only after validation succeeds.
        os.replace(partial_csv, monthly_file)

        print(
            f"Validated and saved {city} {year_str}-{month_str}: "
            f"{len(df)} hourly rows."
        )

    finally:
        partial_csv.unlink(missing_ok=True)


def combine_city(city):
    """Combine available monthly CSVs for one city into a weather dataset."""

    files = sorted(MONTHLY_DIR.glob(f"{city}_????_??.csv"))

    if not files:
        print(f"No monthly CSV files found for {city}; skipping combination.")
        return

    frames = []

    for file in files:
        # Parse year and month from City_YYYY_MM.csv.
        parts = file.stem.rsplit("_", 2)

        if len(parts) != 3:
            raise ValueError(f"Unexpected monthly filename: {file.name}")

        file_city, year_str, month_str = parts

        if file_city != city:
            continue

        df = validate_monthly_csv(
            file,
            int(year_str),
            int(month_str),
        )
        frames.append(df)

    if not frames:
        print(f"No valid monthly datasets found for {city}.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("valid_time").reset_index(drop=True)

    # Each month has one grid point; all months must share the same one.
    # (Older raw archives cover a smaller area than newer ones.)
    grid_points = combined[["latitude", "longitude"]].drop_duplicates()
    if len(grid_points) != 1:
        raise ValueError(
            f"{city}: months use different ERA5 grid points: "
            f"{grid_points.values.tolist()}"
        )

    if combined["valid_time"].duplicated().any():
        raise ValueError(
            f"{city}: duplicate timestamps found when combining months."
        )

    # Confirm that the available records form one uninterrupted timeline.
    expected_timeline = pd.date_range(
        start=combined["valid_time"].iloc[0],
        end=combined["valid_time"].iloc[-1],
        freq="h",
    )

    if not combined["valid_time"].reset_index(drop=True).equals(
        pd.Series(expected_timeline, name="valid_time")
    ):
        raise ValueError(
            f"{city}: gaps found in the combined hourly timeline."
        )

    combined.insert(0, "city", city)

    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    output_file = WEATHER_DIR / f"{city}_weather.csv"
    partial_file = WEATHER_DIR / f"{city}_weather.partial.csv"

    # Write to a temporary file first so a failed write keeps the old CSV.
    try:
        combined.to_csv(partial_file, index=False)
        os.replace(partial_file, output_file)
    finally:
        partial_file.unlink(missing_ok=True)

    print(f"\nCombined weather dataset saved: {output_file}")
    print(f"Rows: {len(combined)}")
    print(
        f"Period: {combined['valid_time'].iloc[0]} "
        f"to {combined['valid_time'].iloc[-1]}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build validated hourly ERA5 weather datasets "
            "for the selected city and months."
        )
    )

    parser.add_argument("--city", required=True, help="City name")
    parser.add_argument("--year", type=int, required=True, help="Year")
    parser.add_argument(
        "--start-month",
        type=int,
        required=True,
        help="First month, from 1 to 12",
    )
    parser.add_argument(
        "--end-month",
        type=int,
        required=True,
        help="Last month, from 1 to 12",
    )

    args = parser.parse_args()

    if not 1 <= args.start_month <= 12:
        parser.error("--start-month must be between 1 and 12.")

    if not 1 <= args.end_month <= 12:
        parser.error("--end-month must be between 1 and 12.")

    if args.start_month > args.end_month:
        parser.error("--start-month cannot exceed --end-month.")

    # Import the city list from the existing project configuration.
    sys.path.insert(0, str(PIPELINE_DIR))
    from cities import CITIES

    if args.city not in CITIES:
        parser.error(
            f"Unknown city '{args.city}'. Available cities: "
            f"{', '.join(CITIES)}"
        )

    try:
        for month in range(args.start_month, args.end_month + 1):
            run_month(args.city, args.year, month)

        combine_city(args.city)

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
