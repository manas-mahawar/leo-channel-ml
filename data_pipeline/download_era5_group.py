
import calendar
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cdsapi
import pandas as pd

from cities import CITIES
from build_dataset import validate_monthly_csv


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "era5" / "batches"
MONTHLY_DIR = ROOT / "data" / "processed" / "weather" / "monthly"
PREPARE_SCRIPT = ROOT / "data_pipeline" / "prepare_era5_weather.py"

DATASET = "reanalysis-era5-single-levels"

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "total_precipitation",
    "total_cloud_cover",
    "total_column_cloud_liquid_water",
]

EXPECTED_FILES = {
    "data_stream-oper_stepType-instant.nc",
    "data_stream-oper_stepType-accum.nc",
}



def valid_archive(path):
    """Validate the ZIP and both expected NetCDF files."""
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                return False

            names = archive.namelist()

            for name in EXPECTED_FILES:
                # The existing preparation script expects files at the ZIP root.
                if name not in names:
                    return False

                with archive.open(name) as file:
                    signature = file.read(8)

                if not signature.startswith((b"CDF", b"\x89HDF\r\n\x1a\n")):
                    return False

            return True

    except (OSError, zipfile.BadZipFile, RuntimeError):
        return False


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python data_pipeline/download_era5_group.py "
            "<city> <year> <months>\n"
            "Example: python data_pipeline/download_era5_group.py "
            "Bergen 2021 05,07"
        )
        return 2

    city = sys.argv[1]
    year = int(sys.argv[2])

    try:
        months = sorted({int(value) for value in sys.argv[3].split(",")})
    except ValueError:
        print("Months must be comma-separated numbers, e.g. 05,07")
        return 2

    if city not in CITIES:
        print(f"Unknown city: {city}")
        print("Available cities:", ", ".join(CITIES.keys()))
        return 2

    if not months or any(month < 1 or month > 12 for month in months):
        print("Months must be between 01 and 12.")
        return 2

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

    # Keep only months whose existing CSV is not already valid.
    missing = []
    for month in months:
        target = MONTHLY_DIR / f"{city}_{year}_{month:02d}.csv"
        if target.exists():
            try:
                validate_monthly_csv(target, year, month)
                print(f"SKIP: valid CSV already exists: {target.name}")
                continue
            except Exception as exc:
                print(f"Existing CSV needs rebuilding: {target.name} ({exc})")
        missing.append(month)

    if not missing:
        print("All requested months already have valid CSVs.")
        return 0

    # A single ERA5 request needs a consistent set of calendar days.
    day_counts = {
        calendar.monthrange(year, month)[1]
        for month in missing
    }
    if len(day_counts) != 1:
        print("These months have different numbers of days.")
        print("Group months with the same number of days in separate requests.")
        print("Months still needing data:", ", ".join(f"{m:02d}" for m in missing))
        return 2

    days_in_month = day_counts.pop()
    month_labels = "_".join(f"{month:02d}" for month in missing)
    archive_path = RAW_DIR / f"{city}_{year}_months_{month_labels}.zip"
    partial_path = archive_path.with_suffix(".partial.zip")

    if archive_path.exists() and valid_archive(archive_path):
        print(f"Reusing existing group archive: {archive_path.name}")
    else:
        if archive_path.exists():
            print(f"Existing group archive is invalid: {archive_path.name}")
            return 2

        request = {
            "product_type": ["reanalysis"],
            "variable": VARIABLES,
            "year": [str(year)],
            "month": [f"{month:02d}" for month in missing],
            "day": [f"{day:02d}" for day in range(1, days_in_month + 1)],
            "time": [f"{hour:02d}:00" for hour in range(24)],
            "area": [
                CITIES[city]["latitude"] + 0.25,
                CITIES[city]["longitude"] - 0.25,
                CITIES[city]["latitude"] - 0.25,
                CITIES[city]["longitude"] + 0.25,
            ],
            "data_format": "netcdf",
            "download_format": "zip",
        }

        try:
            print(f"Requesting {city} {year}, months {month_labels}...")
            print("This request may take a while. Please do not start another CDS request.")

            client = cdsapi.Client()
            client.retrieve(DATASET, request, str(partial_path))

            if not valid_archive(partial_path):
                raise RuntimeError(
                    "Downloaded archive is invalid or does not contain "
                    "the expected NetCDF files."
                )

            os.replace(partial_path, archive_path)
            print(f"Download complete: {archive_path}")
        finally:
            if partial_path.exists():
                partial_path.unlink()

    extract_dir = RAW_DIR / f"extracted_{city}_{year}_months_{month_labels}"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    combined_csv = extract_dir / f"{city}_{year}_{month_labels}_combined.csv"

    command = [
    sys.executable,
    str(PREPARE_SCRIPT),
    str(extract_dir),
    str(combined_csv),
    city,
]

    print("Preparing weather data...")
    subprocess.run(command, cwd=ROOT, check=True)

    if not combined_csv.exists():
        raise RuntimeError("Preparation script did not create the expected CSV.")

    df = pd.read_csv(combined_csv, parse_dates=["valid_time"])

    for month in missing:
        month_rows = df[
            (df["valid_time"].dt.year == year)
            & (df["valid_time"].dt.month == month)
        ].copy()

        target = MONTHLY_DIR / f"{city}_{year}_{month:02d}.csv"
        temporary = target.with_suffix(".partial.csv")

        if target.exists():
            try:
                validate_monthly_csv(target, year, month)
                print(f"SKIP: another valid CSV now exists: {target.name}")
                continue
            except Exception:
                pass

        if temporary.exists():
            temporary.unlink()

        month_rows.to_csv(temporary, index=False)

        try:
            validate_monthly_csv(temporary, year, month)
            os.replace(temporary, target)
            print(f"SUCCESS: {target.name} ({len(month_rows)} rows)")
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise

    print("Group processing finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
