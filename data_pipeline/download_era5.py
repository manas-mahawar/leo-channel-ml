
import os
import sys
import calendar
import zipfile
import cdsapi
from pathlib import Path

from cities import CITIES


# NetCDF files expected inside each ERA5 archive.
EXPECTED_FILES = [
    "data_stream-oper_stepType-instant.nc",
    "data_stream-oper_stepType-accum.nc",
]

# NetCDF4 files use the HDF5 signature; classic NetCDF starts with CDF.
NETCDF_SIGNATURES = (b"\x89HDF\r\n\x1a\n", b"CDF")


def is_valid_archive(path):
    """Return True if the ZIP is complete and contains both expected NetCDF files."""
    if not zipfile.is_zipfile(path):
        return False

    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                return False

            names = archive.namelist()

            for name in EXPECTED_FILES:
                if name not in names:
                    return False

                with archive.open(name) as file:
                    if not file.read(8).startswith(NETCDF_SIGNATURES):
                        return False

    except (OSError, zipfile.BadZipFile, RuntimeError):
        return False

    return True


def download_era5(city, year, month):
    """Download and validate one month of ERA5 single-level data."""

    if city not in CITIES:
        raise ValueError(
            f"Unknown city: {city}. Available cities: {list(CITIES)}"
        )

    year_int = int(year)
    month_int = int(month)

    if not 1 <= month_int <= 12:
        raise ValueError("Month must be between 1 and 12.")

    year = str(year_int)
    month = f"{month_int:02d}"

    latitude = CITIES[city]["latitude"]
    longitude = CITIES[city]["longitude"]

    output_dir = Path("data/raw/era5")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{city}_{year}_{month}.zip"
    partial_file = output_dir / f"{city}_{year}_{month}.partial.zip"

    # Reuse only archives that pass validation.
    if output_file.exists():
        if is_valid_archive(output_file):
            print(f"Valid archive already exists; skipping: {output_file}")
            return

        print(f"Existing archive is invalid; downloading again: {output_file}")

    times = [f"{hour:02d}:00" for hour in range(24)]
    days = [
        f"{day:02d}"
        for day in range(1, calendar.monthrange(year_int, month_int)[1] + 1)
    ]

    # Request a wider area because a small box may contain no ERA5 grid point.
    # The preparation step must select the grid point nearest to the city.
    area = [
        latitude + 0.25,   # North
        longitude - 0.25,  # West
        latitude - 0.25,   # South
        longitude + 0.25,  # East
    ]

    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "2m_temperature",
            "2m_dewpoint_temperature",
            "surface_pressure",
            "total_precipitation",
            "total_cloud_cover",
            "total_column_cloud_liquid_water",
        ],
        "year": [year],
        "month": [month],
        "day": days,
        "time": times,
        "area": area,
        "data_format": "netcdf",
        "download_format": "zip",
    }

    print(f"\nDownloading ERA5: {city}, {year}-{month}")
    print(f"City coordinates: {latitude}, {longitude}")
    print(f"Requested area: {area}")
    print(f"Output: {output_file}")

    try:
        client = cdsapi.Client()

        client.retrieve(
            "reanalysis-era5-single-levels",
            request,
            str(partial_file),
        )

        if not is_valid_archive(partial_file):
            raise RuntimeError(
                f"Downloaded archive is incomplete or invalid: {partial_file}"
            )

        # Replace the final file only after successful validation.
        os.replace(partial_file, output_file)

    finally:
        # Remove incomplete downloads after failure.
        partial_file.unlink(missing_ok=True)

    print(f"Download complete: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python data_pipeline/download_era5.py "
            "<city> <year> <month>"
        )
        print(f"Cities: {', '.join(CITIES)}")
        sys.exit(1)

    download_era5(sys.argv[1], sys.argv[2], sys.argv[3])
