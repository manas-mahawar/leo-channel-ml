import sys
import calendar
import cdsapi
from pathlib import Path

from cities import CITIES

def download_era5(city, year, month):
    if city not in CITIES:
        raise ValueError(
            f"Unknown city: {city}. Available cities: {list(CITIES)}"
        )

    year = str(year)
    month = f"{int(month):02d}"

    if not 1 <= int(month) <= 12:
        raise ValueError("Month must be between 1 and 12.")

    latitude = CITIES[city]["latitude"]
    longitude = CITIES[city]["longitude"]

    output_dir = Path("data/raw/era5")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{city}_{year}_{month}.zip"

    # Skip downloads that already exist and are non-empty.
    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"File already exists; skipping: {output_file}")
        return

    times = [f"{hour:02d}:00" for hour in range(24)]
    days = [f"{day:02d}" for day in range(1, calendar.monthrange(int(year), int(month))[1] + 1)]

    request = {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature","2m_dewpoint_temperature","surface_pressure","total_precipitation","total_cloud_cover","total_column_cloud_liquid_water",],
        "year": [year],"month": [month],"day": days,"time": times,"area": [
            latitude + 0.1,
            longitude - 0.1,
            latitude - 0.1,
            longitude + 0.1,
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    print(f"\nDownloading ERA5: {city}, {year}-{month}")
    print(f"Coordinates: {latitude}, {longitude}")
    print(f"Output: {output_file}")

    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        request,
        str(output_file),
    )

    if not output_file.exists() or output_file.stat().st_size == 0:
        raise RuntimeError("Download failed or produced an empty file.")

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
