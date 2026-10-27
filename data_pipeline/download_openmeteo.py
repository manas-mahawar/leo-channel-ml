
import sys
from pathlib import Path

import pandas as pd
import requests

from cities import CITIES, UNSEEN_CITIES


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "processed" / "weather" / "open_meteo"

ALL_CITIES = {**CITIES, **UNSEEN_CITIES}

URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "rain",
    "cloud_cover",
]


def download_year(city, year, output_dir=OUTPUT_DIR):
    if city not in ALL_CITIES:
        raise ValueError(f"Unknown city: {city}")

    if not 2020 <= year <= 2024:
        raise ValueError("Year must be between 2020 and 2024.")

    output = Path(output_dir) / f"{city}_{year}.csv"

    if output.exists():
        existing = pd.read_csv(output)
        times = pd.to_datetime(existing["valid_time"], errors="coerce")
        expected = 8784 if pd.Timestamp(year, 12, 31).is_leap_year else 8760

        if (
            len(existing) == expected
            and times.notna().all()
            and times.is_unique
            and times.min() == pd.Timestamp(year, 1, 1)
            and times.max() == pd.Timestamp(year, 12, 31, 23)
            and "rain_mm" in existing.columns
        ):
            print(f"Already validated; skipping {output.name}")
            return

        # Keep the old file instead of silently replacing it.
        old_copy = output.with_name(f"{output.stem}.before_rebuild.csv")
        if old_copy.exists():
            raise FileExistsError(f"{old_copy.name} already exists; move it first.")
        output.replace(old_copy)
        print(f"Rebuilding weather file: {output.name} (old file kept as {old_copy.name})")

    coordinates = ALL_CITIES[city]

    params = {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": ",".join(HOURLY_VARIABLES),
        "models": "era5",
        "timezone": "UTC",
    }

    print(f"Requesting ERA5 data for {city}, {year}...")

    response = requests.get(URL, params=params, timeout=300)
    response.raise_for_status()
    payload = response.json()

    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise RuntimeError(f"No hourly data returned for {city}, {year}.")

    df = pd.DataFrame(hourly)
    df["valid_time"] = pd.to_datetime(df.pop("time"), errors="raise")

    expected_rows = 8784 if pd.Timestamp(year, 12, 31).is_leap_year else 8760
    expected_times = pd.date_range(
        f"{year}-01-01 00:00:00",
        f"{year}-12-31 23:00:00",
        freq="h",
    )

    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, received {len(df)}."
        )

    if not df["valid_time"].reset_index(drop=True).equals(
        pd.Series(expected_times, name="valid_time")
    ):
        raise ValueError("Hourly timestamps are incomplete or out of order.")

    if df[HOURLY_VARIABLES].isna().any().any():
        raise ValueError("Missing weather values were returned.")

    # Convert to the project's existing weather-column conventions.
    df = df.rename(
        columns={
            "temperature_2m": "temperature_C",
            "relative_humidity_2m": "relative_humidity_percent",
            "dew_point_2m": "dewpoint_C",
            "surface_pressure": "pressure_hPa",
            "precipitation": "precipitation_mm",
            "rain": "rain_mm",
            "cloud_cover": "cloud_cover_percent",
        }
    )

    # Open-Meteo pressure is in hPa for this requested variable.
    # Preserve the actual grid coordinates returned by the API.
    df["latitude"] = payload["latitude"]
    df["longitude"] = payload["longitude"]

    columns = [
        "valid_time",
        "latitude",
        "longitude",
        "temperature_C",
        "dewpoint_C",
        "relative_humidity_percent",
        "pressure_hPa",
        "precipitation_mm",
        "rain_mm",
        "cloud_cover_percent",
    ]
    df = df[columns]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.csv")
    df.to_csv(temporary, index=False)
    temporary.replace(output)

    print(f"Saved: {output}")
    print(f"Rows: {len(df)}")
    print(f"Grid point: {payload['latitude']}, {payload['longitude']}")
    print("Rain column included: rain_mm")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(
            "Usage: python data_pipeline/download_openmeteo.py "
            "<city> <year> [output_dir]"
        )
        raise SystemExit(2)

    if len(sys.argv) == 4:
        download_year(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    else:
        download_year(sys.argv[1], int(sys.argv[2]))
