"""Download hourly mean-sea-level pressure (ERA5 via Open-Meteo).

The paper's Table 2 defines the pressure input as "atmospheric pressure at
mean sea level", while Sect. 3.1 says "surface pressure". The main pipeline
uses surface pressure; this file adds MSL pressure for the separate
`multitask_msl` experiment. Surface pressure is downloaded again only to
check that the hours line up with the existing files.

Usage:
    python data_pipeline/download_pressure_msl.py
Output: data/processed/weather/open_meteo_msl/<City>_<year>.csv
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from data_pipeline.cities import CITIES, UNSEEN_CITIES

URL = "https://archive-api.open-meteo.com/v1/archive"
OUTPUT_DIR = ROOT / "data" / "processed" / "weather" / "open_meteo_msl"


def download(city, coords, year):
    output = OUTPUT_DIR / f"{city}_{year}.csv"
    if output.exists():
        print(f"Exists, skipping {output.name}")
        return
    params = {
        "latitude": coords["latitude"], "longitude": coords["longitude"],
        "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
        "hourly": "pressure_msl,surface_pressure", "models": "era5", "timezone": "UTC",
    }
    response = requests.get(URL, params=params, timeout=300)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    df = pd.DataFrame({
        "valid_time": pd.to_datetime(hourly["time"]),
        "pressure_msl_hPa": hourly["pressure_msl"],
        "surface_pressure_hPa": hourly["surface_pressure"],
    })
    expected = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    if not df["valid_time"].reset_index(drop=True).equals(pd.Series(expected, name="valid_time")):
        raise ValueError(f"{city} {year}: incomplete hourly series.")
    if df.isna().any().any():
        raise ValueError(f"{city} {year}: missing values.")
    df.to_csv(output, index=False)
    print(f"Saved {output.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(c, v, y) for c, v in CITIES.items() for y in config.YEARS]
    jobs += [(c, v, y) for c, v in UNSEEN_CITIES.items() for y in config.UNSEEN_YEARS]
    for city, coords, year in jobs:
        download(city, coords, year)
        time.sleep(2)


if __name__ == "__main__":
    main()
