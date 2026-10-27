
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from cities import CITIES


# Usage:
# python data_pipeline/prepare_era5_weather.py <input_folder> <output_file> <city>

if len(sys.argv) != 4:
    print(
        "Usage: python data_pipeline/prepare_era5_weather.py "
        "<input_folder> <output_file> <city>"
    )
    sys.exit(1)

INPUT_FOLDER = Path(sys.argv[1])
OUTPUT_FILE = Path(sys.argv[2])
CITY = sys.argv[3]

if CITY not in CITIES:
    raise ValueError(
        f"Unknown city: {CITY}. Available cities: {list(CITIES)}"
    )

target_lat = CITIES[CITY]["latitude"]
target_lon = CITIES[CITY]["longitude"]

instant_file = INPUT_FOLDER / "data_stream-oper_stepType-instant.nc"
accum_file = INPUT_FOLDER / "data_stream-oper_stepType-accum.nc"

print(f"Loading ERA5 data for {CITY}...")
print(f"Target coordinates: {target_lat}, {target_lon}")
print(f"Input folder: {INPUT_FOLDER}")

with xr.open_dataset(instant_file) as instant, xr.open_dataset(
    accum_file
) as accum:

    # Select the nearest ERA5 grid point to the requested city.
    # This prevents multiple spatial grid points from entering the time series.
    instant_point = instant.sel(
        latitude=target_lat,
        longitude=target_lon,
        method="nearest",
    )

    accum_point = accum.sel(
        latitude=target_lat,
        longitude=target_lon,
        method="nearest",
    )

    selected_lat = float(instant_point["latitude"].values)
    selected_lon = float(instant_point["longitude"].values)

    print(f"Selected ERA5 grid point: {selected_lat}, {selected_lon}")

    # Convert the selected grid point to hourly DataFrames.
    weather = instant_point[
        ["t2m", "d2m", "sp", "tcc", "tclw"]
    ].to_dataframe().reset_index()

    rain = accum_point[["tp"]].to_dataframe().reset_index()

# Ensure the precipitation data is aligned with the same grid point and time.
merge_keys = ["valid_time", "latitude", "longitude"]

weather = weather[
    merge_keys + ["t2m", "d2m", "sp", "tcc", "tclw"]
]

rain = rain[merge_keys + ["tp"]]

df = weather.merge(
    rain,
    on=merge_keys,
    how="inner",
    validate="one_to_one",
)

if df.empty:
    raise ValueError(
        f"No matching weather and precipitation records found for {CITY}."
    )

# Unit conversions.
df["temperature_C"] = df["t2m"] - 273.15
df["dewpoint_C"] = df["d2m"] - 273.15
df["pressure_hPa"] = df["sp"] / 100.0
df["cloud_cover_percent"] = df["tcc"] * 100.0
df["precipitation_mm"] = df["tp"] * 1000.0

# Calculate relative humidity from temperature and dew point.
es = 6.112 * np.exp(
    (17.67 * df["temperature_C"])
    / (df["temperature_C"] + 243.5)
)

e = 6.112 * np.exp(
    (17.67 * df["dewpoint_C"])
    / (df["dewpoint_C"] + 243.5)
)

df["relative_humidity_percent"] = (100.0 * e / es).clip(0, 100)

# Keep the existing downstream column names.
df = df[
    [
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
]

df = df.sort_values("valid_time").reset_index(drop=True)

# Validate the output before saving.
if df["valid_time"].duplicated().any():
    raise ValueError("Duplicate timestamps remain after grid-point selection.")

if df.isna().any().any():
    raise ValueError(
        f"Missing values found:\n{df.isna().sum()[df.isna().sum() > 0]}"
    )

if not np.isfinite(
    df.select_dtypes(include=[np.number]).to_numpy()
).all():
    raise ValueError("Non-finite numeric values found in the prepared dataset.")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nFirst 5 rows:")
print(df.head())

print(f"\nSaved to: {OUTPUT_FILE}")
