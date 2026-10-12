import sys
import pandas as pd
import numpy as np
import xarray as xr

# Usage:
# python prepare_era5_weather.py <input_folder> <output_file>

if len(sys.argv) != 3:
    print("Usage:")
    print("python data_pipeline/prepare_era5_weather.py <input_folder> <output_file>")
    sys.exit(1)

INPUT_FOLDER = sys.argv[1]
OUTPUT_FILE = sys.argv[2]

instant_file = f"{INPUT_FOLDER}/data_stream-oper_stepType-instant.nc"
accum_file = f"{INPUT_FOLDER}/data_stream-oper_stepType-accum.nc"

print("Loading ERA5 data...")
print(f"Input folder: {INPUT_FOLDER}")

instant = xr.open_dataset(instant_file)
accum = xr.open_dataset(accum_file)

# Convert ERA5 grid point into DataFrames
weather = instant.to_dataframe().reset_index()
rain = accum.to_dataframe().reset_index()

# Keep required variables
weather = weather[["valid_time", "latitude", "longitude", "t2m", "d2m", "sp", "tcc", "tclw",]]

rain = rain[["valid_time","tp",]]

# Merge weather and precipitation
df = weather.merge(rain, on="valid_time")


# Unit conversions

df["temperature_C"] = df["t2m"] - 273.15
df["dewpoint_C"] = df["d2m"] - 273.15
df["pressure_hPa"] = df["sp"] / 100
df["cloud_cover_percent"] = df["tcc"] * 100
df["precipitation_mm"] = df["tp"] * 1000

# Relative humidity from temperature + dew point

es = 6.112 * np.exp((17.67 * df["temperature_C"])/ (df["temperature_C"] + 243.5))

e = 6.112 * np.exp((17.67 * df["dewpoint_C"])/ (df["dewpoint_C"] + 243.5))

df["relative_humidity_percent"] = 100 * e / es

df["relative_humidity_percent"] = (df["relative_humidity_percent"].clip(0, 100))

# Final columns

df = df[["valid_time","latitude","longitude","temperature_C","relative_humidity_percent","pressure_hPa","precipitation_mm","cloud_cover_percent","tclw",]]

df = df.sort_values("valid_time").reset_index(drop=True)

# Validation

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nFirst 5 rows:")
print(df.head())

# Save
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to: {OUTPUT_FILE}")

instant.close()
accum.close()