from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")

WEATHER_FILE = DATA_DIR / "athens_2020_01_weather.csv"
GAS_FILE = DATA_DIR / "athens_2020_01_gas.csv"
CLOUD_FILE = DATA_DIR / "athens_2020_01_cloud.csv"
RAIN_FILE = DATA_DIR / "athens_2020_01_rain.csv"

OUTPUT_FILE = DATA_DIR / "athens_2020_01_physics.csv"


def load_csv(path):
    df = pd.read_csv(path, parse_dates=["valid_time"])

    if df["valid_time"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {path.name}")

    return df


print("Loading weather and attenuation data...")

weather = load_csv(WEATHER_FILE)
gas = load_csv(GAS_FILE)
cloud = load_csv(CLOUD_FILE)
rain = load_csv(RAIN_FILE)

# Keep only the new attenuation column from each physics file.
gas = gas[["valid_time", "gas_attenuation_dB"]]
cloud = cloud[["valid_time", "cloud_attenuation_dB"]]
rain = rain[["valid_time", "rain_attenuation_dB"]]

# Merge by timestamp, requiring every source to have a matching hour.
merged = weather.merge(gas, on="valid_time", how="outer", validate="one_to_one",indicator="_gas")

if not (merged["_gas"] == "both").all():
    raise ValueError("Weather and gas timestamps do not match.")

merged = merged.drop(columns="_gas")

merged = merged.merge(cloud, on="valid_time", how="outer", validate="one_to_one",indicator="_cloud")

if not (merged["_cloud"] == "both").all():
    raise ValueError("Weather and cloud timestamps do not match.")

merged = merged.drop(columns="_cloud")

merged = merged.merge(rain, on="valid_time", how="outer", validate="one_to_one",indicator="_rain")

if not (merged["_rain"] == "both").all():
    raise ValueError("Weather and rain timestamps do not match.")

merged = merged.drop(columns="_rain")
merged = merged.sort_values("valid_time").reset_index(drop=True)

# Final validation
if len(merged) != 744:
    raise ValueError(f"Expected 744 hourly records, found {len(merged)}.")

if merged.isna().any().any():
    raise ValueError("Missing values found in the merged dataset.")

merged.to_csv(OUTPUT_FILE, index=False)

print("\nMerged dataset created successfully.")
print(f"Rows: {len(merged)}")
print(f"Columns: {len(merged.columns)}")
print(f"Date range: {merged['valid_time'].min()} to "
      f"{merged['valid_time'].max()}")
print(f"Saved to: {OUTPUT_FILE}")
print("\nColumns:")
print(merged.columns.tolist())
