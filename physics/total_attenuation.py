import pandas as pd

WEATHER_FILE = "data/raw/athens_era5_january.csv"
GAS_FILE = "data/raw/athens_gas_january.csv"
CLOUD_FILE = "data/raw/athens_cloud_january.csv"
RAIN_FILE = "data/raw/athens_rain_january.csv"

OUTPUT_FILE = "data/raw/athens_physics_january.csv"

weather = pd.read_csv(WEATHER_FILE)
gas = pd.read_csv(GAS_FILE)
cloud = pd.read_csv(CLOUD_FILE)
rain = pd.read_csv(RAIN_FILE)

df = weather[["valid_time","latitude","longitude","temperature_C","relative_humidity_percent","pressure_hPa","precipitation_mm","cloud_cover_percent"]].copy()

df = df.merge(gas[["valid_time", "gas_attenuation_dB"]],on="valid_time",how="left")

df = df.merge(cloud[["valid_time", "cloud_attenuation_dB"]],on="valid_time",how="left")

df = df.merge(rain[["valid_time", "rain_attenuation_dB"]],on="valid_time",how="left")

df["total_attenuation_dB"] = (df["gas_attenuation_dB"] + df["cloud_attenuation_dB"] + df["rain_attenuation_dB"])

print(df[["gas_attenuation_dB", "cloud_attenuation_dB", "rain_attenuation_dB", "total_attenuation_dB"]].describe())

print("\nMissing values:")
print(df.isna().sum())

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved to: {OUTPUT_FILE}")