import pandas as pd

INPUT_FILE = "data/raw/athens_physics_january.csv"
OUTPUT_FILE = "data/processed/athens_features_january.csv"

ELEVATION_DEG = 30.0
ROLLING_WINDOW = 60

print("Loading physics data...")

df = pd.read_csv(INPUT_FILE)

df["valid_time"] = pd.to_datetime(df["valid_time"])

# Sort chronologically
df = df.sort_values("valid_time").reset_index(drop=True)

# [OUR IMPLEMENTATION]
# 60-hour rolling statistics for attenuation features.
# The paper does not disclose the exact rolling-window
# construction for these mean/std features.

df["gas_mean"] = (df["gas_attenuation_dB"].rolling(ROLLING_WINDOW, min_periods=1).mean())

df["gas_std"] = (df["gas_attenuation_dB"].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0))

df["cloud_mean"] = (df["cloud_attenuation_dB"].rolling(ROLLING_WINDOW, min_periods=1).mean())

df["cloud_std"] = (df["cloud_attenuation_dB"].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0))

df["rain_mean"] = (df["rain_attenuation_dB"].rolling(ROLLING_WINDOW, min_periods=1).mean())

df["rain_std"] = (df["rain_attenuation_dB"].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0))

# Elevation angle
# [OUR IMPLEMENTATION]
# The paper does not disclose the numerical satellite geometry.
df["elevation_angle"] = ELEVATION_DEG

# Select the 14 features described in the paper

features = df[["latitude","longitude","temperature_C","relative_humidity_percent","pressure_hPa","precipitation_mm","cloud_cover_percent",
        "gas_mean","gas_std","cloud_mean","cloud_std","rain_mean","rain_std","elevation_angle",]].copy()

print("\nFeature matrix:")
print(features.head())

print("\nShape:")
print(features.shape)

print("\nMissing values:")
print(features.isna().sum())

features.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved to: {OUTPUT_FILE}")