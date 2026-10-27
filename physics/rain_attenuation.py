import pandas as pd
import numpy as np
import itur

INPUT_FILE = "data/processed/athens_2020_01_weather.csv"
OUTPUT_FILE = "data/processed/athens_2020_01_rain.csv"

FREQUENCY_GHZ = 39.0
ELEVATION_DEG = 30.0
POLARIZATION_DEG = 45.0

LATITUDE = 37.9842
LONGITUDE = 23.7275

print("Loading weather data...")

df = pd.read_csv(INPUT_FILE)

# Check that the required precipitation column exists
if "precipitation_mm" not in df.columns:
    raise ValueError("Missing precipitation_mm column in input data.")

# ERA5 precipitation is hourly accumulated precipitation in mm.
# Treating it as mm/h is an approximation for this implementation.
rain_rate = df["precipitation_mm"].fillna(0).clip(lower=0)

# Rain height from ITU-R P.839
rain_height = float(np.asarray(itur.models.itu839.rain_height(LATITUDE, LONGITUDE)).squeeze())

# Simple slant-path approximation at the selected elevation
slant_path = rain_height / np.sin(np.deg2rad(ELEVATION_DEG))

rain_loss = []

for rate in rain_rate:
    if rate <= 0:
        rain_loss.append(0.0)
        continue

    # Specific rain attenuation from ITU-R P.838
    gamma = itur.models.itu838.rain_specific_attenuation(float(rate),FREQUENCY_GHZ,ELEVATION_DEG,POLARIZATION_DEG)
    gamma = float(np.asarray(gamma).squeeze())

    # Approximate total rain attenuation
    attenuation = gamma * slant_path
    rain_loss.append(attenuation)

df["rain_height_km"] = rain_height
df["rain_slant_path_km"] = slant_path
df["rain_attenuation_dB"] = rain_loss

print("\nRain height:")
print(f"{rain_height:.3f} km")

print("\nRain slant path:")
print(f"{slant_path:.3f} km")

print("\nRain attenuation statistics:")
print(df["rain_attenuation_dB"].describe())
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to: {OUTPUT_FILE}")