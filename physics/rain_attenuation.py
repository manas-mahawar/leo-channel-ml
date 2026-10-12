import pandas as pd
import numpy as np
import itur

INPUT_FILE = "data/raw/athens_era5_january.csv"
OUTPUT_FILE = "data/raw/athens_rain_january.csv"

FREQUENCY_GHZ = 39.0
ELEVATION_DEG = 30.0

LATITUDE = 37.9842
LONGITUDE = 23.7275

print("Loading weather data...")

df = pd.read_csv(INPUT_FILE)

# Rain height from ITU-R P.839
rain_height = float(np.asarray(itur.models.itu839.rain_height(LATITUDE,LONGITUDE)).squeeze())

rain_loss = []

for rain_rate in df["precipitation_mm"]:

    if rain_rate <= 0:
        rain_loss.append(0.0)
        continue

    # Specific rain attenuation from ITU-R P.838
    gamma = itur.models.itu838.rain_specific_attenuation(rain_rate,FREQUENCY_GHZ,ELEVATION_DEG,45.0)

    gamma = float(np.asarray(gamma).squeeze())

    # Simple slant path approximation
    slant_path = rain_height / np.sin(np.deg2rad(ELEVATION_DEG))

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