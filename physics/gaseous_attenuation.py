import pandas as pd
import numpy as np
import itur

INPUT_FILE = "data/raw/athens_era5_january.csv"
OUTPUT_FILE = "data/raw/athens_gas_january.csv"

FREQUENCY_GHZ = 39.0
ELEVATION_DEG = 30.0

print("Loading weather data...")

df = pd.read_csv(INPUT_FILE)

# Temperature: Celsius → Kelvin
temperature = df["temperature_C"] + 273.15

# Relative humidity: percentage → fraction
rh = df["relative_humidity_percent"] / 100.0

# Saturation vapour pressure in hPa
es = 6.112 * np.exp((17.67 * df["temperature_C"])/ (df["temperature_C"] + 243.5))

# Actual vapour pressure in hPa
e = rh * es

# Water vapour density in g/m³
rho = 216.7 * e / temperature

gas_loss = []

for i in range(len(df)):

    loss = itur.models.itu676.gaseous_attenuation_slant_path(FREQUENCY_GHZ,ELEVATION_DEG, rho.iloc[i], df["pressure_hPa"].iloc[i], temperature.iloc[i], mode="exact")

    gas_loss.append(float(loss.value))

df["gas_attenuation_dB"] = gas_loss

print("\nGas attenuation statistics:")
print(df["gas_attenuation_dB"].describe())

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved to: {OUTPUT_FILE}")