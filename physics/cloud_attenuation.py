import pandas as pd
import numpy as np
import itur

INPUT_FILE = "data/raw/athens_era5_january.csv"
OUTPUT_FILE = "data/raw/athens_cloud_january.csv"

FREQUENCY_GHZ = 39.0
ELEVATION_DEG = 30.0

df = pd.read_csv(INPUT_FILE)

cloud_loss = []

for i in range(len(df)):
    # ERA5 total column liquid water
    lwp = df["tclw"].iloc[i]

    # Temperature in Celsius for ITU-R P.840
    temperature = df["temperature_C"].iloc[i]

    # ITU-R P.840 specific attenuation coefficient
    kl = itur.models.itu840.specific_attenuation_coefficients(FREQUENCY_GHZ,temperature)

    kl = float(np.asarray(kl).squeeze())

    # Cloud attenuation
    if lwp <= 0:
        attenuation = 0.0
    else:
        attenuation = (lwp * kl / np.sin(np.deg2rad(ELEVATION_DEG)))

    cloud_loss.append(attenuation)

df["cloud_attenuation_dB"] = cloud_loss

print(df["cloud_attenuation_dB"].describe())

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved to: {OUTPUT_FILE}")