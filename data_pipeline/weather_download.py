import requests
import pandas as pd
from pathlib import Path

# ATHENS CONFIGURATION
LATITUDE = 37.9842
LONGITUDE = 23.7275

START_DATE = "2020-01-01"
END_DATE = "2020-01-31"

# OPEN-METEO HISTORICAL WEATHER API
URL = "https://archive-api.open-meteo.com/v1/archive"

PARAMS = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "precipitation",
        "cloud_cover",
    ],
    "timezone": "UTC",
}

# DOWNLOAD
print("Downloading Athens weather data...")
response = requests.get(URL, params=PARAMS, timeout=60)
response.raise_for_status()
data = response.json()


# CONVERT TO DATAFRAME
df = pd.DataFrame(data["hourly"])

print("\nDownload successful.")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum())

# SAVE
output_path = Path("data/raw/athens_weather_test.csv")

output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print(f"\nSaved to: {output_path}")