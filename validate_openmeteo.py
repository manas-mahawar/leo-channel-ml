import pandas as pd

from config import WEATHER_DIR

# The training cities are read from WEATHER_DIR (open_meteo_v2).
folder = WEATHER_DIR
cities = ["Athens", "Bergen", "Budapest", "Chilbolton", "Vigo", "Warsaw"]
years = range(2020, 2025)

required = [
    "valid_time", "latitude", "longitude", "temperature_C",
    "dewpoint_C", "relative_humidity_percent", "pressure_hPa",
    "precipitation_mm", "cloud_cover_percent"
]

issues = []
checked = 0

for city in cities:
    for year in years:
        file = folder / f"{city}_{year}.csv"
        expected = 8784 if year in (2020, 2024) else 8760

        if not file.exists():
            issues.append(f"MISSING FILE: {file.name}")
            continue

        try:
            df = pd.read_csv(file)
            checked += 1

            missing_columns = set(required) - set(df.columns)
            if missing_columns:
                issues.append(f"{file.name}: missing columns {missing_columns}")
                continue

            timestamps = pd.to_datetime(df["valid_time"], errors="coerce", utc=True)

            if len(df) != expected:
                issues.append(f"{file.name}: {len(df)} rows, expected {expected}")

            if timestamps.isna().any() or timestamps.nunique() != expected:
                issues.append(f"{file.name}: invalid or duplicate timestamps")

            if df[required[1:]].isna().any().any():
                issues.append(f"{file.name}: missing numeric values")

        except Exception as error:
            issues.append(f"{file.name}: {error}")

print(f"\nFiles checked: {checked}/30")
print("RESULT:", "PASSED" if not issues else "ISSUES FOUND")

for issue in issues:
    print("-", issue)

if not issues:
    print("All 30 files passed the basic validation checks.")
