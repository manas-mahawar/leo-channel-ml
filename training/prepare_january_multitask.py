
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

# Files
WEATHER_FILE = "data/processed/athens_2020_01_weather.csv"
PREFIX = "data/processed/athens_proxy"

SEQ_LEN = 60
HORIZON = 5

# Load data
weather_df = pd.read_csv(
    WEATHER_FILE,
    parse_dates=["valid_time"],
)

weather_cols = [
    "temperature_C",
    "relative_humidity_percent",
    "pressure_hPa",
    "precipitation_mm",
    "cloud_cover_percent",
]

if weather_df["valid_time"].duplicated().any():
    raise ValueError("Duplicate weather timestamps found.")

if not weather_df["valid_time"].is_monotonic_increasing:
    raise ValueError("Weather timestamps are not chronological.")

weather = weather_df[weather_cols].to_numpy(dtype=np.float64)

# Load timestamps for the existing feature/target split
target_df = pd.read_csv(
    "data/processed/athens_2020_01_targets.csv",
    parse_dates=["valid_time"],
)

if not np.array_equal(
    weather_df["valid_time"].to_numpy(),
    target_df["valid_time"].to_numpy(),
):
    raise ValueError("Weather and target timestamps do not match.")

n = len(weather)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Scale weather targets using training-period statistics only
weather_scaler = StandardScaler()
weather_scaler.fit(weather[:train_end])
weather_scaled = weather_scaler.transform(weather)

joblib.dump(
    weather_scaler,
    f"{PREFIX}_weather_scaler.pkl",
)

# Use the same forecast start times as the prepared proxy arrays
def make_weather_targets(start, end):
    targets = []

    first_target = max(start, SEQ_LEN)

    for i in range(first_target, end - HORIZON + 1):
        targets.append(weather_scaled[i:i + HORIZON])

    return np.asarray(targets, dtype=np.float32)

weather_train = make_weather_targets(0, train_end)
weather_val = make_weather_targets(train_end, val_end)
weather_test = make_weather_targets(val_end, n)

# Save separately from the older experiment
np.save(f"{PREFIX}_weather_train.npy", weather_train)
np.save(f"{PREFIX}_weather_val.npy", weather_val)
np.save(f"{PREFIX}_weather_test.npy", weather_test)

print("Weather target shapes:")
print("Train:", weather_train.shape)
print("Validation:", weather_val.shape)
print("Test:", weather_test.shape)

print("\nSaved weather scaler and multitask targets.")
