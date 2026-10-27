
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

FEATURE_FILE = DATA_DIR / "athens_2020_01_features.csv"
TARGET_FILE = DATA_DIR / "athens_2020_01_targets.csv"

SEQUENCE_LENGTH = 60
PREDICTION_HORIZON = 5

OUTPUT_X = DATA_DIR / "athens_2020_01_X_proxy.npy"
OUTPUT_Y = DATA_DIR / "athens_2020_01_y_proxy.npy"


print("Loading features and targets...")

features = pd.read_csv(FEATURE_FILE)
targets = pd.read_csv(TARGET_FILE)

# Align observations by timestamp before creating sequences.
physics_time = pd.to_datetime(targets["valid_time"])
feature_time = pd.to_datetime(pd.read_csv(DATA_DIR / "athens_2020_01_physics.csv")["valid_time"])

# The feature CSV has no timestamp column, so verify its row count
# against the timestamped physics dataset.
if len(features) != len(feature_time):
    raise ValueError("Feature and physics row counts do not match.")

if not physics_time.equals(feature_time):
    raise ValueError("Feature and target timestamps are not aligned.")

X_data = features.to_numpy(dtype=np.float32)
y_data = targets["total_attenuation_proxy_dB"].to_numpy(dtype=np.float32)

if not np.isfinite(X_data).all() or not np.isfinite(y_data).all():
    raise ValueError("Non-finite values found in features or targets.")

print("Feature data shape:", X_data.shape)
print("Target shape:", y_data.shape)

X = []
y = []

for i in range(
    len(X_data) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1
):
    X.append(X_data[i:i + SEQUENCE_LENGTH])
    y.append(
        y_data[
            i + SEQUENCE_LENGTH:
            i + SEQUENCE_LENGTH + PREDICTION_HORIZON
        ]
    )

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)

print("\nSequence shapes:")
print("X:", X.shape)
print("y:", y.shape)

np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)

print("\nSaved:")
print(OUTPUT_X)
print(OUTPUT_Y)
