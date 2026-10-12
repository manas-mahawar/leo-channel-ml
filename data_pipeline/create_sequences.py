import numpy as np
import pandas as pd

INPUT_FILE = "data/processed/athens_features_january.csv"
PHYSICS_FILE = "data/raw/athens_physics_january.csv"

SEQUENCE_LENGTH = 60
PREDICTION_HORIZON = 5

OUTPUT_X = "data/processed/athens_X_january.npy"
OUTPUT_Y = "data/processed/athens_y_january.npy"

print("Loading data...")

features = pd.read_csv(INPUT_FILE)
physics = pd.read_csv(PHYSICS_FILE)

# Make sure both datasets have the same chronological order
physics["valid_time"] = pd.to_datetime(physics["valid_time"])

physics = physics.sort_values("valid_time").reset_index(drop=True)
features = features.reset_index(drop=True)

X_data = features.values
y_data = physics["total_attenuation_dB"].values

print("Feature data shape:", X_data.shape)
print("Target shape:", y_data.shape)

# Create sliding-window sequences
# X = previous 60 observations
# y = next 5 observations

X = []
y = []

for i in range(len(X_data) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1):
    X.append(X_data[i:i + SEQUENCE_LENGTH])
    y.append(y_data[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + PREDICTION_HORIZON])

X = np.array(X)
y = np.array(y)

print("\nSequence shapes:")
print("X:", X.shape)
print("y:", y.shape)

np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)

print("\nSaved:")
print(OUTPUT_X)
print(OUTPUT_Y)