import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_FILE = "data/processed/athens_features_january.csv"
PHYSICS_FILE = "data/raw/athens_physics_january.csv"

SEQUENCE_LENGTH = 60
PREDICTION_HORIZON = 5

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

print("Loading data...")

features = pd.read_csv(FEATURE_FILE)
physics = pd.read_csv(PHYSICS_FILE)

target = physics["total_attenuation_dB"].values
X_data = features.values

# Chronological split

n = len(X_data)

train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

X_train_raw = X_data[:train_end]
X_val_raw = X_data[train_end:val_end]
X_test_raw = X_data[val_end:]

y_train_raw = target[:train_end]
y_val_raw = target[train_end:val_end]
y_test_raw = target[val_end:]

print("\nRaw split:")
print("Train:", len(X_train_raw))
print("Validation:", len(X_val_raw))
print("Test:", len(X_test_raw))

# Standardization
# Fit ONLY on training data

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

joblib.dump(scaler,"training/feature_scaler.pkl")

# Create sequences

def create_sequences(X, y):
    X_sequences = []
    y_sequences = []

    for i in range(
        len(X) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1 ):
        X_sequences.append(X[i:i + SEQUENCE_LENGTH])
        y_sequences.append(
            y[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + PREDICTION_HORIZON])
    return np.array(X_sequences), np.array(y_sequences)

X_train, y_train = create_sequences(
    X_train_scaled,
    y_train_raw
)

X_val, y_val = create_sequences(
    X_val_scaled,
    y_val_raw
)

X_test, y_test = create_sequences(
    X_test_scaled,
    y_test_raw
)

# Print shapes

print("\nSequence shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Save

np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/y_train.npy", y_train)

np.save("data/processed/X_val.npy", X_val)
np.save("data/processed/y_val.npy", y_val)

np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_test.npy", y_test)

print("\nSaved processed datasets.")
print("Saved scaler: training/feature_scaler.pkl")