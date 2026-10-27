
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

FEATURE_FILE = "data/processed/athens_2020_01_features.csv"
TARGET_FILE = "data/processed/athens_2020_01_targets.csv"

SEQ_LEN = 60
HORIZON = 5

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Separate outputs: do not overwrite older experiments.
OUTPUT_PREFIX = "data/processed/athens_proxy"

# --------------------------------------------------
# 1. LOAD AND VALIDATE DATA
# --------------------------------------------------

features_df = pd.read_csv(FEATURE_FILE)
targets_df = pd.read_csv(TARGET_FILE, parse_dates=["valid_time"])

feature_cols = [
    "latitude",
    "longitude",
    "temperature_C",
    "relative_humidity_percent",
    "pressure_hPa",
    "precipitation_mm",
    "cloud_cover_percent",
    "gas_mean",
    "gas_std",
    "cloud_mean",
    "cloud_std",
    "rain_mean",
    "rain_std",
    "elevation_angle",
]

target_col = "total_attenuation_proxy_dB"

if len(features_df) != len(targets_df):
    raise ValueError("Feature and target row counts do not match.")

if not targets_df["valid_time"].is_monotonic_increasing:
    raise ValueError("Target timestamps are not chronological.")

if targets_df["valid_time"].duplicated().any():
    raise ValueError("Duplicate target timestamps found.")

if features_df[feature_cols].isna().any().any():
    raise ValueError("Missing values found in input features.")

if targets_df[target_col].isna().any():
    raise ValueError("Missing attenuation targets found.")

X = features_df[feature_cols].to_numpy(dtype=np.float64)
y = targets_df[target_col].to_numpy(dtype=np.float64)

n = len(X)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

print("Rows:", n)
print("Features:", X.shape)
print("Target:", y.shape)
print("Train boundary:", train_end)
print("Validation boundary:", val_end)

# --------------------------------------------------
# 2. FIT SCALER ON TRAINING PERIOD ONLY
# --------------------------------------------------

scaler = StandardScaler()
scaler.fit(X[:train_end])

X_scaled = scaler.transform(X)

joblib.dump(scaler, f"{OUTPUT_PREFIX}_feature_scaler.pkl")

# --------------------------------------------------
# 3. CREATE CHRONOLOGICAL SEQUENCES
# --------------------------------------------------

def create_sequences(start_target, end_target):
    """
    Target indices belong to [start_target, end_target).

    Historical input may come from earlier periods.
    Every future target must remain inside this split.
    """
    X_seq = []
    y_seq = []
    target_times = []

    first_target = max(start_target, SEQ_LEN)

    for target_start in range(
        first_target,
        end_target - HORIZON + 1,
    ):
        X_seq.append(
            X_scaled[target_start - SEQ_LEN:target_start]
        )

        y_seq.append(
            y[target_start:target_start + HORIZON]
        )

        target_times.append(
            targets_df["valid_time"].iloc[target_start]
        )

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(target_times, dtype="datetime64[ns]"),
    )


X_train, y_train, t_train = create_sequences(0, train_end)
X_val, y_val, t_val = create_sequences(train_end, val_end)
X_test, y_test, t_test = create_sequences(val_end, n)

# --------------------------------------------------
# 4. SAVE SEPARATE EXPERIMENT ARRAYS
# --------------------------------------------------

for name, array in [
    ("X_train", X_train),
    ("y_train", y_train),
    ("X_val", X_val),
    ("y_val", y_val),
    ("X_test", X_test),
    ("y_test", y_test),
    ("t_train", t_train),
    ("t_val", t_val),
    ("t_test", t_test),
]:
    np.save(f"{OUTPUT_PREFIX}_{name}.npy", array)

print("\nSequence shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape, "y_test: ", y_test.shape)

print("\nTarget periods:")
for name, times in [
    ("Train", t_train),
    ("Validation", t_val),
    ("Test", t_test),
]:
    if len(times):
        print(f"{name}: {times[0]} to {times[-1]}")

print("\nSaved arrays with prefix:", OUTPUT_PREFIX)
print("Target: simplified attenuation proxy, not verified EPL.")
