import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler

# CONFIGURATION

FEATURE_FILE = "data/processed/athens_features_january.csv"
PHYSICS_FILE = "data/raw/athens_physics_january.csv"

SEQ_LEN = 60
HORIZON = 5
# 1. LOAD DATA
features_df = pd.read_csv(FEATURE_FILE)
physics_df = pd.read_csv(PHYSICS_FILE)

print("Features shape:", features_df.shape)
print("Physics shape:", physics_df.shape)

# 2. CHECK THAT BOTH FILES HAVE SAME TIMELINE
if len(features_df) != len(physics_df):
    raise ValueError(
        f"Feature rows ({len(features_df)}) and "
        f"physics rows ({len(physics_df)}) do not match."
    )
print("Row alignment check: OK")

# 3. DEFINE 14 INPUT FEATURES

feature_cols = ["latitude","longitude","temperature_C","relative_humidity_percent","pressure_hPa","precipitation_mm","cloud_cover_percent",
    "gas_mean","gas_std","cloud_mean","cloud_std","rain_mean","rain_std","elevation_angle"]

# 4. EXTRACT INPUTS AND TARGETS
X = features_df[feature_cols].values
weather_cols = ["temperature_C","relative_humidity_percent","pressure_hPa","precipitation_mm","cloud_cover_percent"]

weather = features_df[weather_cols].values

# EPL proxy from physics pipeline
epl = physics_df["total_attenuation_dB"].values

print("\nRaw data:")
print("X:", X.shape)
print("Weather:", weather.shape)
print("EPL:", epl.shape)

# 5. CHRONOLOGICAL 70 / 15 / 15 SPLIT

n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
print("\nSplit:")
print("Train:", 0, "to", train_end)
print("Validation:", train_end, "to", val_end)
print("Test:", val_end, "to", n)


X_train_raw = X[:train_end]
X_val_raw = X[train_end:val_end]
X_test_raw = X[val_end:]

weather_train_raw = weather[:train_end]
weather_val_raw = weather[train_end:val_end]
weather_test_raw = weather[val_end:]

epl_train_raw = epl[:train_end]
epl_val_raw = epl[train_end:val_end]
epl_test_raw = epl[val_end:]

# 6. STANDARDIZE INPUT FEATURES
#    FIT ONLY ON TRAINING DATA

feature_scaler = StandardScaler()

X_train_scaled = feature_scaler.fit_transform(X_train_raw)

X_val_scaled = feature_scaler.transform(X_val_raw)

X_test_scaled = feature_scaler.transform(X_test_raw)

joblib.dump(feature_scaler,"training/feature_scaler_multitask.pkl")

# 7. STANDARDIZE WEATHER TARGETS
#    FIT ONLY ON TRAINING DATA

weather_scaler = StandardScaler()

weather_train_scaled = weather_scaler.fit_transform(weather_train_raw)

weather_val_scaled = weather_scaler.transform(weather_val_raw)

weather_test_scaled = weather_scaler.transform(weather_test_raw)

joblib.dump(weather_scaler,"training/weather_target_scaler.pkl")

# 8. CREATE SEQUENCES
def make_sequences(features, weather, epl):
    X_seq = []
    weather_seq = []
    epl_seq = []
    for i in range(
        len(features) - SEQ_LEN - HORIZON + 1
    ):
        # 60 historical time steps
        X_seq.append(features[ i : i + SEQ_LEN]
        )

        # Next 5 hours of weather
        weather_seq.append(weather[i + SEQ_LEN :i + SEQ_LEN + HORIZON])

        # Next 5 hours of EPL
        epl_seq.append(
            epl[
                i + SEQ_LEN :
                i + SEQ_LEN + HORIZON
            ]
        )

    return (
        np.array(X_seq),
        np.array(weather_seq),
        np.array(epl_seq)
    )

# 9. CREATE TRAIN / VALIDATION / TEST SEQUENCES
X_train, weather_train, epl_train = make_sequences(
    X_train_scaled,
    weather_train_scaled,
    epl_train_raw
)

X_val, weather_val, epl_val = make_sequences(
    X_val_scaled,
    weather_val_scaled,
    epl_val_raw
)

X_test, weather_test, epl_test = make_sequences(
    X_test_scaled,
    weather_test_scaled,
    epl_test_raw
)

# 10. SAVE DATA
np.save(
    "training/X_train_multitask.npy",
    X_train
)

np.save(
    "training/X_val_multitask.npy",
    X_val
)

np.save(
    "training/X_test_multitask.npy",
    X_test
)

np.save(
    "training/weather_train.npy",
    weather_train
)

np.save(
    "training/weather_val.npy",
    weather_val
)

np.save(
    "training/weather_test.npy",
    weather_test
)

np.save(
    "training/epl_train.npy",
    epl_train
)

np.save(
    "training/epl_val.npy",
    epl_val
)

np.save(
    "training/epl_test.npy",
    epl_test
)

# 11. PRINT FINAL SHAPES

print("\nFinal sequence shapes:")

print("X_train:", X_train.shape)
print("Weather train:", weather_train.shape)
print("EPL train:", epl_train.shape)

print("\nX_val:", X_val.shape)
print("Weather val:", weather_val.shape)
print("EPL val:", epl_val.shape)

print("\nX_test:", X_test.shape)
print("Weather test:", weather_test.shape)
print("EPL test:", epl_test.shape)

# 12. PRINT WEATHER SCALER
print("\nWeather target means:")
print(weather_scaler.mean_)

print("\nWeather target standard deviations:")
print(weather_scaler.scale_)

print("\nSaved:")
print("training/feature_scaler_multitask.pkl")
print("training/weather_target_scaler.pkl")