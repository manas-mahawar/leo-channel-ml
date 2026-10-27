import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Reshape
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. REPRODUCIBILITY
np.random.seed(42)
tf.random.set_seed(42)

PREFIX = "data/processed/athens_proxy"

# 2. LOAD THE NEW JANUARY EXPERIMENT
X_train = np.load(f"{PREFIX}_X_train.npy")
X_val = np.load(f"{PREFIX}_X_val.npy")
X_test = np.load(f"{PREFIX}_X_test.npy")

weather_train = np.load(f"{PREFIX}_weather_train.npy")
weather_val = np.load(f"{PREFIX}_weather_val.npy")
weather_test = np.load(f"{PREFIX}_weather_test.npy")

atten_train = np.load(f"{PREFIX}_y_train.npy")
atten_val = np.load(f"{PREFIX}_y_val.npy")
atten_test = np.load(f"{PREFIX}_y_test.npy")

print("Input shapes:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# 3. BUILD THE THREE-LAYER LSTM
inputs = Input(shape=(60, 14), name="historical_features")

x = LSTM(256, return_sequences=True)(inputs)
x = Dropout(0.2)(x)

x = LSTM(256, return_sequences=True)(x)
x = Dropout(0.2)(x)

x = LSTM(256)(x)

# Weather forecast: 5 future hours x 5 variables
weather_output = Dense(25)(x)
weather_output = Reshape((5, 5), name="weather")(weather_output)

# Attenuation forecast: 5 means + 5 log variances
atten_output = Dense(10, name="attenuation")(x)

model = Model(
    inputs=inputs,
    outputs={
        "weather": weather_output,
        "attenuation": atten_output,
    },
)

# 4. GAUSSIAN NEGATIVE LOG-LIKELIHOOD
def gaussian_nll(y_true, y_pred):
    mean = y_pred[:, :5]
    log_variance = tf.clip_by_value(
        y_pred[:, 5:], -10.0, 10.0
    )

    variance = tf.exp(log_variance) + 1e-6

    loss = 0.5 * (
        log_variance
        + tf.square(y_true - mean) / variance
    )

    return tf.reduce_mean(loss)

# 5. COMPILE
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        "weather": "mse",
        "attenuation": gaussian_nll,
    },
    loss_weights={
        "weather": 1.0,
        "attenuation": 1.0,
    },
)

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True,
)

# 6. TRAIN
history = model.fit(
    X_train,
    {
        "weather": weather_train,
        "attenuation": atten_train,
    },
    validation_data=(
        X_val,
        {
            "weather": weather_val,
            "attenuation": atten_val,
        },
    ),
    epochs=50,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=1,
)

# 7. EVALUATE ATTENUATION FORECAST
predictions = model.predict(X_test, verbose=0)
atten_output_test = predictions["attenuation"]

predicted_mean = atten_output_test[:, :5]
predicted_log_variance = np.clip(
    atten_output_test[:, 5:], -10.0, 10.0
)

mae = mean_absolute_error(
    atten_test.ravel(),
    predicted_mean.ravel(),
)

rmse = np.sqrt(
    mean_squared_error(
        atten_test.ravel(),
        predicted_mean.ravel(),
    )
)

variance = np.exp(predicted_log_variance)
std = np.sqrt(variance)

lower_95 = predicted_mean - 1.96 * std
upper_95 = predicted_mean + 1.96 * std

picp = np.mean(
    (atten_test >= lower_95)
    & (atten_test <= upper_95)
)

sharpness = np.mean(upper_95 - lower_95)

nll = np.mean(
    0.5 * (
        predicted_log_variance
        + np.square(atten_test - predicted_mean) / variance
    )
)

print("\nAttenuation-proxy test results")
print(f"MAE:       {mae:.6f} dB")
print(f"RMSE:      {rmse:.6f} dB")
print(f"PICP 95%:  {picp * 100:.2f}%")
print(f"Sharpness: {sharpness:.6f} dB")
print(f"Gaussian NLL: {nll:.6f}")

# 8. SAVE MODEL AND TRAINING HISTORY
model.save("models/multitask_january_proxy.keras")

with open(
    "models/multitask_january_proxy_history.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(history.history, file, indent=2)

np.save(
    f"{PREFIX}_attenuation_predictions.npy",
    predicted_mean,
)

np.save(
    f"{PREFIX}_attenuation_prediction_std.npy",
    std,
)

print("\nSaved model: models/multitask_january_proxy.keras")
print("Saved history: models/multitask_january_proxy_history.json")
