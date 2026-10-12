import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (Input,LSTM,Dropout,Dense,Reshape)

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load data
X_train = np.load("training/X_train_multitask.npy")
weather_train = np.load("training/weather_train.npy")
epl_train = np.load("training/epl_train.npy")

X_val = np.load("training/X_val_multitask.npy")
weather_val = np.load("training/weather_val.npy")
epl_val = np.load("training/epl_val.npy")

X_test = np.load("training/X_test_multitask.npy")
weather_test = np.load("training/weather_test.npy")
epl_test = np.load("training/epl_test.npy")

print("Training:", X_train.shape)
print("Weather:", weather_train.shape)
print("EPL:", epl_train.shape)

# Model
inputs = Input(shape=(60, 14))

x = LSTM(256,return_sequences=True)(inputs)
x = Dropout(0.2)(x)
x = LSTM(256,return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(256)(x)

# Weather Head
# 5 variables × 5 future hours = 25 values
weather_output = Dense(25)(x)
weather_output = Reshape((5, 5),name="weather")(weather_output)
# EPL Head
# 5 future hours:
# mean + log variance
epl_output = Dense(10,name="epl")(x)

# Build model
model = Model(inputs=inputs,outputs={"weather": weather_output,"epl": epl_output})

# Gaussian NLL for EPL
def gaussian_nll(y_true, y_pred):
    mean = y_pred[:, :5]
    log_variance = y_pred[:, 5:]
    variance = tf.exp(log_variance) + 1e-6
    loss = 0.5 * (log_variance + tf.square(y_true - mean) / variance)
    return tf.reduce_mean(loss)

# Compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss={"weather": "mse","epl": gaussian_nll},
    loss_weights={"weather": 1.0, "epl": 1.0}, metrics={ "weather": ["mae"]})

# Model summary
model.summary()

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

# Train
history = model.fit(X_train,{"weather": weather_train, "epl": epl_train},
    validation_data=(X_val,{"weather": weather_val, "epl": epl_val}),
    epochs=50, batch_size=128, callbacks=[ early_stopping],verbose=1)

# Test
results = model.evaluate(X_test,{"weather": weather_test,"epl": epl_test},verbose=0)
print("\nTest results:")
print(results)

# Save model
model.save("models/multitask_lstm.keras")
print("\nModel saved to:")
print("models/multitask_lstm.keras")