import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load prepared data
X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

X_val = np.load("data/processed/X_val.npy")
y_val = np.load("data/processed/y_val.npy")

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Test data:", X_test.shape)

# LSTM model

model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(60, 14)),Dropout(0.2), LSTM(256, return_sequences=True), 
    Dropout(0.2), LSTM(256),Dense(5)])

# Compile

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss="mse",metrics=["mae"])
model.summary()

# Train

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

history = model.fit(X_train,y_train,validation_data=(X_val, y_val),epochs=50,batch_size=128,callbacks=[early_stopping],verbose=1)

# Test
test_loss, test_mae = model.evaluate(X_test, y_test,verbose=0) 

print("\nTest results:")
print("MSE:", test_loss)
print("MAE:", test_mae)

# Save model

model.save("models/lstm_baseline.keras")

print("\nModel saved to:")
print("models/lstm_baseline.keras")