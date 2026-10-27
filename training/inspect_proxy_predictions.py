import numpy as np

PREFIX = "data/processed/athens_proxy"

y_true = np.load(f"{PREFIX}_y_test.npy")
y_pred = np.load(f"{PREFIX}_attenuation_predictions.npy")
pred_std = np.load(f"{PREFIX}_attenuation_prediction_std.npy")

error = y_true - y_pred

print("True target statistics:")
print("Mean:", np.mean(y_true))
print("Std:", np.std(y_true))
print("Min:", np.min(y_true))
print("Max:", np.max(y_true))

print("\nPredicted mean statistics:")
print("Mean:", np.mean(y_pred))
print("Std:", np.std(y_pred))
print("Min:", np.min(y_pred))
print("Max:", np.max(y_pred))

print("\nPredicted uncertainty statistics:")
print("Mean standard deviation:", np.mean(pred_std))
print("Min standard deviation:", np.min(pred_std))
print("Max standard deviation:", np.max(pred_std))

print("\nError statistics:")
print("Mean error:", np.mean(error))
print("Error std:", np.std(error))
print("Maximum absolute error:", np.max(np.abs(error)))
