
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

PREFIX = "data/processed/athens_proxy"

# Load training and test targets
y_train = np.load(f"{PREFIX}_y_train.npy")
y_test = np.load(f"{PREFIX}_y_test.npy")

# Baseline: predict the training-set mean at every future hour
baseline_value = y_train.mean()
y_pred = np.full_like(y_test, baseline_value)

mae = mean_absolute_error(y_test.ravel(), y_pred.ravel())
rmse = np.sqrt(mean_squared_error(y_test.ravel(), y_pred.ravel()))

print("Baseline: training-target mean")
print(f"Training mean: {baseline_value:.6f} dB")
print(f"Test MAE:      {mae:.6f} dB")
print(f"Test RMSE:     {rmse:.6f} dB")
print(f"Test samples:  {y_test.size}")
