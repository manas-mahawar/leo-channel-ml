
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

PREFIX = "data/processed/athens_proxy"

y_train = np.load(f"{PREFIX}_y_train.npy")
y_test = np.load(f"{PREFIX}_y_test.npy")
X_test = np.load(f"{PREFIX}_X_test.npy")

# The latest gas_mean feature is feature index 7.
# It is NOT the total attenuation target, so it cannot
# be used as the persistence value.
#
# For a valid persistence baseline, load the original
# target timeline and obtain the target immediately
# before each test forecast begins.

targets = np.load(f"{PREFIX}_t_test.npy")
target_df = __import__("pandas").read_csv(
    "data/processed/athens_2020_01_targets.csv",
    parse_dates=["valid_time"],
)

target_values = target_df["total_attenuation_proxy_dB"].to_numpy()
timestamps = target_df["valid_time"].to_numpy()

predictions = []

for start_time in targets:
    index = np.searchsorted(timestamps, start_time) - 1
    if index < 0:
        raise ValueError("No preceding target available.")
    predictions.append(
        np.full(y_test.shape[1], target_values[index])
    )

y_pred = np.asarray(predictions)

mae = mean_absolute_error(y_test.ravel(), y_pred.ravel())
rmse = np.sqrt(mean_squared_error(y_test.ravel(), y_pred.ravel()))

print("Persistence baseline")
print(f"Test MAE:  {mae:.6f} dB")
print(f"Test RMSE: {rmse:.6f} dB")
