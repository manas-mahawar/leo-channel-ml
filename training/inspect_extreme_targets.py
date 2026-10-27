import pandas as pd

FILE = "data/processed/athens_2020_01_targets.csv"

df = pd.read_csv(FILE, parse_dates=["valid_time"])
target = "total_attenuation_proxy_dB"

print("Target distribution:")
print(df[target].describe(percentiles=[0.90, 0.95, 0.99]))

print("\nLargest 10 attenuation-proxy values:")
print(
    df.nlargest(10, target)[["valid_time", target]]
    .to_string(index=False)
)

print("\nCounts above thresholds:")
for threshold in [1.0, 2.0, 3.0, 4.0]:
    count = (df[target] > threshold).sum()
    print(f"> {threshold:.1f} dB: {count} hours")
