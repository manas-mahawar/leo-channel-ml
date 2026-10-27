from data_pipeline.cities import CITIES
from data_pipeline.build_dataset import MONTHLY_DIR, validate_monthly_csv

total = 0
invalid = 0

for city in CITIES:
    valid_count = 0
    invalid_count = 0

    for year in range(2020, 2025):
        for month in range(1, 13):
            path = MONTHLY_DIR / f"{city}_{year}_{month:02d}.csv"

            if path.exists():
                try:
                    validate_monthly_csv(path, year, month)
                    valid_count += 1
                except Exception:
                    invalid_count += 1

    total += valid_count
    invalid += invalid_count
    print(f"{city}: {valid_count}/60 valid, {invalid_count} invalid")

print(f"\nTOTAL: {total}/360 valid monthly CSVs")
print(f"INVALID: {invalid}")
