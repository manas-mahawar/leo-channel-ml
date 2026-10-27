
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PIPELINE_DIR))

from cities import CITIES
from build_dataset import (
    run_month,
    combine_city,
    validate_monthly_csv,
    MONTHLY_DIR,
)

START_YEAR = 2020
END_YEAR = 2024

# Conservative request pacing to reduce CDS queue pressure.
PAUSE_AFTER_SUCCESS = 120
PAUSE_AFTER_FAILURE = 600

REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "batch_logs"
REPORT_FILE = REPORT_DIR / "batch_report_sequential.csv"

FIELDS = [
    "timestamp", "city", "year", "month",
    "status", "attempts", "error",
]


def process_month(city, year, month):
    target = MONTHLY_DIR / f"{city}_{year}_{month:02d}.csv"

    if target.exists():
        try:
            validate_monthly_csv(target, year, month)
            print(f"REUSE: {city} {year}-{month:02d}", flush=True)
            return "REUSED", ""
        except Exception as exc:
            print(
                f"REBUILD: {city} {year}-{month:02d}: {exc}",
                flush=True,
            )

    print(f"START: {city} {year}-{month:02d}", flush=True)

    try:
        # Exactly one month at a time; no automatic retries.
        run_month(city, year, month)

        if not target.exists():
            raise RuntimeError("Monthly CSV was not created.")

        validate_monthly_csv(target, year, month)
        print(f"SUCCESS: {city} {year}-{month:02d}", flush=True)
        return "SUCCESS", ""

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"FAILED: {city} {year}-{month:02d}: {error}", flush=True)
        return "FAILED", error


def write_report(writer, report_file, city, year, month, status, error):
    writer.writerow({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "city": city,
        "year": year,
        "month": f"{month:02d}",
        "status": status,
        "attempts": 0 if status == "REUSED" else 1,
        "error": error,
    })
    report_file.flush()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [
        (city, year, month)
        for city in CITIES
        for year in range(START_YEAR, END_YEAR + 1)
        for month in range(1, 13)
    ]

    completed = 0
    failed = 0

    print(f"Total city-months: {len(jobs)}")
    print("Workers: 1 (sequential)")
    print("Valid monthly CSVs will be reused.")
    print(f"Report: {REPORT_FILE}", flush=True)

    # Write a new report for this run; previous reports remain untouched.
    try:
        with REPORT_FILE.open("w", newline="", encoding="utf-8") as report_file:
            writer = csv.DictWriter(report_file, fieldnames=FIELDS)
            writer.writeheader()
            report_file.flush()

            for index, (city, year, month) in enumerate(jobs, start=1):
                try:
                    status, error = process_month(city, year, month)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    status = "FAILED"
                    error = f"{type(exc).__name__}: {exc}"

                write_report(
                    writer, report_file, city, year, month, status, error
                )

                completed += 1
                if status == "FAILED":
                    failed += 1

                print(
                    f"Progress: {index}/{len(jobs)} | "
                    f"Failed this run: {failed}",
                    flush=True,
                )

                if index < len(jobs):
                    if status == "FAILED":
                        print(
                            "Pausing 10 minutes after failure before "
                            "the next month.",
                            flush=True,
                        )
                        time.sleep(PAUSE_AFTER_FAILURE)
                    elif status == "SUCCESS":
                        time.sleep(PAUSE_AFTER_SUCCESS)

    except KeyboardInterrupt:
        print(
            "\nStopped by user. Completed CSVs and the report are preserved. "
            "You can run the script again later.",
            flush=True,
        )
        return

    print("\nMONTHLY PROCESSING COMPLETE")
    print(f"Processed this run: {completed}")
    print(f"Failed this run: {failed}")

    print(
        "\nChecking whether each city has a complete timeline before "
        "combining datasets...",
        flush=True,
    )

    for city in CITIES:
        expected = [
            MONTHLY_DIR / f"{city}_{year}_{month:02d}.csv"
            for year in range(START_YEAR, END_YEAR + 1)
            for month in range(1, 13)
        ]

        missing = [path.name for path in expected if not path.exists()]

        if missing:
            print(
                f"SKIP COMBINE: {city} has {len(missing)} missing monthly "
                f"file(s). First missing: {missing[0]}",
                flush=True,
            )
            continue

        try:
            for year in range(START_YEAR, END_YEAR + 1):
                for month in range(1, 13):
                    validate_monthly_csv(
                        MONTHLY_DIR / f"{city}_{year}_{month:02d}.csv",
                        year,
                        month,
                    )

            print(f"\nCombining {city}...", flush=True)
            combine_city(city)
        except Exception as exc:
            print(f"COMBINE FAILED for {city}: {exc}", flush=True)

    print(f"\nReport saved to: {REPORT_FILE}", flush=True)


if __name__ == "__main__":
    main()
