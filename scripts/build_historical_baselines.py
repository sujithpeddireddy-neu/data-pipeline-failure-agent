import csv
from pathlib import Path
from statistics import median

PROFILE_DIR = Path(r"C:\Capstone\datasets\profiles")

SUMMARY_FILE = PROFILE_DIR / "monthly_summary.csv"
QUALITY_FILE = PROFILE_DIR / "monthly_quality.csv"

OUTPUT_FILE = PROFILE_DIR / "monthly_reference_baselines.csv"

WINDOW = 6
MIN_HISTORY = 3


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_split(month):
    if month <= "2022-12":
        return "train"
    elif month <= "2023-12":
        return "validation"
    return "test"


def pct_change(current, baseline):
    if baseline == 0:
        return None
    return round(
        100 * (current - baseline) / baseline, 4
    )


summary = {
    row["month"]: row
    for row in read_csv(SUMMARY_FILE)
}

quality = {
    row["month"]: row
    for row in read_csv(QUALITY_FILE)
}

months = sorted(set(summary) & set(quality))

if set(summary) != set(quality):
    raise ValueError(
        "Monthly summary and quality reports cover different months."
    )

records = []

for index, month in enumerate(months):

    # Historical reference must exclude the current month.
    previous_months = months[max(0, index - WINDOW):index]

    if len(previous_months) < MIN_HISTORY:
        print(f"Skipping {month}: insufficient history")
        continue

    historical_volume = [
        float(summary[m]["average_rows_per_day"])
        for m in previous_months
    ]

    historical_null = [
        float(quality[m]["null_ratecode_pct"])
        for m in previous_months
    ]

    historical_invalid = [
        float(quality[m]["negative_distance_pct"])
        for m in previous_months
    ]

    baseline_volume = median(historical_volume)
    baseline_null = median(historical_null)
    baseline_invalid = median(historical_invalid)

    current_volume = float(
        summary[month]["average_rows_per_day"]
    )

    current_null = float(
        quality[month]["null_ratecode_pct"]
    )

    current_invalid = float(
        quality[month]["negative_distance_pct"]
    )

    record = {
        "month": month,
        "split": get_split(month),
        "history_months": len(previous_months),

        "baseline_rows_per_day": round(
            baseline_volume, 4
        ),
        "current_rows_per_day": current_volume,
        "volume_deviation_pct": pct_change(
            current_volume, baseline_volume
        ),

        "baseline_null_ratecode_pct": round(
            baseline_null, 4
        ),
        "current_null_ratecode_pct": current_null,
        "null_deviation_pp": round(
            current_null - baseline_null, 4
        ),

        "baseline_invalid_distance_pct": round(
            baseline_invalid, 4
        ),
        "current_invalid_distance_pct": current_invalid,
        "invalid_distance_deviation_pp": round(
            current_invalid - baseline_invalid, 4
        ),
    }

    records.append(record)

    print(
        f"{month} [{record['split']}] | "
        f"Volume change: {record['volume_deviation_pct']}% | "
        f"Null change: {record['null_deviation_pp']} pp"
    )


if not records:
    raise RuntimeError("No baseline records generated.")

with OUTPUT_FILE.open(
    "w", newline="", encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=records[0].keys()
    )

    writer.writeheader()
    writer.writerows(records)


print("\nHistorical baseline generation completed.")
print(f"Months processed: {len(records)}")
print(f"Output: {OUTPUT_FILE}")