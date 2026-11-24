from pathlib import Path

import pandas as pd

# Define paths relative to this script.
backpath = Path(__file__).resolve().parent
score_path = backpath / "score_weekly.csv"
st_path = backpath / "ST.csv"
status_path = backpath / "trading_status.csv"
output_path = backpath / "top100_filtered.csv"

# Preserve the trading-status value used in the source dataset.
TRADING_STATUS_VALUE = "交易"

# Load stock scores and eligibility tables.
score_df = pd.read_csv(score_path, dtype={"stock": str})
score_df.columns = score_df.columns.str.strip()

st_df = pd.read_csv(st_path, index_col=0)
status_df = pd.read_csv(status_path, index_col=0)

# Preserve the original eligibility rules:
# missing ST entries and an active trading status.
mask_st = st_df.isna()
mask_status = status_df.eq(TRADING_STATUS_VALUE)
mask_valid = mask_st & mask_status

all_results = []

# Select up to 100 eligible stocks per scoring date.
for date, group in score_df.groupby("date"):
    if date not in mask_valid.index:
        continue

    valid_stocks = mask_valid.loc[date]
    valid_stocks_set = set(
        valid_stocks[valid_stocks].index
    )

    filtered = group[
        group["stock"].isin(valid_stocks_set)
    ].copy()
    filtered["score"] = pd.to_numeric(filtered["score"])

    top100 = filtered.sort_values(
        by="score",
        ascending=False
    ).head(100)

    all_results.append(top100)

# Save the combined selection table.
final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv(output_path, index=False)

print(f"Filtered Top 100 selections saved to: {output_path}")