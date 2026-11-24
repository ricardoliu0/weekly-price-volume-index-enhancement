import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

# Define paths relative to this script.
backpath = Path(__file__).resolve().parent
csv_path = backpath / "market_data.csv"
output_dir = backpath / "daily_data_pkl"
output_dir.mkdir(parents=True, exist_ok=True)

# Load the input dataset.
print("Loading data...")
df = pd.read_csv(
    csv_path,
    dtype={"instrument": "category", "datetime": str}
)
df = df[
    [
        "datetime",
        "instrument",
        "close",
        "open",
        "high",
        "low",
        "vwap",
        "volume",
        "0c-10c-ret"
    ]
]

# Preserve the original backward-fill procedure within each stock.
# Backward filling uses later observations and may introduce look-ahead bias.
df = df.sort_values(["instrument", "datetime"])
df = df.groupby("instrument").apply(lambda g: g.bfill())
df = df.reset_index(drop=True)
df = df.sort_values(["datetime", "instrument"]).reset_index(drop=True)

# Shift the source return label by one observation within each stock.
# Apply the shift across year boundaries.
df["1c-11c-ret"] = df.groupby("instrument")["0c-10c-ret"].shift(-1)

features = ["close", "open", "high", "low", "vwap", "volume"]
group_dict = dict(tuple(df.groupby("instrument", observed=False)))

# Store samples by observation date.
daily_data = {}

# Construct rolling windows of 30 observations.
print("Building rolling windows...")
for stock, stock_df in tqdm(
    group_dict.items(),
    total=len(group_dict)
):
    if len(stock_df) < 30:
        continue

    stock_df = stock_df.copy()
    stock_df["volume"] = np.log1p(stock_df["volume"])
    stock_df = stock_df.reset_index(drop=True)

    feature_array = stock_df[features].values

    try:
        # Window shape: [number of windows, 6, 30].
        windows = sliding_window_view(
            feature_array,
            window_shape=(30,),
            axis=0
        )
        windows = windows[:, :, :]
    except:
        continue

    label_idx = np.arange(29, len(stock_df))

    for i, raw_window in enumerate(windows):
        window = raw_window.copy()
        last_close = raw_window[0, -1]

        # Scale the five price channels by the final closing price.
        window[0:5, :] = window[0:5, :] / last_close
        window_tensor = window.astype(np.float32)

        dt = stock_df["datetime"].iloc[label_idx[i]]
        ret = stock_df["1c-11c-ret"].iloc[label_idx[i]]

        # Skip missing labels, including the final shifted observation.
        if pd.isna(ret):
            continue

        if dt not in daily_data:
            daily_data[dt] = {
                "X": [],
                "meta": [],
                "label": []
            }

        daily_data[dt]["X"].append(window_tensor)
        daily_data[dt]["meta"].append((stock, dt))
        daily_data[dt]["label"].append(ret)

# Save one pickle file per observation date.
print("Saving daily pickle files...")
date_index = defaultdict(list)

for dt, content in tqdm(
    daily_data.items(),
    total=len(daily_data)
):
    file_path = output_dir / f"{dt}.pkl"

    with open(file_path, "wb") as f:
        pickle.dump(content, f)

    year = dt[:4]
    date_index[year].append(f"{dt}.pkl")

# Save the mapping from years to daily filenames.
index_path = output_dir / "date_index.pkl"

with open(index_path, "wb") as f:
    pickle.dump(date_index, f)

print(f"Saved {len(daily_data)} daily pickle files.")
print(f"Date index saved to: {index_path}")