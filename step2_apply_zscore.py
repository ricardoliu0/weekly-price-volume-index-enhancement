import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from scipy.stats import rankdata, norm

# Define paths relative to this script.
root = Path(__file__).resolve().parent
input_dir = root / "daily_data_pkl"
output_dir = root / "daily_data_pkl_norm"

output_dir.mkdir(parents=True, exist_ok=True)

print("Processing each daily dataset...")

# Exclude the date index file.
files = [
    f for f in input_dir.glob("*.pkl")
    if f.name != "date_index.pkl"
]


def compute_mad(x):
    """Calculate the median and median absolute deviation."""
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return median, mad


for file in tqdm(files, desc="Processing files"):
    with open(file, "rb") as f:
        data = pickle.load(f)

    if not data["X"]:
        continue

    X = np.stack(data["X"])  # Shape: [N, 6, 30].
    labels = np.array(data["label"])  # Shape: [N].

    # Normalize each channel using all stocks and window observations
    # within the current daily dataset.
    for i in range(6):
        median, mad = compute_mad(X[:, i, :])

        # Avoid division by zero without rescaling the original MAD.
        mad = np.where(mad == 0, 1e-6, mad)
        X[:, i, :] = (X[:, i, :] - median) / mad

        # Clip normalized values at the 1st and 99th percentiles.
        lower = np.percentile(X[:, i, :], 1)
        upper = np.percentile(X[:, i, :], 99)
        X[:, i, :] = np.clip(X[:, i, :], lower, upper)

    # Transform daily label ranks into standard normal quantiles.
    ranks = rankdata(labels, method="average")

    # Avoid probabilities equal to zero or one.
    percentiles = (ranks - 0.5) / len(ranks)
    labels = norm.ppf(percentiles)

    # Preserve metadata and save the normalized dataset.
    new_data = {
        "X": [X[i] for i in range(X.shape[0])],
        "meta": data["meta"],
        "label": labels.tolist()
    }

    with open(output_dir / file.name, "wb") as f:
        pickle.dump(new_data, f)

print(f"\nAll processed files saved to: {output_dir}")