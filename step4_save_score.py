import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Define paths relative to this script.
backpath = Path(__file__).resolve().parent
data_dir = backpath / "daily_data_pkl_norm"
model_path = backpath / "best_model.pth"
output_file = backpath / "score_weekly.csv"


# Define the model architecture used for scoring.
class BiGRU_Attention_Model(nn.Module):
    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        attn_size=64,
        output_size=32,
        dropout_rate=0.5
    ):
        super().__init__()

        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_w = nn.Linear(2 * hidden_size, attn_size)
        self.attn_v = nn.Linear(attn_size, 1, bias=False)

        # Preserve the original fixed output dimension of 32.
        self.final_linear = nn.Sequential(
            nn.Linear(2 * hidden_size, 32),
            nn.Tanh()
        )

    def forward(self, x):
        gru_out, _ = self.bigru(x)
        gru_out = self.dropout(gru_out)
        e = torch.tanh(self.attn_w(gru_out))
        scores = self.attn_v(e)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        output = self.final_linear(context)
        return output


# Load trained model weights.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model = BiGRU_Attention_Model().to(device)
model.load_state_dict(
    torch.load(model_path, map_location=device)
)
model.eval()

# Collect daily files and group Friday scores by week.
weekly_data = defaultdict(list)
files = sorted([
    f for f in data_dir.glob("*.pkl")
    if f.name != "date_index.pkl"
])

for file in tqdm(files, desc="Processing"):
    date_str = file.stem

    # Score Friday observations from 2016 through 2025.
    if not ("2016" <= date_str[:4] <= "2025"):
        continue

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    if date_obj.weekday() != 4:
        continue

    with open(file, "rb") as f:
        data = pickle.load(f)

    if not data["X"]:
        continue

    X = np.stack(data["X"])  # Shape: [N, 6, 30].
    meta = data["meta"]

    # Convert inputs to [N, 30, 6].
    X_tensor = torch.tensor(
        X.transpose(0, 2, 1),
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        output = model(X_tensor)  # Shape: [N, 32].
        scores = output.mean(dim=1).cpu().numpy()

    # Preserve the original year-and-week grouping.
    week_key = (
        f"{date_obj.year}-W{date_obj.isocalendar().week:02d}"
    )

    for i in range(len(scores)):
        stock, _ = meta[i]
        weekly_data[week_key].append({
            "date": date_str,
            "stock": stock,
            "score": scores[i]
        })

# Combine weekly scores into one table.
all_results = []

for week, records in weekly_data.items():
    df = pd.DataFrame(records)
    df = df.sort_values(by="score", ascending=False)
    df = df[["date", "stock", "score"]]
    all_results.append(df)

final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv(output_file, index=False)

print(f"All scores saved to: {output_file}")