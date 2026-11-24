import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Set random seeds.
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Define paths relative to this script.
backpath = Path(__file__).resolve().parent
data_dir = backpath / "daily_data_pkl_norm"
model_path = backpath / "best_model.pth"
log_path = backpath / "training_log.csv"


# Load stock samples from the selected years.
class StockDataset(Dataset):
    def __init__(self, data_dir, years, cutoff_date=None):
        self.samples = []
        data_dir = Path(data_dir)

        for fname in os.listdir(data_dir):
            if not fname.endswith(".pkl"):
                continue

            year = int(fname[:4])
            if year in years:
                if (
                    cutoff_date
                    and pd.to_datetime(fname[:10])
                    > pd.to_datetime(cutoff_date)
                ):
                    continue

                with open(data_dir / fname, "rb") as f:
                    data = pickle.load(f)

                for x, y in zip(data["X"], data["label"]):
                    self.samples.append(
                        (x.astype(np.float32), float(y))
                    )

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]

        # Convert [6, 30] to [30, 6].
        x = np.transpose(x, (1, 0))
        return x, y


def get_dataloader(
    data_dir,
    years,
    batch_size=1024,
    shuffle=True,
    cutoff_date=None
):
    dataset = StockDataset(data_dir, years, cutoff_date)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


# Define the bidirectional GRU with temporal attention.
class BiGRU_Attention_Model(nn.Module):
    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        attn_size=64,
        output_size=32,
        dropout_rate=0.4
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

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        gru_out, _ = self.bigru(x)
        gru_out = self.dropout(gru_out)
        e = torch.tanh(self.attn_w(gru_out))
        scores = self.attn_v(e)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        output = self.final_linear(context)
        return output


def spearman_ic(pred, true):
    """Average Spearman correlation across output channels."""
    pred = pred.detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    ic_list = []

    for i in range(pred.shape[1]):
        ic = spearmanr(pred[:, i], true)[0]
        if not np.isnan(ic):
            ic_list.append(ic)

    return np.mean(ic_list) if ic_list else 0.0


def compute_epoch_loss(preds, targets, criterion):
    # Average output channels to obtain one prediction per sample.
    pred_mean = preds.mean(dim=1)
    mse = criterion(pred_mean, targets)
    rank_ic = spearman_ic(preds, targets)

    # Preserve the original combined loss value.
    # RankIC is detached, so only MSE contributes gradients.
    loss = mse + 2 * (1 - rank_ic) ** 2
    return loss, mse.item(), rank_ic, loss.item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_mse, total_rank_ic, total_loss = 0.0, 0.0, 0.0
    count = 0

    for x, y in tqdm(loader, desc="Training"):
        x = x.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()
        out = model(x)
        loss, mse_val, rank_ic, loss_val = compute_epoch_loss(
            out, y, criterion
        )
        loss.backward()
        optimizer.step()

        total_mse += mse_val * x.size(0)
        total_rank_ic += rank_ic * x.size(0)
        total_loss += loss_val * x.size(0)
        count += x.size(0)

    return (
        total_mse / count,
        total_rank_ic / count,
        total_loss / count
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_mse, total_rank_ic, total_loss = 0.0, 0.0, 0.0
    count = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x = x.to(device)
            y = y.to(device).float()

            out = model(x)
            loss, mse, rank_ic, loss_val = compute_epoch_loss(
                out, y, criterion
            )

            total_mse += mse * x.size(0)
            total_rank_ic += rank_ic * x.size(0)
            total_loss += loss_val * x.size(0)
            count += x.size(0)

    return (
        total_mse / count,
        total_rank_ic / count,
        total_loss / count
    )


def main():
    set_seed(1)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = BiGRU_Attention_Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.MSELoss()

    # Apply the original cutoff to the initial training loader.
    train_loader = get_dataloader(
        data_dir,
        years=range(2007, 2015),
        cutoff_date="2014-12-15"
    )
    val_loader = get_dataloader(
        data_dir,
        years=[2015],
        shuffle=False
    )

    # Inspect the input batch shape: [batch_size, 30, 6].
    sample_batch = next(iter(train_loader))
    x_sample, y_sample = sample_batch
    print(f"[DEBUG] x_sample shape: {x_sample.shape}")

    best_val_loss = float("inf")
    best_rankic = -float("inf")
    patience, counter = 10, 0
    rankic_wait = 0
    history = []

    for epoch in range(1, 201):
        # Preserve the original loader recreation without a cutoff.
        # The initial cutoff therefore does not apply during training.
        train_loader = get_dataloader(
            data_dir,
            years=range(2007, 2015)
        )

        print(
            f"\nEpoch {epoch} | Current LR: "
            f"{optimizer.param_groups[0]['lr']:.6f}"
        )

        train_mse, train_rankic, train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_mse, val_rankic, val_loss = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Train MSE: {train_mse:.4f}, "
            f"RankIC: {train_rankic:.4f}, Loss: {train_loss:.4f}"
        )
        print(
            f"Val MSE: {val_mse:.4f}, "
            f"RankIC: {val_rankic:.4f}, Loss: {val_loss:.4f}"
        )

        history.append({
            "Epoch": epoch,
            "Train MSE": train_mse,
            "Train RankIC": train_rankic,
            "Train Loss": train_loss,
            "Val MSE": val_mse,
            "Val RankIC": val_rankic,
            "Val Loss": val_loss,
            "Learning Rate": optimizer.param_groups[0]["lr"]
        })

        # Save the best model and stop after repeated non-improvement.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), model_path)
            print("Best model saved.")
        else:
            counter += 1
            print(
                f"No improvement. Patience: {counter}/{patience}"
            )
            if counter >= patience:
                print("Early stopping.")
                break

        # Halve the learning rate after three epochs without
        # a validation RankIC improvement greater than 0.0001.
        if val_rankic > best_rankic + 1e-4:
            best_rankic = val_rankic
            rankic_wait = 0
        else:
            rankic_wait += 1
            print(
                f"No RankIC improvement. Wait count: {rankic_wait}/3"
            )

            if rankic_wait >= 3:
                old_lr = optimizer.param_groups[0]["lr"]
                new_lr = old_lr * 0.5

                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr

                print(
                    f"Learning rate reduced: "
                    f"{old_lr:.6f} -> {new_lr:.6f}"
                )
                rankic_wait = 0

    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()