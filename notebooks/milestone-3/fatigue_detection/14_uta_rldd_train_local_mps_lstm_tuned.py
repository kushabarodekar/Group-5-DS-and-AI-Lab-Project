from pathlib import Path
import time
import random
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CANDIDATE_SEQUENCE_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "milestone-3"
    / "model_outputs"
    / "uta_rldd_candidate_sequence_records.csv"
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "milestone-3" / "training_outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_ROOT / "best_cnn_lstm_tuned_mps.pth"
HISTORY_CSV_PATH = OUTPUT_ROOT / "cnn_lstm_tuned_training_history.csv"
TEST_REPORT_JSON_PATH = OUTPUT_ROOT / "cnn_lstm_tuned_test_report.json"
CONFUSION_MATRIX_CSV_PATH = OUTPUT_ROOT / "cnn_lstm_tuned_confusion_matrix.csv"


# =========================
# Config
# =========================
SEED = 42
BATCH_SIZE = 4
NUM_WORKERS = 0
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 5e-4
EPOCHS = 5
EARLY_STOPPING_PATIENCE = 2
NUM_CLASSES = 3
CLASS_NAMES = ["Safe", "Caution", "High Risk"]
HIDDEN_SIZE = 256
DROPOUT = 0.5


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)


# =========================
# Device selection
# =========================
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# =========================
# Load metadata
# =========================
candidate_sequence_df = pd.read_csv(CANDIDATE_SEQUENCE_CSV)
print(f"Loaded candidate sequence records: {len(candidate_sequence_df)}")

class_to_idx = {
    "Safe": 0,
    "Caution": 1,
    "High Risk": 2,
}
idx_to_class = {v: k for k, v in class_to_idx.items()}

candidate_sequence_df["label_idx"] = candidate_sequence_df["project_class"].map(class_to_idx)

candidate_sequence_df["file_exists"] = candidate_sequence_df["sequence_path"].apply(lambda x: (PROJECT_ROOT / Path(x)).exists())
missing_count = (~candidate_sequence_df["file_exists"]).sum()
print(f"Missing sequence files: {missing_count}")
if missing_count > 0:
    missing_rows = candidate_sequence_df[~candidate_sequence_df["file_exists"]].head(10)
    print("Sample missing rows:")
    print(missing_rows[["sequence_path", "split", "project_class"]])
    raise FileNotFoundError("Some sequence files are missing. Fix paths before training.")

train_df = candidate_sequence_df[candidate_sequence_df["split"] == "train"].reset_index(drop=True)
val_df = candidate_sequence_df[candidate_sequence_df["split"] == "val"].reset_index(drop=True)
test_df = candidate_sequence_df[candidate_sequence_df["split"] == "test"].reset_index(drop=True)

print("\nSequence counts by split:")
print(candidate_sequence_df["split"].value_counts())

print("\nSequence counts by split and class:")
print(candidate_sequence_df.groupby(["split", "project_class"]).size())


# =========================
# Dataset
# =========================
class FatigueSequenceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        sequence = np.load(PROJECT_ROOT / row["sequence_path"]).astype(np.float32) / 255.0
        # from (T, H, W, C) -> (T, C, H, W)
        sequence = np.transpose(sequence, (0, 3, 1, 2))

        label = int(row["label_idx"])

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


train_dataset = FatigueSequenceDataset(train_df)
val_dataset = FatigueSequenceDataset(val_df)
test_dataset = FatigueSequenceDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

print("\nDataloaders created.")
sample_x, sample_y = next(iter(train_loader))
print("Sample batch sequence shape:", sample_x.shape)
print("Sample batch label shape:", sample_y.shape)


# =========================
# Model
# =========================
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return x


class CNNLSTMModel(nn.Module):
    def __init__(self, hidden_size: int = 256, num_classes: int = 3, dropout: float = 0.5):
        super().__init__()
        self.encoder = CNNEncoder()
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        frame_features = self.encoder(x)              # (B*T, 256)
        frame_features = frame_features.view(b, t, 256)  # (B, T, 256)

        _, (h_n, _) = self.lstm(frame_features)       # h_n: (1, B, hidden)
        final_hidden = h_n[-1]                        # (B, hidden)
        final_hidden = self.dropout(final_hidden)

        logits = self.classifier(final_hidden)
        return logits


model = CNNLSTMModel(hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES, dropout=DROPOUT).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("\nModel initialized.")
print(model)


# =========================
# Train / Eval helpers
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            running_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

            all_labels.extend(batch_y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1, all_labels, all_preds


# =========================
# Training loop
# =========================
history = []
best_val_f1 = -1.0
best_epoch = -1
patience_counter = 0

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    epoch_start = time.time()

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE)

    epoch_time = time.time() - epoch_start

    row = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "epoch_time_sec": epoch_time,
    }
    history.append(row)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
        f"Val Macro F1: {val_f1:.4f} | Time: {epoch_time:.1f}s"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch + 1
        patience_counter = 0

        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "config": {
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "hidden_size": HIDDEN_SIZE,
                    "dropout": DROPOUT,
                    "epochs": EPOCHS,
                    "sequence_csv": str(CANDIDATE_SEQUENCE_CSV),
                },
            },
            BEST_MODEL_PATH,
        )
        print(f"  -> Saved new best model to: {BEST_MODEL_PATH}")
    else:
        patience_counter += 1
        print(f"  -> No improvement. Early-stop counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("\nEarly stopping triggered.")
        break


# =========================
# Save history
# =========================
history_df = pd.DataFrame(history)
history_df.to_csv(HISTORY_CSV_PATH, index=False)

print("\nTraining history saved to:")
print(HISTORY_CSV_PATH)

print(f"\nBest epoch: {best_epoch}")
print(f"Best validation macro F1: {best_val_f1:.4f}")


# =========================
# Final test evaluation
# =========================
checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)

print("\nFinal Test Metrics:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Macro F1: {test_f1:.4f}")

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

report_payload = {
    "best_epoch": best_epoch,
    "best_val_macro_f1": best_val_f1,
    "test_loss": test_loss,
    "test_accuracy": test_acc,
    "test_macro_f1": test_f1,
    "classification_report": report,
}

with open(TEST_REPORT_JSON_PATH, "w") as f:
    json.dump(report_payload, f, indent=2)

cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
cm_df.to_csv(CONFUSION_MATRIX_CSV_PATH, index=True)

print("\nClassification report saved to:")
print(TEST_REPORT_JSON_PATH)

print("Confusion matrix saved to:")
print(CONFUSION_MATRIX_CSV_PATH)

print("\nText Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

print("\nConfusion Matrix:")
print(cm_df)
