from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(r"D:\Final_Year_Project\Deepfake_Dissertation")
DATASET_ROOT = PROJECT_ROOT / "dataset"
METADATA_PATH = DATASET_ROOT / "metadata" / "metadata_standardized.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "cnn_logmel"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sample_rate: int = 16000,
        duration_seconds: float = 4.0,
        n_mels: int = 64,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration_seconds)
        self.n_mels = n_mels

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, audio_path: str) -> np.ndarray:
        y, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        if len(y) < self.target_length:
            y = np.pad(y, (0, self.target_length - len(y)))
        else:
            y = y[: self.target_length]

        return y.astype(np.float32)

    def _to_logmel(self, y: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=1024,
            hop_length=256,
        )
        logmel = librosa.power_to_db(mel, ref=np.max)
        logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-8)
        return logmel.astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        y = self._load_audio(row["filepath"])
        logmel = self._to_logmel(y)
        features = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(int(row["label_id"]), dtype=torch.long)
        return features, label


class SimpleAudioCNN(nn.Module):
    def __init__(self, n_mels: int = 64, time_frames: int = 251) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, time_frames)
            flattened_dim = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


@torch.no_grad()
def evaluate_by_language(
    model: nn.Module,
    df_test: pd.DataFrame,
    batch_size: int,
    sample_rate: int,
    duration_seconds: float,
    n_mels: int,
    device: torch.device,
) -> dict:
    results = {}

    for language in sorted(df_test["language"].unique()):
        lang_df = df_test[df_test["language"] == language].reset_index(drop=True)
        dataset = AudioDataset(
            lang_df,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            n_mels=n_mels,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_labels = []
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        results[language] = {
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "count": int(len(lang_df)),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN on log-Mel spectrograms for real/fake audio detection.")
    parser.add_argument("--metadata", type=str, default=str(METADATA_PATH))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=4.0, help="Audio duration in seconds after pad/crop.")
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.random_state)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.metadata)
    train_df = df[df["project_split"] == "train"].reset_index(drop=True)
    test_df = df[df["project_split"] == "test"].reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty. Check project_split values in metadata.")

    train_dataset = AudioDataset(
        train_df,
        sample_rate=args.sample_rate,
        duration_seconds=args.duration,
        n_mels=args.n_mels,
    )
    test_dataset = AudioDataset(
        test_df,
        sample_rate=args.sample_rate,
        duration_seconds=args.duration,
        n_mels=args.n_mels,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    sample_x, _ = train_dataset[0]
    _, n_mels, time_frames = sample_x.shape
    model = SimpleAudioCNN(n_mels=n_mels, time_frames=time_frames).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    best_test_acc = -1.0
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _, _ = evaluate_model(model, test_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state_dict is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state_dict)
    model.to(device)

    final_test_loss, final_test_acc, y_test, y_pred = evaluate_model(model, test_loader, criterion, device)
    report = classification_report(y_test, y_pred, target_names=["real", "fake"], digits=4)
    cm = confusion_matrix(y_test, y_pred)
    language_results = evaluate_by_language(
        model,
        test_df,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        duration_seconds=args.duration,
        n_mels=args.n_mels,
        device=device,
    )

    print("\nBest Test Accuracy:")
    print(f"{final_test_acc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nLanguage-wise Accuracy:")
    for language, metrics in language_results.items():
        print(f"{language}: {metrics['accuracy']:.4f} ({metrics['count']} samples)")

    torch.save(best_state_dict, OUTPUT_DIR / "cnn_logmel_best.pt")
    joblib.dump(history, OUTPUT_DIR / "training_history.joblib")

    metrics = {
        "run_time": datetime.now().isoformat(),
        "device": str(device),
        "best_test_accuracy": float(final_test_acc),
        "test_loss": float(final_test_loss),
        "confusion_matrix": cm.tolist(),
        "language_wise_accuracy": language_results,
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "n_mels": int(args.n_mels),
        "duration_seconds": float(args.duration),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_text = []
    report_text.append(f"Run time: {datetime.now().isoformat()}")
    report_text.append(f"Device: {device}")
    report_text.append(f"Metadata: {args.metadata}")
    report_text.append(f"Train samples: {len(train_df)}")
    report_text.append(f"Test samples: {len(test_df)}")
    report_text.append(f"Best Test Accuracy: {final_test_acc:.4f}")
    report_text.append("")
    report_text.append("Classification Report:")
    report_text.append(report)
    report_text.append("")
    report_text.append("Confusion Matrix:")
    report_text.append(str(cm))
    report_text.append("")
    report_text.append("Language-wise Accuracy:")
    for language, metrics_dict in language_results.items():
        report_text.append(f"{language}: {metrics_dict['accuracy']:.4f} ({metrics_dict['count']} samples)")

    with open(OUTPUT_DIR / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_text))

    print("\nSaved model:")
    print(OUTPUT_DIR / "cnn_logmel_best.pt")
    print("\nSaved metrics:")
    print(OUTPUT_DIR / "metrics.json")
    print("\nSaved report:")
    print(OUTPUT_DIR / "results.txt")


if __name__ == "__main__":
    main()
