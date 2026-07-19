from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import softmax


PROJECT_ROOT = Path(r"D:\Final_Year_Project\Deepfake_Dissertation")
DATASET_ROOT = PROJECT_ROOT / "dataset"
METADATA_PATH = DATASET_ROOT / "metadata" / "metadata_standardized.csv"
RF_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_mfcc" / "random_forest_mfcc.joblib"
CNN_MODEL_PATH = PROJECT_ROOT / "models" / "cnn_logmel" / "cnn_logmel_best.pt"
RESULTS_DIR = PROJECT_ROOT / "results"


def extract_mfcc_features(audio_path: str, sample_rate: int = 16000, n_mfcc: int = 20) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    feature_blocks = [mfcc, delta, delta2, zcr, rms, centroid, bandwidth, rolloff]

    features = []
    for block in feature_blocks:
        features.extend([
            float(np.mean(block)),
            float(np.std(block)),
            float(np.min(block)),
            float(np.max(block)),
        ])

    return np.array(features, dtype=np.float32)


def load_audio_fixed(audio_path: str, sample_rate: int = 16000, duration_seconds: float = 4.0) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    target_length = int(sample_rate * duration_seconds)

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    return y.astype(np.float32)


def to_logmel(y: np.ndarray, sample_rate: int = 16000, n_mels: int = 64) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=256,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-8)
    return logmel.astype(np.float32)


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


def load_cnn_model(model_path: Path, sample_rate: int, duration_seconds: float, n_mels: int) -> SimpleAudioCNN:
    dummy_audio = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)
    dummy_logmel = to_logmel(dummy_audio, sample_rate=sample_rate, n_mels=n_mels)
    time_frames = dummy_logmel.shape[1]

    model = SimpleAudioCNN(n_mels=n_mels, time_frames=time_frames)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def label_to_name(label_id: int) -> str:
    return "fake" if int(label_id) == 1 else "real"


def validate_metadata(df_test: pd.DataFrame) -> None:
    required_cols = ["filepath", "label_id", "language"]
    missing = [col for col in required_cols if col not in df_test.columns]
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")

    missing_files = [str(path) for path in df_test["filepath"] if not Path(path).exists()]
    if missing_files:
        print(f"Warning: {len(missing_files)} audio files are missing.")
        print("First few missing files:")
        for path in missing_files[:5]:
            print(path)
        raise FileNotFoundError("Some audio files listed in metadata do not exist.")


def export_rf_predictions(
    df_test: pd.DataFrame,
    model_path: Path,
    sample_rate: int,
    n_mfcc: int,
    output_csv: Path,
) -> None:
    print(f"Loading Random Forest model from: {model_path}")
    model = joblib.load(model_path)

    rows = []
    total = len(df_test)
    start_time = time.time()

    for i, (_, row) in enumerate(df_test.iterrows(), start=1):
        if i == 1 or i % 100 == 0 or i == total:
            elapsed = time.time() - start_time
            print(f"[RF] Processed {i}/{total} files | elapsed: {elapsed:.1f}s")

        x = extract_mfcc_features(
            str(row["filepath"]),
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
        ).reshape(1, -1)

        pred_id = int(model.predict(x)[0])
        prob_fake = float(model.predict_proba(x)[0][1])

        rows.append(
            {
                "file_path": str(row["filepath"]),
                "true_label": label_to_name(int(row["label_id"])),
                "pred_label": label_to_name(pred_id),
                "score_fake": prob_fake,
                "language": str(row["language"]),
            }
        )

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved RF predictions to: {output_csv}")


def export_cnn_predictions(
    df_test: pd.DataFrame,
    model_path: Path,
    sample_rate: int,
    duration_seconds: float,
    n_mels: int,
    output_csv: Path,
) -> None:
    print(f"Loading CNN model from: {model_path}")
    model = load_cnn_model(
        model_path=model_path,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        n_mels=n_mels,
    )

    rows = []
    total = len(df_test)
    start_time = time.time()

    with torch.no_grad():
        for i, (_, row) in enumerate(df_test.iterrows(), start=1):
            if i == 1 or i % 100 == 0 or i == total:
                elapsed = time.time() - start_time
                print(f"[CNN] Processed {i}/{total} files | elapsed: {elapsed:.1f}s")

            y = load_audio_fixed(
                str(row["filepath"]),
                sample_rate=sample_rate,
                duration_seconds=duration_seconds,
            )
            logmel = to_logmel(y, sample_rate=sample_rate, n_mels=n_mels)
            x = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            logits = model(x).cpu().numpy()[0]
            probs = softmax(logits)
            pred_id = int(np.argmax(probs))
            prob_fake = float(probs[1])

            rows.append(
                {
                    "file_path": str(row["filepath"]),
                    "true_label": label_to_name(int(row["label_id"])),
                    "pred_label": label_to_name(pred_id),
                    "score_fake": prob_fake,
                    "language": str(row["language"]),
                }
            )

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved CNN predictions to: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-sample test predictions for RF and CNN models.")
    parser.add_argument("--metadata", type=str, default=str(METADATA_PATH))
    parser.add_argument("--rf-model", type=str, default=str(RF_MODEL_PATH))
    parser.add_argument("--cnn-model", type=str, default=str(CNN_MODEL_PATH))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--n-mels", type=int, default=64)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading metadata...")
    df = pd.read_csv(args.metadata)
    df_test = df[df["project_split"] == "test"].reset_index(drop=True)

    if df_test.empty:
        raise ValueError("Test split is empty. Check project_split values in metadata.")

    print(f"Metadata path: {args.metadata}")
    print(f"Test samples found: {len(df_test)}")

    validate_metadata(df_test)

    rf_csv = RESULTS_DIR / "rf_test_predictions.csv"
    cnn_csv = RESULTS_DIR / "cnn_test_predictions.csv"

    print("\nExporting Random Forest predictions...")
    export_rf_predictions(
        df_test=df_test,
        model_path=Path(args.rf_model),
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        output_csv=rf_csv,
    )

    print("\nExporting CNN predictions...")
    export_cnn_predictions(
        df_test=df_test,
        model_path=Path(args.cnn_model),
        sample_rate=args.sample_rate,
        duration_seconds=args.duration,
        n_mels=args.n_mels,
        output_csv=cnn_csv,
    )

    print("\nDone.")
    print(f"Saved RF predictions to: {rf_csv}")
    print(f"Saved CNN predictions to: {cnn_csv}")


if __name__ == "__main__":
    main()
