from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(r"D:\Final_Year_Project\Deepfake_Dissertation")
MODEL_PATH = PROJECT_ROOT / "models" / "cnn_logmel" / "cnn_logmel_best.pt"
METADATA_PATH = PROJECT_ROOT / "dataset" / "metadata" / "metadata_standardized.csv"


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


def load_audio(audio_path: str, sample_rate: int, duration_seconds: float) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    target_length = int(sample_rate * duration_seconds)

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    return y.astype(np.float32)


def to_logmel(y: np.ndarray, sample_rate: int, n_mels: int) -> np.ndarray:
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


def extract_summary_features(y: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> dict[str, float]:
    duration_seconds = float(len(y) / sample_rate)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)

    return {
        "duration_seconds": duration_seconds,
        "rms_mean": float(np.mean(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "rolloff_mean": float(np.mean(rolloff)),
        "mfcc1_mean": float(np.mean(mfcc[0])),
        "mfcc2_mean": float(np.mean(mfcc[1])),
        "mfcc3_mean": float(np.mean(mfcc[2])),
    }


def build_reference_statistics(
    metadata_path: Path, sample_rate: int, duration_seconds: float
) -> tuple[dict[str, float], dict[str, float]]:
    df = pd.read_csv(metadata_path)
    train_df = df[df["project_split"] == "train"].reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Training split is empty in metadata. Cannot build explanation statistics.")

    real_rows: list[dict[str, float]] = []
    fake_rows: list[dict[str, float]] = []

    for _, row in train_df.iterrows():
        y = load_audio(str(row["filepath"]), sample_rate=sample_rate, duration_seconds=duration_seconds)
        summary = extract_summary_features(y, sample_rate=sample_rate)
        if int(row["label_id"]) == 0:
            real_rows.append(summary)
        else:
            fake_rows.append(summary)

    if not real_rows or not fake_rows:
        raise ValueError("Need both real and fake training samples to build explanation statistics.")

    real_df = pd.DataFrame(real_rows)
    fake_df = pd.DataFrame(fake_rows)
    return real_df.mean().to_dict(), fake_df.mean().to_dict()


def build_reason_lines(
    summary_features: dict[str, float],
    pred_label: str,
    real_reference: dict[str, float],
    fake_reference: dict[str, float],
    top_k: int = 4,
) -> list[str]:
    predicted_reference = fake_reference if pred_label == "fake" else real_reference
    opposite_reference = real_reference if pred_label == "fake" else fake_reference

    scored = []
    for feature_name, value in summary_features.items():
        if feature_name == "duration_seconds":
            continue

        pred_diff = abs(value - predicted_reference[feature_name])
        opp_diff = abs(value - opposite_reference[feature_name])
        support_score = opp_diff - pred_diff

        scored.append(
            {
                "feature": feature_name,
                "value": value,
                "predicted_avg": predicted_reference[feature_name],
                "opposite_avg": opposite_reference[feature_name],
                "score": support_score,
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)

    lines = []
    for item in scored[:top_k]:
        lines.append(
            f"{item['feature']}={item['value']:.4f}, which is closer to the average "
            f"{pred_label} value ({item['predicted_avg']:.4f}) than to the opposite class "
            f"({item['opposite_avg']:.4f})"
        )
    return lines


def build_user_friendly_description(
    pred_label: str,
    real_pct: float,
    fake_pct: float,
    reasons: list[str],
) -> str:
    confidence = fake_pct if pred_label == "fake" else real_pct
    confidence_word = "strong" if confidence >= 85 else "moderate" if confidence >= 65 else "limited"

    intro = (
        f"The model classified this audio as {pred_label} with {confidence:.2f}% confidence. "
        f"This is a {confidence_word} decision based on the acoustic profile of the clip."
    )

    if pred_label == "fake":
        body = (
            "In plain terms, the clip's spectral and voice-texture patterns look more like the "
            "synthetic examples seen during training than the natural human recordings."
        )
    else:
        body = (
            "In plain terms, the clip's spectral and voice-texture patterns look more like the "
            "natural human recordings seen during training than the synthetic examples."
        )

    reason_text = " Key supporting cues: " + "; ".join(reasons[:3]) + "."
    return intro + " " + body + reason_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict whether a single audio file is real or fake using the CNN log-Mel model."
    )
    parser.add_argument("--file", required=True, help="Path to the audio file to classify.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to the trained CNN .pt model.")
    parser.add_argument(
        "--metadata",
        default=str(METADATA_PATH),
        help="Path to metadata_standardized.csv used to build reference statistics for explanations.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=4.0, help="Audio duration in seconds after pad/crop.")
    parser.add_argument("--n-mels", type=int, default=64)
    args = parser.parse_args()

    audio_path = Path(args.file)
    model_path = Path(args.model)
    metadata_path = Path(args.metadata)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    y = load_audio(str(audio_path), sample_rate=args.sample_rate, duration_seconds=args.duration)
    logmel = to_logmel(y, sample_rate=args.sample_rate, n_mels=args.n_mels)
    summary_features = extract_summary_features(y, sample_rate=args.sample_rate)
    real_reference, fake_reference = build_reference_statistics(
        metadata_path=metadata_path,
        sample_rate=args.sample_rate,
        duration_seconds=args.duration,
    )
    input_tensor = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    _, _, n_mels, time_frames = input_tensor.shape
    model = SimpleAudioCNN(n_mels=n_mels, time_frames=time_frames)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_id = int(np.argmax(probabilities))
    pred_label = "fake" if pred_id == 1 else "real"
    real_pct = float(probabilities[0] * 100.0)
    fake_pct = float(probabilities[1] * 100.0)
    reason_lines = build_reason_lines(
        summary_features=summary_features,
        pred_label=pred_label,
        real_reference=real_reference,
        fake_reference=fake_reference,
    )
    user_description = build_user_friendly_description(pred_label, real_pct, fake_pct, reason_lines)

    print("\nCNN Prediction Result")
    print(f"File: {audio_path}")
    print(f"Predicted label: {pred_label}")
    print(f"Real probability: {real_pct:.2f}%")
    print(f"Fake probability: {fake_pct:.2f}%")
    print("\nExtracted audio features:")
    for feature_name, value in summary_features.items():
        print(f"- {feature_name}: {value:.4f}")
    print("\nReason for prediction:")
    for line in reason_lines:
        print(f"- {line}")
    print("\nUser-friendly explanation:")
    print(user_description)


if __name__ == "__main__":
    main()
