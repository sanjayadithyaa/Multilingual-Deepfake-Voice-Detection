from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"D:\Final_Year_Project\Deepfake_Dissertation")
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_mfcc" / "random_forest_mfcc.joblib"
METADATA_PATH = PROJECT_ROOT / "dataset" / "metadata" / "metadata_standardized.csv"


def extract_features(audio_path: str, sample_rate: int = 16000, n_mfcc: int = 20) -> np.ndarray:
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
        features.extend(
            [
                float(np.mean(block)),
                float(np.std(block)),
                float(np.min(block)),
                float(np.max(block)),
            ]
        )

    return np.array(features, dtype=np.float32)


def get_feature_names() -> list[str]:
    blocks = ["mfcc", "delta", "delta2", "zcr", "rms", "centroid", "bandwidth", "rolloff"]
    stats = ["mean", "std", "min", "max"]
    return [f"{block}_{stat}" for block in blocks for stat in stats]


def build_reference_statistics(
    metadata_path: Path, sample_rate: int, n_mfcc: int
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(metadata_path)
    train_df = df[df["project_split"] == "train"].reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Training split is empty in metadata. Cannot build explanation statistics.")

    real_features = []
    fake_features = []

    for _, row in train_df.iterrows():
        feature_vector = extract_features(str(row["filepath"]), sample_rate=sample_rate, n_mfcc=n_mfcc)
        if int(row["label_id"]) == 0:
            real_features.append(feature_vector)
        else:
            fake_features.append(feature_vector)

    if not real_features or not fake_features:
        raise ValueError("Need both real and fake training samples to build explanation statistics.")

    real_mean = np.mean(np.vstack(real_features), axis=0)
    fake_mean = np.mean(np.vstack(fake_features), axis=0)
    return real_mean, fake_mean


def explain_prediction(
    features: np.ndarray,
    pred_label: str,
    model,
    real_mean: np.ndarray,
    fake_mean: np.ndarray,
    top_k: int = 5,
) -> list[str]:
    feature_names = get_feature_names()
    importances = getattr(model, "feature_importances_", np.ones(features.shape[0], dtype=np.float32))
    predicted_mean = fake_mean if pred_label == "fake" else real_mean
    other_mean = real_mean if pred_label == "fake" else fake_mean

    reasons = []
    for idx, feature_value in enumerate(features):
        predicted_distance = abs(float(feature_value) - float(predicted_mean[idx]))
        other_distance = abs(float(feature_value) - float(other_mean[idx]))
        closeness_gain = other_distance - predicted_distance
        weighted_gain = closeness_gain * float(importances[idx])

        direction = "higher" if float(feature_value) >= float(other_mean[idx]) else "lower"
        reasons.append(
            {
                "name": feature_names[idx],
                "value": float(feature_value),
                "predicted_mean": float(predicted_mean[idx]),
                "other_mean": float(other_mean[idx]),
                "direction": direction,
                "score": weighted_gain,
            }
        )

    reasons.sort(key=lambda item: item["score"], reverse=True)
    top_reasons = []
    for reason in reasons[:top_k]:
        top_reasons.append(
            f"{reason['name']}: value={reason['value']:.4f}, "
            f"closer to {pred_label} average ({reason['predicted_mean']:.4f}) "
            f"than the opposite class average ({reason['other_mean']:.4f})"
        )

    return top_reasons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict whether a single audio file is real or fake using the baseline MFCC model."
    )
    parser.add_argument("--file", required=True, help="Path to the audio file to classify.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to the trained .joblib model.")
    parser.add_argument(
        "--metadata",
        default=str(METADATA_PATH),
        help="Path to metadata_standardized.csv used to build explanation statistics.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate for feature extraction.")
    parser.add_argument("--n-mfcc", type=int, default=20, help="Number of MFCCs to extract.")
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

    model = joblib.load(model_path)
    feature_vector = extract_features(str(audio_path), sample_rate=args.sample_rate, n_mfcc=args.n_mfcc)
    features = feature_vector.reshape(1, -1)
    real_mean, fake_mean = build_reference_statistics(
        metadata_path=metadata_path,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
    )

    pred_id = int(model.predict(features)[0])
    pred_label = "fake" if pred_id == 1 else "real"

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        real_pct = float(proba[0] * 100.0)
        fake_pct = float(proba[1] * 100.0)
    else:
        real_pct = 100.0 if pred_label == "real" else 0.0
        fake_pct = 100.0 if pred_label == "fake" else 0.0

    print("\nPrediction Result")
    print(f"File: {audio_path}")
    print(f"Predicted label: {pred_label}")
    print(f"Real probability: {real_pct:.2f}%")
    print(f"Fake probability: {fake_pct:.2f}%")
    print("\nReason for prediction:")
    for reason in explain_prediction(feature_vector, pred_label, model, real_mean, fake_mean):
        print(f"- {reason}")


if __name__ == "__main__":
    main()
