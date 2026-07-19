from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path(r"D:\Final_Year_Project\Deepfake_Dissertation")
DATASET_ROOT = PROJECT_ROOT / "dataset"
METADATA_PATH = DATASET_ROOT / "metadata" / "metadata_standardized.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "baseline_mfcc"


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


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_list = []
    y_list = []

    for _, row in df.iterrows():
        features = extract_features(row["filepath"])
        x_list.append(features)
        y_list.append(int(row["label_id"]))

    return np.vstack(x_list), np.array(y_list)


def evaluate_by_language(model: RandomForestClassifier, df_test: pd.DataFrame) -> dict:
    results = {}

    for language in sorted(df_test["language"].unique()):
        lang_df = df_test[df_test["language"] == language]
        x_lang, y_lang = build_feature_matrix(lang_df)
        y_pred = model.predict(x_lang)

        results[language] = {
            "accuracy": float(accuracy_score(y_lang, y_pred)),
            "count": int(len(lang_df)),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline MFCC + RandomForest deepfake detector.")
    parser.add_argument("--metadata", type=str, default=str(METADATA_PATH))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata)

    train_df = df[df["project_split"] == "train"].reset_index(drop=True)
    test_df = df[df["project_split"] == "test"].reset_index(drop=True)

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty. Check project_split values in metadata.")

    print("Building training features...")
    x_train, y_train = build_feature_matrix(train_df)

    print("Building test features...")
    x_test, y_test = build_feature_matrix(test_df)

    print("\nTraining RandomForest...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["real", "fake"], digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print("\nTest Accuracy:")
    print(f"{accuracy:.4f}")

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(cm)

    language_results = evaluate_by_language(model, test_df)
    report_text = []
    report_text.append(f"Run time: {datetime.now().isoformat()}")
    report_text.append("")
    report_text.append(f"Metadata: {args.metadata}")
    report_text.append(f"Train samples: {len(train_df)}")
    report_text.append(f"Test samples: {len(test_df)}")
    report_text.append(f"Feature dimension: {x_train.shape[1]}")
    report_text.append("")
    report_text.append(f"Test Accuracy: {accuracy:.4f}")
    report_text.append("")
    report_text.append("Classification Report:")
    report_text.append(report)
    report_text.append("")
    report_text.append("Confusion Matrix:")
    report_text.append(str(cm))
    report_text.append("")
    report_text.append("Language-wise Accuracy:")
    for language, metrics in language_results.items():
        report_text.append(f"{language}: {metrics['accuracy']:.4f} ({metrics['count']} samples)")

    print("\nLanguage-wise Accuracy:")
    for language, metrics in language_results.items():
        print(f"{language}: {metrics['accuracy']:.4f} ({metrics['count']} samples)")
    print("\nLanguage-wise Accuracy:")
    for language, metrics in language_results.items():
        print(f"{language}: {metrics['accuracy']:.4f} ({metrics['count']} samples)")

    joblib.dump(model, OUTPUT_DIR / "random_forest_mfcc.joblib")

    metrics = {
        "test_accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "language_wise_accuracy": language_results,
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "feature_dim": int(x_train.shape[1]),
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(OUTPUT_DIR / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_text))

    print("\nSaved model:")
    print(OUTPUT_DIR / "random_forest_mfcc.joblib")

    print("\nSaved metrics:")
    print(OUTPUT_DIR / "metrics.json")


if __name__ == "__main__":
    main()
