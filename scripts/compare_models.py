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
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


PROJECT_ROOT = Path(r"D:\Final_Year_Project\Deepfake_Dissertation")
DATASET_ROOT = PROJECT_ROOT / "dataset"
METADATA_PATH = DATASET_ROOT / "metadata" / "metadata_standardized.csv"
RF_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_mfcc" / "random_forest_mfcc.joblib"
CNN_MODEL_PATH = PROJECT_ROOT / "models" / "cnn_logmel" / "cnn_logmel_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "results" / "model_comparison"


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
        features.extend(
            [
                float(np.mean(block)),
                float(np.std(block)),
                float(np.min(block)),
                float(np.max(block)),
            ]
        )

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


def evaluate_random_forest(df_test: pd.DataFrame, model_path: Path, sample_rate: int, n_mfcc: int) -> tuple[np.ndarray, np.ndarray]:
    model = joblib.load(model_path)
    x_test = np.vstack(
        [extract_mfcc_features(str(path), sample_rate=sample_rate, n_mfcc=n_mfcc) for path in df_test["filepath"]]
    )
    y_pred = model.predict(x_test)
    return df_test["label_id"].to_numpy(dtype=int), y_pred.astype(int)


def evaluate_cnn(
    df_test: pd.DataFrame,
    model_path: Path,
    sample_rate: int,
    duration_seconds: float,
    n_mels: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = load_cnn_model(model_path, sample_rate=sample_rate, duration_seconds=duration_seconds, n_mels=n_mels)

    preds = []
    labels = df_test["label_id"].to_numpy(dtype=int)
    with torch.no_grad():
        for audio_path in df_test["filepath"]:
            y = load_audio_fixed(str(audio_path), sample_rate=sample_rate, duration_seconds=duration_seconds)
            logmel = to_logmel(y, sample_rate=sample_rate, n_mels=n_mels)
            x = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).item())
            preds.append(pred)

    return labels, np.array(preds, dtype=int)


def build_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=["real", "fake"], digits=4),
    }


def build_language_metrics(df_test: pd.DataFrame, predictions: np.ndarray) -> dict:
    results = {}
    temp_df = df_test.copy()
    temp_df["prediction"] = predictions
    temp_df["correct"] = temp_df["label_id"] == temp_df["prediction"]

    for language in sorted(temp_df["language"].unique()):
        lang_df = temp_df[temp_df["language"] == language]
        results[language] = {
            "accuracy": float(lang_df["correct"].mean()),
            "count": int(len(lang_df)),
        }

    return results


def mcnemar_summary(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        chi2_stat = 0.0
        p_value_chi2 = 1.0
        p_value_exact = 1.0
    else:
        chi2_stat = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value_chi2 = float(chi2.sf(chi2_stat, df=1))
        p_value_exact = float(binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue)

    return {
        "rf_correct_cnn_wrong": b,
        "rf_wrong_cnn_correct": c,
        "continuity_corrected_chi_square": float(chi2_stat),
        "chi_square_p_value": float(p_value_chi2),
        "exact_binomial_p_value": float(p_value_exact),
        "interpretation": (
            "Higher 'rf_wrong_cnn_correct' than 'rf_correct_cnn_wrong' favors the CNN."
            if c > b
            else "Higher 'rf_correct_cnn_wrong' than 'rf_wrong_cnn_correct' favors the Random Forest."
            if b > c
            else "The paired disagreement counts are equal."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Random Forest baseline and CNN log-Mel model on the same test set.")
    parser.add_argument("--metadata", type=str, default=str(METADATA_PATH))
    parser.add_argument("--rf-model", type=str, default=str(RF_MODEL_PATH))
    parser.add_argument("--cnn-model", type=str, default=str(CNN_MODEL_PATH))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--n-mels", type=int, default=64)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata)
    df_test = df[df["project_split"] == "test"].reset_index(drop=True)

    if df_test.empty:
        raise ValueError("Test split is empty. Check project_split values in metadata.")

    print("Evaluating Random Forest baseline...")
    y_true_rf, y_pred_rf = evaluate_random_forest(
        df_test=df_test,
        model_path=Path(args.rf_model),
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
    )

    print("Evaluating CNN log-Mel model...")
    y_true_cnn, y_pred_cnn = evaluate_cnn(
        df_test=df_test,
        model_path=Path(args.cnn_model),
        sample_rate=args.sample_rate,
        duration_seconds=args.duration,
        n_mels=args.n_mels,
    )

    if not np.array_equal(y_true_rf, y_true_cnn):
        raise RuntimeError("Ground-truth labels differ between model evaluations.")

    y_true = y_true_rf
    rf_metrics = build_metrics(y_true, y_pred_rf)
    cnn_metrics = build_metrics(y_true, y_pred_cnn)
    rf_language = build_language_metrics(df_test, y_pred_rf)
    cnn_language = build_language_metrics(df_test, y_pred_cnn)
    mcnemar = mcnemar_summary(y_true, y_pred_rf, y_pred_cnn)

    comparison_rows = [
        {
            "model": "RandomForest_MFCC",
            "accuracy": rf_metrics["accuracy"],
            "precision_fake": rf_metrics["precision_fake"],
            "recall_fake": rf_metrics["recall_fake"],
            "f1_fake": rf_metrics["f1_fake"],
        },
        {
            "model": "CNN_LogMel",
            "accuracy": cnn_metrics["accuracy"],
            "precision_fake": cnn_metrics["precision_fake"],
            "recall_fake": cnn_metrics["recall_fake"],
            "f1_fake": cnn_metrics["f1_fake"],
        },
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)

    detailed = {
        "run_time": datetime.now().isoformat(),
        "test_samples": int(len(df_test)),
        "random_forest": {
            **rf_metrics,
            "language_wise_accuracy": rf_language,
        },
        "cnn_logmel": {
            **cnn_metrics,
            "language_wise_accuracy": cnn_language,
        },
        "mcnemar_summary": mcnemar,
    }

    with open(OUTPUT_DIR / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)

    report_lines = [
        f"Run time: {detailed['run_time']}",
        f"Test samples: {len(df_test)}",
        "",
        "Overall Comparison:",
        comparison_df.to_string(index=False),
        "",
        "Random Forest Classification Report:",
        rf_metrics["classification_report"],
        "",
        "CNN Classification Report:",
        cnn_metrics["classification_report"],
        "",
        "Random Forest Language-wise Accuracy:",
    ]
    for language, metrics in rf_language.items():
        report_lines.append(f"{language}: {metrics['accuracy']:.4f} ({metrics['count']} samples)")

    report_lines.extend(["", "CNN Language-wise Accuracy:"])
    for language, metrics in cnn_language.items():
        report_lines.append(f"{language}: {metrics['accuracy']:.4f} ({metrics['count']} samples)")

    report_lines.extend(
        [
            "",
            "Paired Model Comparison (McNemar-style summary):",
            f"RF correct / CNN wrong: {mcnemar['rf_correct_cnn_wrong']}",
            f"RF wrong / CNN correct: {mcnemar['rf_wrong_cnn_correct']}",
            f"Continuity-corrected chi-square: {mcnemar['continuity_corrected_chi_square']:.4f}",
            f"Chi-square p-value: {mcnemar['chi_square_p_value']:.6f}",
            f"Exact binomial p-value: {mcnemar['exact_binomial_p_value']:.6f}",
            mcnemar["interpretation"],
        ]
    )

    with open(OUTPUT_DIR / "comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\nOverall Comparison:")
    print(comparison_df.to_string(index=False))
    print("\nPaired Model Comparison:")
    print(f"RF correct / CNN wrong: {mcnemar['rf_correct_cnn_wrong']}")
    print(f"RF wrong / CNN correct: {mcnemar['rf_wrong_cnn_correct']}")
    print(f"Continuity-corrected chi-square: {mcnemar['continuity_corrected_chi_square']:.4f}")
    print(f"Chi-square p-value: {mcnemar['chi_square_p_value']:.6f}")
    print(f"Exact binomial p-value: {mcnemar['exact_binomial_p_value']:.6f}")
    print(mcnemar["interpretation"])

    print("\nSaved comparison table:")
    print(OUTPUT_DIR / "comparison_table.csv")
    print("\nSaved comparison metrics:")
    print(OUTPUT_DIR / "comparison_metrics.json")
    print("\nSaved comparison report:")
    print(OUTPUT_DIR / "comparison_report.txt")


if __name__ == "__main__":
    main()
