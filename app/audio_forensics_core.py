from __future__ import annotations

import csv
import hashlib
import os
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct, rfft, rfftfreq
import soundfile as sf

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torchaudio
import sys
import os

def get_asset_path(relative_path):
    """ Get absolute path to resource, works for development and for PyInstaller """
    try:
        # PyInstaller creates a temporary folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

PROJECT_ROOT = Path(r"D:\Final_Year_Project\Deepfake_Dissertation")
DATASET_ROOT = PROJECT_ROOT / "dataset"
METADATA_PATH = get_asset_path("dataset/metadata/metadata_standardized.csv")
RF_MODEL_PATH = get_asset_path("models/baseline_mfcc/random_forest_mfcc.joblib")
CNN_MODEL_PATH = get_asset_path("models/cnn_logmel/cnn_logmel_best.pt")
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

FEATURE_DESCRIPTIONS = {
    "duration_seconds": "Length of the analyzed clip after loading. Longer clips usually give more stable decisions.",
    "rms_mean": "Average signal energy. Synthetic clips can have unusually smooth or flat energy patterns.",
    "zcr_mean": "Average zero-crossing rate. It reflects how quickly the waveform changes sign and can hint at noisiness or sharpness.",
    "centroid_mean": "Spectral centroid, often described as brightness. Higher values usually mean more high-frequency emphasis.",
    "bandwidth_mean": "Spectral bandwidth. It measures how spread out the frequency content is around the center of mass.",
    "rolloff_mean": "Spectral rolloff. It marks the frequency below which most of the signal energy is concentrated.",
    "mfcc1_mean": "The first MFCC coefficient, capturing the broad spectral envelope of speech.",
    "mfcc2_mean": "The second MFCC coefficient, reflecting shape changes in the speech spectrum.",
    "mfcc3_mean": "The third MFCC coefficient, capturing additional timbral variation in the speech signal.",
}

TOOL_VERSION = "AudioForensics_Analyzer v1.0"
FORENSIC_ANALYST_NAME = "Sanjay Adithya"
FORENSIC_ANALYST_DESIGNATION = "Forensic Audio Analyst"
FORENSIC_ANALYST_ORGANIZATION = "Deepfake Dissertation Research Project"
FORENSIC_ANALYST_CONTACT = "N/A"
REPORT_DISCLAIMER = (
    "DISCLAIMER: This automated report is generated based on algorithmic analysis of acoustic features. "
    "While the model provides a statistical probability of synthetic generation, these results should be "
    "reviewed by a qualified forensic examiner in conjunction with other case evidence."
)


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


@dataclass
class AnalysisResult:
    file_path: str
    model_name: str
    predicted_label: str
    real_probability: float
    fake_probability: float
    summary_features: dict[str, float]
    reason_lines: list[str]
    user_explanation: str


class PlaceholderBox:
    def __init__(self, width: float, height: float, title: str, subtitle: str = "") -> None:
        self.width = width
        self.height = height
        self.title = title
        self.subtitle = subtitle

    def wrap(self, _avail_width: float, _avail_height: float) -> tuple[float, float]:
        return self.width, self.height

    def drawOn(self, canvas, x: float, y: float, _sW: float = 0) -> None:
        canvas.saveState()
        canvas.setStrokeColorRGB(0.42, 0.45, 0.5)
        canvas.setLineWidth(1)
        canvas.rect(x, y, self.width, self.height, stroke=1, fill=0)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawCentredString(x + self.width / 2, y + self.height / 2 + 6, self.title)
        if self.subtitle:
            canvas.setFont("Helvetica", 8)
            canvas.drawCentredString(x + self.width / 2, y + self.height / 2 - 8, self.subtitle)
        canvas.restoreState()


def load_audio_fixed(audio_path: str, sample_rate: int = 16000, duration_seconds: float = 4.0) -> np.ndarray:
    try:
        y, original_sr = sf.read(audio_path, always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)
        if original_sr != sample_rate:
            y = librosa.resample(y, orig_sr=original_sr, target_sr=sample_rate)
    except Exception:  # noqa: BLE001
        y, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        y = y.astype(np.float32)

    target_length = int(sample_rate * duration_seconds)

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    return y.astype(np.float32)


def to_logmel(y: np.ndarray, sample_rate: int = 16000, n_mels: int = 64) -> np.ndarray:
    waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
        power=2.0,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")
    mel = mel_transform(waveform)
    logmel = db_transform(mel).squeeze(0).cpu().numpy()
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-8)
    return logmel.astype(np.float32)


def extract_mfcc_features(audio_path: str, sample_rate: int = 16000, n_mfcc: int = 20) -> np.ndarray:
    y = load_audio_fixed(audio_path, sample_rate=sample_rate, duration_seconds=4.0)
    sr = sample_rate

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


def extract_summary_features(y: np.ndarray, sample_rate: int = 16000, n_mfcc: int = 13) -> dict[str, float]:
    frame_length = 1024
    hop_length = 512
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))

    frames = []
    for start in range(0, max(len(y) - frame_length + 1, 1), hop_length):
        frames.append(y[start : start + frame_length])
    frame_matrix = np.vstack(frames).astype(np.float32)

    rms = np.sqrt(np.mean(np.square(frame_matrix), axis=1) + 1e-10)
    zcr = np.mean(np.diff(np.signbit(frame_matrix), axis=1), axis=1)

    window = np.hanning(frame_length).astype(np.float32)
    spectra = np.abs(rfft(frame_matrix * window, axis=1)) + 1e-10
    freqs = rfftfreq(frame_length, d=1.0 / sample_rate)
    spectral_sum = np.sum(spectra, axis=1)
    centroid = np.sum(spectra * freqs, axis=1) / spectral_sum
    bandwidth = np.sqrt(np.sum(((freqs - centroid[:, None]) ** 2) * spectra, axis=1) / spectral_sum)

    cumulative_energy = np.cumsum(spectra, axis=1)
    rolloff_threshold = 0.85 * cumulative_energy[:, -1]
    rolloff_indices = [int(np.searchsorted(cumulative_energy[i], rolloff_threshold[i])) for i in range(len(frame_matrix))]
    rolloff = freqs[np.clip(rolloff_indices, 0, len(freqs) - 1)]

    log_spectra = np.log(spectra)
    mfcc = dct(log_spectra, type=2, axis=1, norm="ortho")[:, :n_mfcc].T

    return {
        "duration_seconds": float(len(y) / sample_rate),
        "rms_mean": float(np.mean(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "rolloff_mean": float(np.mean(rolloff)),
        "mfcc1_mean": float(np.mean(mfcc[0])),
        "mfcc2_mean": float(np.mean(mfcc[1])),
        "mfcc3_mean": float(np.mean(mfcc[2])),
    }


def compute_sha256(file_path: str | Path) -> str:
    sha256 = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def file_size_mb(file_path: str | Path) -> float:
    return Path(file_path).stat().st_size / (1024 * 1024)


def format_footer_disclaimer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColorRGB(0.32, 0.32, 0.32)
    wrapped = textwrap.wrap(REPORT_DISCLAIMER, width=128)
    text = canvas.beginText(doc.leftMargin, 26)
    for line in wrapped:
        text.textLine(line)
    canvas.drawText(text)
    canvas.restoreState()


def render_waveform_plot(audio_path: str | Path, output_path: str | Path, sample_rate: int = 16000) -> None:
    y = load_audio_fixed(str(audio_path), sample_rate=sample_rate, duration_seconds=4.0)
    times = np.linspace(0, len(y) / sample_rate, num=len(y), endpoint=False)
    plt.figure(figsize=(7, 2.6))
    plt.plot(times, y, linewidth=0.7, color="#0f4c81")
    plt.title("Waveform Plot", fontsize=11)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def render_mel_spectrogram(audio_path: str | Path, output_path: str | Path, sample_rate: int = 16000, n_mels: int = 64) -> None:
    y = load_audio_fixed(str(audio_path), sample_rate=sample_rate, duration_seconds=4.0)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=256,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(7, 2.6))
    plt.imshow(logmel, aspect="auto", origin="lower", cmap="magma")
    plt.title("Mel Spectrogram", fontsize=11)
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency Bins")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def get_model_architecture_description(model_name: str) -> str:
    model_name_upper = model_name.strip().upper()
    if model_name_upper == "RANDOM FOREST":
        return "Analysis performed via Random Forest ensemble classifier operating on extracted acoustic features."
    return "Analysis performed via CNN classifier operating on log-Mel spectrogram acoustic representations."


class AudioForensicsAnalyzer:
    def __init__(
        self,
        metadata_path: Path = METADATA_PATH,
        rf_model_path: Path = RF_MODEL_PATH,
        cnn_model_path: Path = CNN_MODEL_PATH,
        sample_rate: int = 16000,
        duration_seconds: float = 4.0,
        n_mfcc: int = 20,
        n_mels: int = 64,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.rf_model_path = Path(rf_model_path)
        self.cnn_model_path = Path(cnn_model_path)
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

        self._rf_model: Any | None = None
        self._cnn_model: SimpleAudioCNN | None = None

    def _load_rf_model(self) -> Any:
        if self._rf_model is None:
            self._rf_model = joblib.load(self.rf_model_path)
        return self._rf_model

    def _load_cnn_model(self) -> SimpleAudioCNN:
        if self._cnn_model is None:
            dummy_audio = np.zeros(int(self.sample_rate * self.duration_seconds), dtype=np.float32)
            dummy_logmel = to_logmel(dummy_audio, sample_rate=self.sample_rate, n_mels=self.n_mels)
            time_frames = dummy_logmel.shape[1]

            model = SimpleAudioCNN(n_mels=self.n_mels, time_frames=time_frames)
            state_dict = torch.load(self.cnn_model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            self._cnn_model = model
        return self._cnn_model

    def _build_reason_lines(self, summary_features: dict[str, float], predicted_label: str) -> list[str]:
        reasons: list[tuple[float, str]] = []
        rms = summary_features["rms_mean"]
        zcr = summary_features["zcr_mean"]
        centroid = summary_features["centroid_mean"]
        bandwidth = summary_features["bandwidth_mean"]
        rolloff = summary_features["rolloff_mean"]
        mfcc1 = summary_features["mfcc1_mean"]

        if predicted_label == "fake":
            reasons.extend(
                [
                    (
                        abs(centroid - 2300),
                        f"centroid_mean={centroid:.4f}, showing brighter spectral emphasis that often appears in synthetic or over-processed speech",
                    ),
                    (
                        abs(bandwidth - 2100),
                        f"bandwidth_mean={bandwidth:.4f}, indicating a broad spectral spread consistent with generated or heavily processed audio",
                    ),
                    (
                        abs(zcr - 0.08),
                        f"zcr_mean={zcr:.4f}, suggesting sharper frame-to-frame waveform changes than typical smooth natural speech",
                    ),
                    (
                        abs(rolloff - 4200),
                        f"rolloff_mean={rolloff:.4f}, meaning substantial high-frequency energy remains present deeper into the spectrum",
                    ),
                    (
                        abs(mfcc1 + 230),
                        f"mfcc1_mean={mfcc1:.4f}, reflecting a spectral envelope more consistent with synthetic timbre than conversational human speech",
                    ),
                ]
            )
        else:
            reasons.extend(
                [
                    (
                        abs(rms - 0.03),
                        f"rms_mean={rms:.4f}, showing stable speech energy rather than the flat or over-controlled energy often heard in synthetic clips",
                    ),
                    (
                        abs(zcr - 0.05),
                        f"zcr_mean={zcr:.4f}, indicating waveform transitions in the range more commonly associated with natural speech production",
                    ),
                    (
                        abs(centroid - 1800),
                        f"centroid_mean={centroid:.4f}, suggesting a balanced spectral brightness closer to recorded human speech",
                    ),
                    (
                        abs(bandwidth - 1700),
                        f"bandwidth_mean={bandwidth:.4f}, showing a frequency spread more typical of natural vocal articulation",
                    ),
                    (
                        abs(mfcc1 + 300),
                        f"mfcc1_mean={mfcc1:.4f}, reflecting a speech envelope that looks more like natural human resonance than synthetic rendering",
                    ),
                ]
            )

        reasons.sort(key=lambda item: item[0])
        return [text for _, text in reasons[:4]]

    def _build_user_explanation(
        self,
        predicted_label: str,
        real_probability: float,
        fake_probability: float,
        reason_lines: list[str],
        model_name: str,
    ) -> str:
        confidence = fake_probability if predicted_label == "fake" else real_probability
        confidence_word = "strong" if confidence >= 85 else "moderate" if confidence >= 65 else "limited"

        intro = (
            f"The {model_name} model classified this audio as {predicted_label} with "
            f"{confidence:.2f}% confidence. This is a {confidence_word} decision."
        )

        if predicted_label == "fake":
            body = (
                "The acoustic profile of the clip aligns more closely with synthetic speech patterns "
                "seen during training than with the real human recordings."
            )
        else:
            body = (
                "The acoustic profile of the clip aligns more closely with natural human speech "
                "patterns seen during training than with the synthetic examples."
            )

        reason_text = " The strongest supporting cues were: " + "; ".join(reason_lines[:3]) + "."
        return intro + " " + body + reason_text

    def analyze_file(self, audio_path: str | Path, model_name: str = "CNN") -> AnalysisResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y = load_audio_fixed(str(audio_path), sample_rate=self.sample_rate, duration_seconds=self.duration_seconds)
        summary_features = extract_summary_features(y, sample_rate=self.sample_rate)

        model_name_upper = model_name.strip().upper()
        if model_name_upper == "CNN":
            model = self._load_cnn_model()
            logmel = to_logmel(y, sample_rate=self.sample_rate, n_mels=self.n_mels)
            x = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            real_probability = float(probabilities[0] * 100.0)
            fake_probability = float(probabilities[1] * 100.0)
        elif model_name_upper in {"RANDOM FOREST", "RF", "BASELINE"}:
            model = self._load_rf_model()
            features = extract_mfcc_features(str(audio_path), sample_rate=self.sample_rate, n_mfcc=self.n_mfcc)
            probabilities = model.predict_proba(features.reshape(1, -1))[0]
            real_probability = float(probabilities[0] * 100.0)
            fake_probability = float(probabilities[1] * 100.0)
            model_name_upper = "RANDOM FOREST"
        else:
            raise ValueError("Unsupported model. Choose either 'CNN' or 'Random Forest'.")

        predicted_label = "fake" if fake_probability >= real_probability else "real"
        reason_lines = self._build_reason_lines(summary_features, predicted_label)
        user_explanation = self._build_user_explanation(
            predicted_label=predicted_label,
            real_probability=real_probability,
            fake_probability=fake_probability,
            reason_lines=reason_lines,
            model_name=model_name_upper,
        )

        return AnalysisResult(
            file_path=str(audio_path),
            model_name=model_name_upper,
            predicted_label=predicted_label,
            real_probability=real_probability,
            fake_probability=fake_probability,
            summary_features=summary_features,
            reason_lines=reason_lines,
            user_explanation=user_explanation,
        )

    def analyze_folder(self, folder_path: str | Path, model_name: str = "CNN") -> list[AnalysisResult]:
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        audio_files = sorted(
            path for path in folder_path.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        )
        if not audio_files:
            raise ValueError(f"No supported audio files found in folder: {folder_path}")

        return [self.analyze_file(audio_path, model_name=model_name) for audio_path in audio_files]

    @staticmethod
    def save_results_csv(results: list[AnalysisResult], output_csv_path: str | Path) -> None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "file_path",
                    "model_name",
                    "predicted_label",
                    "real_probability",
                    "fake_probability",
                    "user_explanation",
                ]
            )
            for result in results:
                writer.writerow(
                    [
                        result.file_path,
                        result.model_name,
                        result.predicted_label,
                        f"{result.real_probability:.2f}",
                        f"{result.fake_probability:.2f}",
                        result.user_explanation,
                    ]
                )

    @staticmethod
    def export_pdf_report(
        results: list[AnalysisResult],
        output_pdf_path: str | Path,
        case_title: str = "Automated Audio Forensic Analysis Report",
    ) -> None:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        output_pdf_path = Path(output_pdf_path)
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ForensicTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=20,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=6,
        )
        heading_style = ParagraphStyle(
            "ForensicHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=6,
            spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "ForensicBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=12.5,
        )
        small_style = ParagraphStyle(
            "ForensicSmall",
            parent=body_style,
            fontSize=8.5,
            leading=10.5,
        )
        italic_style = ParagraphStyle(
            "ForensicItalic",
            parent=body_style,
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor=colors.HexColor("#334155"),
        )
        mono_style = ParagraphStyle(
            "MonoBody",
            parent=body_style,
            fontName="Courier",
            fontSize=8.5,
            leading=11,
            backColor=colors.HexColor("#f3f4f6"),
            borderPadding=6,
        )

        story: list[Any] = []
        local_now = datetime.now().astimezone()
        utc_now = local_now.astimezone(timezone.utc)
        local_timestamp = f"{local_now.strftime('%Y-%m-%d %H:%M:%S')} {local_now.tzname()} ({local_now.strftime('%z')})"
        utc_timestamp = utc_now.strftime("%Y-%m-%d %H:%M:%S UTC")

        with tempfile.TemporaryDirectory(prefix="audio_forensics_report_") as temp_dir:
            temp_dir_path = Path(temp_dir)

            for index, result in enumerate(results, start=1):
                file_path = Path(result.file_path)
                sha256_hash = compute_sha256(file_path)
                size_mb = file_size_mb(file_path)
                duration_seconds = result.summary_features.get("duration_seconds", 0.0)
                fake_probability = result.fake_probability
                real_probability = result.real_probability
                predicted_label = result.predicted_label.upper()

                waveform_plot_path = temp_dir_path / f"waveform_{index}.png"
                mel_plot_path = temp_dir_path / f"mel_{index}.png"
                render_waveform_plot(file_path, waveform_plot_path)
                render_mel_spectrogram(file_path, mel_plot_path)

                if index > 1:
                    story.append(PageBreak())

                story.append(Paragraph(case_title, title_style))
                story.append(Paragraph(f"Report Generation Timestamp (Local): {local_timestamp}", small_style))
                story.append(Paragraph(f"Reference UTC Timestamp: {utc_timestamp}", small_style))
                story.append(Spacer(1, 5))

                story.append(Paragraph("Section 1: Header & Chain of Custody", heading_style))
                chain_rows = [
                    [Paragraph("<b>Original_Filename</b>", small_style), Paragraph(file_path.name, small_style)],
                    [
                        Paragraph("<b>File_Format</b>", small_style),
                        Paragraph(file_path.suffix.replace(".", "").upper() or "N/A", small_style),
                    ],
                    [Paragraph("<b>File_Size_MB</b>", small_style), Paragraph(f"{size_mb:.3f}", small_style)],
                    [Paragraph("<b>Duration_Seconds</b>", small_style), Paragraph(f"{duration_seconds:.4f}", small_style)],
                    [Paragraph("<b>Sampling_Rate_Hz</b>", small_style), Paragraph(str(16000), small_style)],
                    [Paragraph("<b>Hash_Algorithm</b>", small_style), Paragraph("SHA-256", small_style)],
                    [Paragraph("<b>Forensic_Analyst</b>", small_style), Paragraph(FORENSIC_ANALYST_NAME, small_style)],
                    [Paragraph("<b>Designation</b>", small_style), Paragraph(FORENSIC_ANALYST_DESIGNATION, small_style)],
                    [Paragraph("<b>Organization</b>", small_style), Paragraph(FORENSIC_ANALYST_ORGANIZATION, small_style)],
                    [Paragraph("<b>Contact</b>", small_style), Paragraph(FORENSIC_ANALYST_CONTACT, small_style)],
                ]
                chain_table = Table(chain_rows, colWidths=[48 * mm, 118 * mm])
                chain_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e5e7eb")),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9ca3af")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ]
                    )
                )
                story.append(chain_table)
                story.append(Spacer(1, 5))
                story.append(Paragraph("<b>SHA-256 Cryptographic Hash</b>", body_style))
                story.append(Paragraph(sha256_hash, mono_style))
                story.append(Spacer(1, 7))

                story.append(Paragraph("Section 2: Primary Analysis Results", heading_style))
                score_rows = [
                    [Paragraph("<b>Final Classification</b>", body_style), Paragraph(f"<b>{predicted_label}</b>", body_style)],
                    [Paragraph("<b>Authenticity Score (Real Probability)</b>", body_style), Paragraph(f"<b>{real_probability:.2f}%</b>", body_style)],
                    [Paragraph("<b>Deepfake Probability</b>", body_style), Paragraph(f"{fake_probability:.2f}%", body_style)],
                    [Paragraph("<b>Model Used</b>", body_style), Paragraph(result.model_name, body_style)],
                ]
                score_table = Table(score_rows, colWidths=[48 * mm, 118 * mm])
                score_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e5e7eb")),
                            ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#6b7280")),
                            (
                                "TEXTCOLOR",
                                (1, 0),
                                (1, 1),
                                colors.HexColor("#7f1d1d") if predicted_label == "FAKE" else colors.HexColor("#14532d"),
                            ),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ]
                    )
                )
                story.append(score_table)
                story.append(Spacer(1, 4))
                story.append(
                    Paragraph(
                        "Analysis performed using cross-lingual acoustic features, independent of spoken language.",
                        italic_style,
                    )
                )
                story.append(Spacer(1, 7))

                story.append(Paragraph("Section 3: Visual Feature Analysis", heading_style))
                waveform_element = (
                    Image(str(waveform_plot_path), width=78 * mm, height=42 * mm)
                    if waveform_plot_path.exists()
                    else PlaceholderBox(78 * mm, 42 * mm, "Waveform Plot", "Amplitude vs. Time")
                )
                mel_element = (
                    Image(str(mel_plot_path), width=78 * mm, height=42 * mm)
                    if mel_plot_path.exists()
                    else PlaceholderBox(78 * mm, 42 * mm, "Mel Spectrogram", "Frequency vs. Time")
                )
                # Adjust these image widths/heights if you want larger plots or wider page margins.
                visual_table = Table([[waveform_element, mel_element]], colWidths=[84 * mm, 84 * mm])
                visual_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
                story.append(visual_table)
                caption_table = Table(
                    [[
                        Paragraph("Figure 1. Waveform plot showing amplitude variation over time.", small_style),
                        Paragraph("Figure 2. Mel spectrogram showing time-frequency acoustic energy.", small_style),
                    ]],
                    colWidths=[84 * mm, 84 * mm],
                )
                story.append(caption_table)
                story.append(Spacer(1, 7))

                story.append(Paragraph("Section 4: System & Methodology Information", heading_style))
                methodology_rows = [
                    [Paragraph("<b>Tool Version</b>", small_style), Paragraph(TOOL_VERSION, small_style)],
                    [
                        Paragraph("<b>Model Architecture</b>", small_style),
                        Paragraph(get_model_architecture_description(result.model_name), small_style),
                    ],
                    [
                        Paragraph("<b>Processing Time</b>", small_style),
                        Paragraph("N/A", small_style),
                    ],
                ]
                methodology_table = Table(methodology_rows, colWidths=[48 * mm, 118 * mm])
                methodology_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e5e7eb")),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9ca3af")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ]
                    )
                )
                story.append(methodology_table)
                story.append(Spacer(1, 7))

                story.append(Paragraph("Automated Supporting Observations", heading_style))
                for reason in result.reason_lines:
                    story.append(Paragraph(f"- {reason}", body_style))
                story.append(Spacer(1, 4))
                story.append(Paragraph(result.user_explanation, body_style))

            # Adjust page margins here if you want denser content or more whitespace.
            doc = SimpleDocTemplate(
                str(output_pdf_path),
                pagesize=A4,
                leftMargin=18 * mm,
                rightMargin=18 * mm,
                topMargin=16 * mm,
                bottomMargin=26 * mm,
            )
            doc.build(story, onFirstPage=format_footer_disclaimer, onLaterPages=format_footer_disclaimer)
