from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm


LANGUAGES = {
    "hindi": "hi",
    "marathi": "mr",
    "tamil": "ta",
    "kannada": "kn",
    "malayalam": "ml",
}

SOURCE_DATASET = "ai4bharat/indicvoices_r"
PROJECT_ROOT = Path(r"D:\Final_Year_Project\Deepfake_Dissertation")
DATASET_ROOT = PROJECT_ROOT / "dataset"


def get_first_present(row: dict[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def load_language_dataset(language: str, split: str):
    candidates = [language, language.title()]
    last_error: Exception | None = None

    for candidate in candidates:
        try:
            ds = load_dataset(
                SOURCE_DATASET,
                candidate,
                split=split,
                streaming=True,
            )
            return ds.cast_column("audio", Audio(decode=False)), candidate
        except Exception as exc:  # noqa: BLE001 - show useful fallback error.
            last_error = exc

    raise RuntimeError(
        f"Could not load language config for {language!r} with split {split!r}. "
        f"Tried {candidates}. Last error: {last_error}"
    )

def project_split_for(language: str, index: int, samples: int) -> str:
    if language in {"hindi", "marathi"}:
        return "train"
    return "test"


def save_real_clips(language: str, code: str, samples: int, source_split: str) -> list[dict[str, Any]]:
    ds, hf_config = load_language_dataset(language, source_split)

    output_dir = DATASET_ROOT / "processed" / language / "real"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    print(f"\nDownloading {samples} real clips: {language} ({hf_config}/{source_split})")

    for index, item in enumerate(tqdm(ds, total=samples)):
        if index >= samples:
            break

        if "audio" not in item:
            raise KeyError(
                f"No 'audio' column found for {language}. Available columns: {list(item.keys())}"
            )

        audio = item["audio"]

        filename = f"{code}_real_{index + 1:06d}.wav"
        filepath = output_dir / filename

        audio_bytes = audio.get("bytes")
        audio_path = audio.get("path")

        if audio_bytes is not None:
            filepath.write_bytes(audio_bytes)
        elif audio_path is not None:
            source_path = Path(audio_path)
            filepath.write_bytes(source_path.read_bytes())
        else:
            raise ValueError(f"No audio bytes/path found. Audio value: {audio}")

        duration_seconds = ""


        rows.append(
            {
                "filepath": str(filepath).replace("\\", "/"),
                "language": language,
                "language_code": code,
                "label": "real",
                "label_id": 0,
                "speaker_id": get_first_present(
                    item,
                    ["speaker_id", "speaker", "speaker_name", "speakerId", "client_id"],
                    default="unknown",
                ),
                "source": "IndicVoices-R",
                "generator": "none",
                "duration_seconds": duration_seconds,
                "project_split": project_split_for(language, index, samples),
                "source_split": source_split,
                "transcript": get_first_present(
                    item,
                    ["text", "transcript", "sentence", "normalized_text", "raw_text"],
                    default="",
                ),
                "original_id": get_first_present(
                    item,
                    ["id", "utt_id", "audio_id", "file_id", "path"],
                    default=f"{language}_{index + 1:06d}",
                ),
            }
        )

    return rows


def create_empty_dirs() -> None:
    for language in LANGUAGES:
        for label in ["real", "fake"]:
            (DATASET_ROOT / "processed" / language / label).mkdir(parents=True, exist_ok=True)

    for subdir in ["raw/indicvoices_r", "metadata", "logs"]:
        (DATASET_ROOT / subdir).mkdir(parents=True, exist_ok=True)

    for subdir in [
        "notebooks",
        "scripts",
        "reports/figures",
        "reports/tables",
        "reports/dissertation_draft",
        "models/checkpoints",
        "models/saved_models",
        "results/zero_shot",
        "results/few_shot",
    ]:
        (PROJECT_ROOT / subdir).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download real speech clips from ai4bharat/indicvoices_r."
    )
    parser.add_argument("--samples", type=int, default=500, help="Number of clips per language.")
    parser.add_argument("--split", default="test", help="Hugging Face split to stream.")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(LANGUAGES.keys()),
        choices=list(LANGUAGES.keys()),
        help="Languages to download.",
    )
    args = parser.parse_args()

    create_empty_dirs()

    all_rows: list[dict[str, Any]] = []
    for language in args.languages:
        all_rows.extend(save_real_clips(language, LANGUAGES[language], args.samples, args.split))

    metadata_path = DATASET_ROOT / "metadata" / "metadata_real.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(metadata_path, index=False, encoding="utf-8")

    print("\nSaved metadata:")
    print(metadata_path)
    print("\nCounts by language:")
    print(df["language"].value_counts())
    print("\nCounts by project split:")
    print(df["project_split"].value_counts())


if __name__ == "__main__":
    main()
