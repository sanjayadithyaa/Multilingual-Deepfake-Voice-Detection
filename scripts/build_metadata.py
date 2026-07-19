from __future__ import annotations

from pathlib import Path

import librosa
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(r"D:\Final_Year_Project\Deepfake_Dissertation")
DATASET_ROOT = PROJECT_ROOT / "dataset"

LANGUAGES = {
    "hindi": "hi",
    "marathi": "mr",
    "tamil": "ta",
    "kannada": "kn",
    "malayalam": "ml",
}

LABELS = {
    "real": 0,
    "fake": 1,
}


def project_split_for(language: str, file_index: int, total_files: int) -> str:
    if language in {"hindi", "marathi"}:
        validation_start = int(total_files * 0.85)
        return "validation" if file_index >= validation_start else "train"
    return "test"


def duration_seconds(path: Path) -> float:
    duration = librosa.get_duration(path=str(path))
    return round(float(duration), 3)


def main() -> None:
    rows = []

    for language, code in LANGUAGES.items():
        for label, label_id in LABELS.items():
            folder = DATASET_ROOT / "processed" / language / label
            folder.mkdir(parents=True, exist_ok=True)

            files = sorted(folder.glob("*.wav"))
            print(f"{language}/{label}: {len(files)} files")

            for index, path in enumerate(tqdm(files)):
                rows.append(
                    {
                        "filepath": str(path).replace("\\", "/"),
                        "language": language,
                        "language_code": code,
                        "label": label,
                        "label_id": label_id,
                        "speaker_id": "unknown" if label == "fake" else "from_source_metadata",
                        "source": "IndicVoices-R" if label == "real" else "generated",
                        "generator": "none" if label == "real" else "to_be_filled",
                        "duration_seconds": duration_seconds(path),
                        "project_split": project_split_for(language, index, len(files)),
                        "source_split": "",
                        "transcript": "",
                        "original_id": path.stem,
                    }
                )

    metadata_dir = DATASET_ROOT / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    output_path = metadata_dir / "metadata_full.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    print("\nSaved:")
    print(output_path)

    if not df.empty:
        print("\nCounts:")
        print(df.groupby(["language", "label"]).size())
        print("\nProject splits:")
        print(df.groupby(["project_split", "language", "label"]).size())


if __name__ == "__main__":
    main()
