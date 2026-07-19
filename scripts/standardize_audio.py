from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(r"D:\Final_Year_Project\Deepfake_Dissertation")
DATASET_ROOT = PROJECT_ROOT / "dataset"
STANDARDIZED_ROOT = DATASET_ROOT / "standardized"
INPUT_METADATA = DATASET_ROOT / "metadata" / "metadata_combined.csv"
OUTPUT_METADATA = DATASET_ROOT / "metadata" / "metadata_standardized.csv"


def convert_to_wav(input_path: Path, output_path: Path, sample_rate: int = 16000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(output_path),
    ]

    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize all audio to mono 16kHz WAV.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    df = pd.read_csv(INPUT_METADATA)
    standardized_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        input_path = Path(row["filepath"])
        language = row["language"]
        label = row["label"]
        filename_stem = input_path.stem
        output_path = STANDARDIZED_ROOT / language / label / f"{filename_stem}.wav"

        convert_to_wav(input_path, output_path, sample_rate=args.sample_rate)

        new_row = row.copy()
        new_row["filepath"] = str(output_path).replace("\\", "/")
        standardized_rows.append(new_row)

    out_df = pd.DataFrame(standardized_rows)
    out_df.to_csv(OUTPUT_METADATA, index=False, encoding="utf-8")

    print("\nSaved standardized metadata:")
    print(OUTPUT_METADATA)

    print("\nCounts by label:")
    print(out_df["label"].value_counts())

    print("\nCounts by language and label:")
    print(out_df.groupby(["language", "label"]).size())


if __name__ == "__main__":
    main()
