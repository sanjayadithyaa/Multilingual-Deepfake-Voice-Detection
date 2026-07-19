from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import edge_tts
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(r"D:\Final_Year_Project\Deepfake_Dissertation")
DATASET_ROOT = PROJECT_ROOT / "dataset"

VOICES = {
    "hindi": "hi-IN-MadhurNeural",
    "marathi": "mr-IN-ManoharNeural",
    "tamil": "ta-IN-ValluvarNeural",
    "kannada": "kn-IN-GaganNeural",
    "malayalam": "ml-IN-MidhunNeural",
}


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return ""
    return text.replace("\n", " ").replace("\r", " ")


async def synthesize_one(text: str, voice: str, output_path: Path) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(output_path))


async def generate_fake_clips(samples_per_language: int | None = None) -> None:
    metadata_path = DATASET_ROOT / "metadata" / "metadata_real.csv"
    real_df = pd.read_csv(metadata_path)

    fake_rows = []

    for language, voice in VOICES.items():
        lang_df = real_df[real_df["language"] == language].copy()

        if samples_per_language is not None:
            lang_df = lang_df.head(samples_per_language)

        output_dir = DATASET_ROOT / "processed" / language / "fake"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating fake clips: {language} ({len(lang_df)} clips, voice={voice})")

        for fake_index, (_, row) in enumerate(tqdm(lang_df.iterrows(), total=len(lang_df)), start=1):
            text = clean_text(row.get("transcript", ""))

            if not text:
                text = f"This is a synthetic speech sample for {language}."

            language_code = str(row.get("language_code", language[:2]))
            filename = f"{language_code}_fake_{fake_index:06d}.mp3"
            filepath = output_dir / filename

            if not filepath.exists():
                await synthesize_one(text, voice, filepath)

            fake_rows.append(
                {
                    "filepath": str(filepath).replace("\\", "/"),
                    "language": language,
                    "language_code": language_code,
                    "label": "fake",
                    "label_id": 1,
                    "speaker_id": "edge_tts",
                    "source": "Edge-TTS",
                    "generator": voice,
                    "duration_seconds": "",
                    "project_split": row.get("project_split", ""),
                    "source_split": row.get("source_split", ""),
                    "transcript": text,
                    "original_id": row.get("original_id", f"{language}_{fake_index:06d}"),
                }
            )

    fake_metadata_path = DATASET_ROOT / "metadata" / "metadata_fake.csv"
    fake_df = pd.DataFrame(fake_rows)
    fake_df.to_csv(fake_metadata_path, index=False, encoding="utf-8")

    combined_path = DATASET_ROOT / "metadata" / "metadata_combined.csv"
    real_df = real_df.copy()
    combined_df = pd.concat([real_df, fake_df], ignore_index=True)
    combined_df.to_csv(combined_path, index=False, encoding="utf-8")

    print("\nSaved fake metadata:")
    print(fake_metadata_path)

    print("\nSaved combined metadata:")
    print(combined_path)

    print("\nCounts by label:")
    print(combined_df["label"].value_counts())

    print("\nCounts by language and label:")
    print(combined_df.groupby(["language", "label"]).size())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake speech clips using Edge TTS.")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Optional number of fake clips per language. Default: match metadata_real.csv.",
    )
    args = parser.parse_args()

    asyncio.run(generate_fake_clips(samples_per_language=args.samples))


if __name__ == "__main__":
    main()
