from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


PROJECT_ROOT = Path(r"D:\Final_Year_Project\Deepfake_Dissertation")
DATASET_ROOT = PROJECT_ROOT / "dataset"

MODEL_ID = "ai4bharat/indic-parler-tts"

LANGUAGES = {
    "hindi": {
        "code": "hi",
        "speakers": ["Rohit", "Divya", "Aman", "Rani"],
        "fallback": "नमस्ते, यह एक सामान्य हिंदी वाक्य है जिसे परीक्षण के लिए बनाया गया है।",
    },
    "marathi": {
        "code": "mr",
        "speakers": ["Sanjay", "Sunita", "Nikhil", "Radha", "Varun", "Isha"],
        "fallback": "नमस्कार, हे परीक्षणासाठी तयार केलेले एक सामान्य मराठी वाक्य आहे.",
    },
    "tamil": {
        "code": "ta",
        "speakers": ["Kavitha", "Jaya"],
        "fallback": "வணக்கம், இது சோதனைக்காக உருவாக்கப்பட்ட ஒரு சாதாரண தமிழ் வாக்கியம்.",
    },
    "kannada": {
        "code": "kn",
        "speakers": ["Suresh", "Anu", "Chetan", "Vidya"],
        "fallback": "ನಮಸ್ಕಾರ, ಇದು ಪರೀಕ್ಷೆಗಾಗಿ ರಚಿಸಿದ ಒಂದು ಸಾಮಾನ್ಯ ಕನ್ನಡ ವಾಕ್ಯವಾಗಿದೆ.",
    },
    "malayalam": {
        "code": "ml",
        "speakers": ["Anjali", "Anju", "Harish"],
        "fallback": "നമസ്കാരം, ഇത് പരീക്ഷണത്തിനായി സൃഷ്ടിച്ച ഒരു സാധാരണ മലയാളം വാക്യമാണ്.",
    },
}


def clean_text(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def description_for(language: str, index: int) -> str:
    info = LANGUAGES[language]
    speaker = info["speakers"][index % len(info["speakers"])]
    styles = [
        "speaks in a clear voice at a normal pace with very clear audio and no background noise",
        "speaks in a slightly expressive voice at a moderate pace with very clear audio",
        "speaks in a calm voice with balanced pitch and very clear close-up audio",
        "speaks naturally with a steady pace, clear pronunciation, and high quality recording",
    ]
    return f"{speaker} {styles[index % len(styles)]}."


def load_prompts(language: str, samples: int) -> list[str]:
    metadata_path = DATASET_ROOT / "metadata" / "metadata_real.csv"
    fallback = LANGUAGES[language]["fallback"]

    if not metadata_path.exists():
        return [fallback] * samples

    df = pd.read_csv(metadata_path)
    lang_df = df[df["language"] == language].copy()
    if "transcript" not in lang_df.columns or lang_df.empty:
        return [fallback] * samples

    prompts = [clean_text(value, fallback) for value in lang_df["transcript"].tolist()]
    prompts = [prompt for prompt in prompts if prompt]

    if not prompts:
        prompts = [fallback]

    while len(prompts) < samples:
        prompts.extend(prompts)

    return prompts[:samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake speech with Indic Parler-TTS.")
    parser.add_argument("--samples", type=int, default=500, help="Fake clips per language.")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(LANGUAGES.keys()),
        choices=list(LANGUAGES.keys()),
        help="Languages to generate.",
    )
    args = parser.parse_args()

    try:
        from parler_tts import ParlerTTSForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            "Missing parler-tts. Install it with:\n"
            "pip install git+https://github.com/huggingface/parler-tts.git"
        ) from exc

    random.seed(42)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

    rows = []

    for language in args.languages:
        code = LANGUAGES[language]["code"]
        prompts = load_prompts(language, args.samples)
        out_dir = DATASET_ROOT / "processed" / language / "fake"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating {args.samples} fake clips: {language}")

        for index, prompt in enumerate(tqdm(prompts, total=args.samples)):
            description = description_for(language, index)

            description_inputs = description_tokenizer(description, return_tensors="pt").to(device)
            prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.inference_mode():
                generation = model.generate(
                    input_ids=description_inputs.input_ids,
                    attention_mask=description_inputs.attention_mask,
                    prompt_input_ids=prompt_inputs.input_ids,
                    prompt_attention_mask=prompt_inputs.attention_mask,
                )

            audio_arr = generation.detach().cpu().numpy().squeeze().astype(np.float32)
            filename = f"{code}_fake_{index + 1:06d}.wav"
            filepath = out_dir / filename
            sf.write(filepath, audio_arr, model.config.sampling_rate)

            rows.append(
                {
                    "filepath": str(filepath).replace("\\", "/"),
                    "language": language,
                    "language_code": code,
                    "label": "fake",
                    "label_id": 1,
                    "speaker_id": "synthetic",
                    "source": "generated",
                    "generator": "Indic-Parler-TTS",
                    "duration_seconds": round(len(audio_arr) / model.config.sampling_rate, 3),
                    "project_split": "train" if language in {"hindi", "marathi"} else "test",
                    "source_split": "",
                    "transcript": prompt,
                    "original_id": Path(filepath).stem,
                }
            )

    metadata_dir = DATASET_ROOT / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    fake_metadata_path = metadata_dir / "metadata_fake.csv"
    pd.DataFrame(rows).to_csv(fake_metadata_path, index=False, encoding="utf-8")

    print("\nSaved fake metadata:")
    print(fake_metadata_path)
    print("\nNow run:")
    print("python scripts/build_metadata.py")


if __name__ == "__main__":
    main()
