# Dataset Preparation Guide

Project: Cross-Lingual Transfer Learning for Deepfake Detection Across Dravidian and Indo-Aryan Languages

This guide prepares the **real speech** part of the dataset using `ai4bharat/indicvoices_r`.

The first target is:

| Language | Real Clips |
|---|---:|
| Hindi | 500 |
| Marathi | 500 |
| Tamil | 500 |
| Kannada | 500 |
| Malayalam | 500 |

Later, generate or collect 500 fake clips for each language and place them in the matching `fake/` folders.

## 1. Final Directory Structure

Create/keep the dissertation project here:

```text
D:\Final_Year_Project\Deepfake_Dissertation
```

Use this structure:

```text
D:\Final_Year_Project\Deepfake_Dissertation\
  dataset/
    raw/
      indicvoices_r/
    processed/
      hindi/
        real/
        fake/
      marathi/
        real/
        fake/
      tamil/
        real/
        fake/
      kannada/
        real/
        fake/
      malayalam/
        real/
        fake/
    metadata/
      metadata_real.csv
      metadata_full.csv
    logs/

  notebooks/
    01_dataset_check.ipynb
    02_baseline_mfcc_svm.ipynb
    03_wav2vec2_experiment.ipynb

  scripts/
    download_real_indicvoices_r.py
    build_metadata.py

  reports/
    figures/
    tables/
    dissertation_draft/

  models/
    checkpoints/
    saved_models/

  results/
    zero_shot/
    few_shot/
```

## 2. Install Requirements

Open PowerShell in this folder and run:

```powershell
pip install -r requirements.txt
```

Then log in to Hugging Face:

```powershell
huggingface-cli login
```

Paste your Hugging Face token.

You must also open the dataset page in your browser and accept access:

```text
https://huggingface.co/datasets/ai4bharat/indicvoices_r
```

## 3. Download 500 Real Clips Per Language

Run:

```powershell
python scripts/download_real_indicvoices_r.py --samples 500 --split valid
```

If `valid` does not work for your Hugging Face access, try:

```powershell
python scripts/download_real_indicvoices_r.py --samples 500 --split test
```

The script writes files like:

```text
D:\Final_Year_Project\Deepfake_Dissertation\dataset\processed\hindi\real\hi_real_000001.wav
D:\Final_Year_Project\Deepfake_Dissertation\dataset\processed\marathi\real\mr_real_000001.wav
D:\Final_Year_Project\Deepfake_Dissertation\dataset\processed\tamil\real\ta_real_000001.wav
```

It also creates:

```text
D:\Final_Year_Project\Deepfake_Dissertation\dataset\metadata\metadata_real.csv
```

## 4. Labels

Use these labels:

| Folder | Label | label_id |
|---|---|---:|
| `real/` | `real` | 0 |
| `fake/` | `fake` | 1 |

For the cross-lingual experiment:

| Language | Project Split |
|---|---|
| Hindi | train/validation |
| Marathi | train/validation |
| Tamil | test |
| Kannada | test |
| Malayalam | test |

The script uses:

- 85% of Hindi/Marathi as `train`
- 15% of Hindi/Marathi as `validation`
- 100% of Tamil/Kannada/Malayalam as `test`

## 5. Build Full Metadata After Adding Fake Audio

After you add fake clips into the `fake/` folders, run:

```powershell
python scripts/build_metadata.py
```

This creates:

```text
D:\Final_Year_Project\Deepfake_Dissertation\dataset\metadata\metadata_full.csv
```

## 6. Expected Metadata Columns

```text
filepath
language
language_code
label
label_id
speaker_id
source
generator
duration_seconds
project_split
source_split
transcript
original_id
```

## 7. Important Rule

For this dissertation, do not mix Tamil, Kannada, or Malayalam into training during the zero-shot experiment.

Use:

```text
Train: Hindi + Marathi
Test: Tamil + Kannada + Malayalam
```
