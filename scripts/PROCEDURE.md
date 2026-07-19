# Reliable Evaluation Procedure for Your Deepfake Models

This procedure is designed to make your dissertation results reproducible, auditable, and statistically defensible.

## 1. Export one prediction CSV per model on the exact same test set

Create one CSV for the Random Forest model and one CSV for the CNN model.

Each CSV should contain one row per test audio file with at least these columns:

```text
file_path,true_label,pred_label,score_fake,language
```

Column meaning:

- `file_path`: unique identifier for the audio clip
- `true_label`: `real` or `fake`
- `pred_label`: model decision, `real` or `fake`
- `score_fake`: model probability or confidence for the fake class
- `language`: Kannada, Malayalam, Tamil, etc.

Important reliability rule:

- Both models must be evaluated on the exact same test files in the same split.
- Do not change the test set after looking at results.
- Use the saved trained models only. Do not retrain during evaluation.

## 2. Run single-model evaluation

Run this once for the Random Forest CSV and once for the CNN CSV.

Example:

```powershell
python evaluate_predictions.py `
  --predictions_csv D:\Final_Year_Project\Deepfake_Dissertation\results\rf_test_predictions.csv `
  --output_dir D:\Final_Year_Project\Deepfake_Dissertation\results\rf_evaluation `
  --model_name Random_Forest
```

```powershell
python evaluate_predictions.py `
  --predictions_csv D:\Final_Year_Project\Deepfake_Dissertation\results\cnn_test_predictions.csv `
  --output_dir D:\Final_Year_Project\Deepfake_Dissertation\results\cnn_evaluation `
  --model_name CNN
```

This script produces:

- `summary_metrics.json`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `roc_points.csv`
- `roc_curve.png`
- `descriptive_stats.csv`
- `classification_report.csv`
- `hypothesis_test.json`
- `language_metrics.csv` if `language` is present

## 3. Run paired model comparison

This checks whether the CNN is truly better than the Random Forest on the same files.

```powershell
python compare_model_predictions.py `
  --csv_a D:\Final_Year_Project\Deepfake_Dissertation\results\rf_test_predictions.csv `
  --csv_b D:\Final_Year_Project\Deepfake_Dissertation\results\cnn_test_predictions.csv `
  --name_a Random_Forest `
  --name_b CNN `
  --output_dir D:\Final_Year_Project\Deepfake_Dissertation\results\model_comparison
```

This script produces:

- `comparison_results.json`
- `paired_predictions.csv`
- `roc_comparison.png`
- `paired_outcomes.png`

## 4. What statistical outputs mean

### ROC curve and AUC

- ROC curve shows the trade-off between false positive rate and true positive rate.
- AUC closer to 1.0 means stronger class separation.
- The script also reports a bootstrap 95% confidence interval for AUC.

This makes your AUC result more trustworthy than reporting only one number.

### Confusion matrix

- Shows `TN`, `FP`, `FN`, `TP`.
- Supports accuracy, precision, recall, and F1-score.

### Mean, median, mode

These are calculated for `score_fake`:

- `all`
- `true_real`
- `true_fake`

Why this is useful:

- Real files should usually have low fake scores.
- Fake files should usually have high fake scores.
- Median is robust to outliers.
- Mode is reported on rounded scores because probabilities are continuous values.

### Hypothesis testing for one model

`evaluate_predictions.py` uses a **Mann-Whitney U test**:

- Null hypothesis: real and fake files have the same score distribution
- Alternative hypothesis: fake files receive higher fake-class scores than real files

This is a strong choice because it is non-parametric and does not assume normality.

### Hypothesis testing for comparing two models

`compare_model_predictions.py` uses **McNemar’s exact test**:

- Null hypothesis: both models have the same error rate on the same test samples
- Alternative: one model makes fewer mistakes on paired data

This is the correct classical test for comparing two classifiers on the same instances.

The script also adds a bootstrap confidence interval for the AUC difference.

## 5. Dissertation-safe reporting language

You can write results like this:

```text
The CNN achieved a higher ROC-AUC than the Random Forest baseline on the same unseen-language test set. McNemar’s exact test showed that the CNN made significantly fewer paired classification errors than the Random Forest (p < 0.05), indicating that the improvement is statistically significant rather than due to random variation.
```

For single-model score separation:

```text
The distribution of fake-class scores for fake audio was significantly higher than that for real audio according to the Mann-Whitney U test (p < 0.05), supporting that the model meaningfully separates the two classes.
```

## 6. Best practices for trustworthy results

- Fix random seeds in all evaluation code.
- Never tune thresholds on the test set unless you clearly state it.
- Keep train, validation, and test sets separate.
- Use the same preprocessing for train and test audio.
- Save prediction CSVs as evidence for your dissertation appendix.
- Report confidence intervals, not only point estimates.
- Keep a copy of the exact scripts used to produce final tables and figures.

## 7. Python packages

Install these if needed:

```powershell
python -m pip install pandas numpy scipy scikit-learn matplotlib seaborn
```
