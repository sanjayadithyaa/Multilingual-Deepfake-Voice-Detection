import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


LABEL_MAP = {
    "real": 0,
    "fake": 1,
    "0": 0,
    "1": 1,
    0: 0,
    1: 1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one model from a prediction CSV."
    )
    parser.add_argument("--predictions_csv", required=True, help="CSV with one row per test file.")
    parser.add_argument("--output_dir", required=True, help="Directory where reports will be saved.")
    parser.add_argument("--model_name", default="model", help="Label used in plots and reports.")
    parser.add_argument(
        "--label_col",
        default="true_label",
        help="Column containing ground-truth labels (real/fake or 0/1).",
    )
    parser.add_argument(
        "--score_col",
        default="score_fake",
        help="Column containing predicted probability/score for the fake class.",
    )
    parser.add_argument(
        "--pred_col",
        default="pred_label",
        help="Optional predicted label column. If missing, it will be derived using --threshold.",
    )
    parser.add_argument(
        "--language_col",
        default="language",
        help="Optional language column for per-language metrics.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold used when predicted labels are not supplied.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for AUC confidence interval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def normalize_labels(series: pd.Series) -> pd.Series:
    normalized = series.map(lambda x: LABEL_MAP.get(str(x).strip().lower(), LABEL_MAP.get(x)))
    if normalized.isna().any():
        bad_values = series[normalized.isna()].unique().tolist()
        raise ValueError(f"Unsupported label values found: {bad_values}")
    return normalized.astype(int)


def bootstrap_auc_ci(y_true, y_score, n_bootstrap=2000, seed=42, alpha=0.95):
    rng = np.random.default_rng(seed)
    aucs = []
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        sample_y = y_true[idx]
        sample_score = y_score[idx]

        if len(np.unique(sample_y)) < 2:
            continue

        aucs.append(roc_auc_score(sample_y, sample_score))

    if not aucs:
        raise ValueError("Bootstrap AUC CI could not be computed because resamples lacked both classes.")

    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q
    return {
        "auc_mean_bootstrap": float(np.mean(aucs)),
        "auc_ci_lower": float(np.quantile(aucs, lower_q)),
        "auc_ci_upper": float(np.quantile(aucs, upper_q)),
    }


def compute_descriptive_stats(df, score_col):
    output_rows = []
    groups = {
        "all": df,
        "true_real": df[df["true_label_num"] == 0],
        "true_fake": df[df["true_label_num"] == 1],
    }

    for group_name, group_df in groups.items():
        if group_df.empty:
            continue

        rounded_scores = group_df[score_col].round(4)
        modes = rounded_scores.mode()
        output_rows.append(
            {
                "group": group_name,
                "count": int(len(group_df)),
                "mean_score_fake": float(group_df[score_col].mean()),
                "median_score_fake": float(group_df[score_col].median()),
                "mode_score_fake_rounded_4dp": ", ".join(map(str, modes.tolist())) if not modes.empty else "",
                "std_score_fake": float(group_df[score_col].std(ddof=1)) if len(group_df) > 1 else 0.0,
                "min_score_fake": float(group_df[score_col].min()),
                "max_score_fake": float(group_df[score_col].max()),
            }
        )

    return pd.DataFrame(output_rows)


def compute_language_metrics(df):
    rows = []
    for language, group_df in df.groupby("language"):
        y_true = group_df["true_label_num"]
        y_pred = group_df["pred_label_num"]
        y_score = group_df["score_fake"]
        rows.append(
            {
                "language": language,
                "n_samples": int(len(group_df)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision_fake": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall_fake": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_fake": float(f1_score(y_true, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("language")


def save_confusion_matrix_plot(cm, labels, output_path, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_roc_plot(fpr, tpr, roc_auc, output_path, title):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions_csv)

    required_cols = [args.label_col, args.score_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    df = df.copy()
    df["true_label_num"] = normalize_labels(df[args.label_col])
    df["score_fake"] = pd.to_numeric(df[args.score_col], errors="raise")

    if args.pred_col in df.columns:
        df["pred_label_num"] = normalize_labels(df[args.pred_col])
    else:
        df["pred_label_num"] = (df["score_fake"] >= args.threshold).astype(int)

    if args.language_col in df.columns:
        df["language"] = df[args.language_col].astype(str)

    y_true = df["true_label_num"].to_numpy()
    y_pred = df["pred_label_num"].to_numpy()
    y_score = df["score_fake"].to_numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    roc_auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_ci = bootstrap_auc_ci(
        y_true=y_true,
        y_score=y_score,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    summary = {
        "model_name": args.model_name,
        "n_samples": int(len(df)),
        "threshold": args.threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    summary.update(auc_ci)

    real_scores = df.loc[df["true_label_num"] == 0, "score_fake"]
    fake_scores = df.loc[df["true_label_num"] == 1, "score_fake"]
    u_stat, p_value = mannwhitneyu(fake_scores, real_scores, alternative="greater")
    hypothesis = {
        "test_name": "Mann-Whitney U",
        "null_hypothesis": "Fake and real files have the same score distribution.",
        "alternative_hypothesis": "Fake files have higher fake-class scores than real files.",
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "decision_alpha_0_05": "Reject H0" if p_value < 0.05 else "Fail to reject H0",
    }

    descriptive_stats = compute_descriptive_stats(df, "score_fake")
    descriptive_stats.to_csv(output_dir / "descriptive_stats.csv", index=False)

    pd.DataFrame(
        cm,
        index=["true_real", "true_fake"],
        columns=["pred_real", "pred_fake"],
    ).to_csv(output_dir / "confusion_matrix.csv")

    pd.DataFrame(
        {"fpr": fpr, "tpr": tpr, "threshold": thresholds}
    ).to_csv(output_dir / "roc_points.csv", index=False)

    if "language" in df.columns:
        compute_language_metrics(df).to_csv(output_dir / "language_metrics.csv", index=False)

    pd.DataFrame(classification_report(y_true, y_pred, target_names=["real", "fake"], output_dict=True)).transpose().to_csv(
        output_dir / "classification_report.csv"
    )

    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "hypothesis_test.json", "w", encoding="utf-8") as f:
        json.dump(hypothesis, f, indent=2)

    save_confusion_matrix_plot(
        cm=cm,
        labels=["real", "fake"],
        output_path=output_dir / "confusion_matrix.png",
        title=f"{args.model_name} Confusion Matrix",
    )

    save_roc_plot(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
        output_path=output_dir / "roc_curve.png",
        title=f"{args.model_name} ROC Curve",
    )

    print(json.dumps(summary, indent=2))
    print(json.dumps(hypothesis, indent=2))


if __name__ == "__main__":
    main()
