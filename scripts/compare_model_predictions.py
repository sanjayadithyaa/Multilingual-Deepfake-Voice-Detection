import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, roc_curve


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
        description="Compare two models using the same test-set prediction CSVs."
    )
    parser.add_argument("--csv_a", required=True, help="Prediction CSV for model A.")
    parser.add_argument("--csv_b", required=True, help="Prediction CSV for model B.")
    parser.add_argument("--name_a", default="Model_A", help="Display name for model A.")
    parser.add_argument("--name_b", default="Model_B", help="Display name for model B.")
    parser.add_argument("--output_dir", required=True, help="Output folder.")
    parser.add_argument("--id_col", default="file_path", help="Unique sample identifier shared by both CSVs.")
    parser.add_argument("--label_col", default="true_label", help="Ground-truth label column.")
    parser.add_argument("--score_col", default="score_fake", help="Probability/score for fake class.")
    parser.add_argument("--pred_col", default="pred_label", help="Predicted label column.")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap seed.")
    parser.add_argument("--n_bootstrap", type=int, default=2000, help="Bootstrap iterations.")
    return parser.parse_args()


def normalize_labels(series: pd.Series) -> pd.Series:
    normalized = series.map(lambda x: LABEL_MAP.get(str(x).strip().lower(), LABEL_MAP.get(x)))
    if normalized.isna().any():
        bad_values = series[normalized.isna()].unique().tolist()
        raise ValueError(f"Unsupported label values found: {bad_values}")
    return normalized.astype(int)


def load_predictions(csv_path, id_col, label_col, score_col, pred_col, suffix):
    df = pd.read_csv(csv_path).copy()
    needed = [id_col, label_col, score_col, pred_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    df["true_label_num"] = normalize_labels(df[label_col])
    df[f"score_fake_{suffix}"] = pd.to_numeric(df[score_col], errors="raise")
    df[f"pred_label_num_{suffix}"] = normalize_labels(df[pred_col])

    return df[[id_col, "true_label_num", f"score_fake_{suffix}", f"pred_label_num_{suffix}"]]


def mcnemar_exact_test(a_correct, b_correct):
    b_only = int(((a_correct == 0) & (b_correct == 1)).sum())
    a_only = int(((a_correct == 1) & (b_correct == 0)).sum())
    n_discordant = a_only + b_only

    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(min(a_only, b_only), n=n_discordant, p=0.5, alternative="two-sided").pvalue

    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "n_discordant": n_discordant,
        "p_value": float(p_value),
        "decision_alpha_0_05": "Reject H0" if p_value < 0.05 else "Fail to reject H0",
    }


def bootstrap_auc_difference(y_true, score_a, score_b, n_bootstrap=2000, seed=42, alpha=0.95):
    rng = np.random.default_rng(seed)
    diffs = []
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        diffs.append(roc_auc_score(sample_y, score_b[idx]) - roc_auc_score(sample_y, score_a[idx]))

    if not diffs:
        raise ValueError("Bootstrap AUC difference could not be computed.")

    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q
    return {
        "auc_diff_mean_bootstrap": float(np.mean(diffs)),
        "auc_diff_ci_lower": float(np.quantile(diffs, lower_q)),
        "auc_diff_ci_upper": float(np.quantile(diffs, upper_q)),
    }


def save_dual_roc_plot(y_true, score_a, score_b, name_a, name_b, output_path):
    fpr_a, tpr_a, _ = roc_curve(y_true, score_a)
    fpr_b, tpr_b, _ = roc_curve(y_true, score_b)
    auc_a = roc_auc_score(y_true, score_a)
    auc_b = roc_auc_score(y_true, score_b)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_a, tpr_a, linewidth=2, label=f"{name_a} AUC = {auc_a:.4f}")
    plt.plot(fpr_b, tpr_b, linewidth=2, label=f"{name_b} AUC = {auc_b:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_disagreement_heatmap(a_correct, b_correct, name_a, name_b, output_path):
    matrix = np.array(
        [
            [int(((a_correct == 1) & (b_correct == 1)).sum()), int(((a_correct == 1) & (b_correct == 0)).sum())],
            [int(((a_correct == 0) & (b_correct == 1)).sum()), int(((a_correct == 0) & (b_correct == 0)).sum())],
        ]
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Oranges",
        cbar=False,
        xticklabels=[f"{name_b} correct", f"{name_b} wrong"],
        yticklabels=[f"{name_a} correct", f"{name_a} wrong"],
    )
    plt.title("Paired Outcome Table")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_a = load_predictions(args.csv_a, args.id_col, args.label_col, args.score_col, args.pred_col, "a")
    df_b = load_predictions(args.csv_b, args.id_col, args.label_col, args.score_col, args.pred_col, "b")

    merged = df_a.merge(df_b, on=[args.id_col, "true_label_num"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping samples found between the two prediction CSVs.")

    y_true = merged["true_label_num"].to_numpy()
    pred_a = merged["pred_label_num_a"].to_numpy()
    pred_b = merged["pred_label_num_b"].to_numpy()
    score_a = merged["score_fake_a"].to_numpy()
    score_b = merged["score_fake_b"].to_numpy()

    a_correct = (pred_a == y_true).astype(int)
    b_correct = (pred_b == y_true).astype(int)

    auc_a = roc_auc_score(y_true, score_a)
    auc_b = roc_auc_score(y_true, score_b)
    mcnemar = mcnemar_exact_test(a_correct, b_correct)
    auc_diff = bootstrap_auc_difference(
        y_true=y_true,
        score_a=score_a,
        score_b=score_b,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    results = {
        "n_common_samples": int(len(merged)),
        f"{args.name_a}_auc": float(auc_a),
        f"{args.name_b}_auc": float(auc_b),
        f"{args.name_b}_minus_{args.name_a}_auc": float(auc_b - auc_a),
        "mcnemar_test": mcnemar,
        "auc_difference_bootstrap": auc_diff,
        "interpretation": (
            f"{args.name_b} is significantly better on paired classification decisions."
            if mcnemar["p_value"] < 0.05 and mcnemar["a_wrong_b_correct"] > mcnemar["a_correct_b_wrong"]
            else "No statistically significant paired classification advantage was detected at alpha = 0.05."
        ),
    }

    merged.to_csv(output_dir / "paired_predictions.csv", index=False)
    with open(output_dir / "comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    save_dual_roc_plot(y_true, score_a, score_b, args.name_a, args.name_b, output_dir / "roc_comparison.png")
    save_disagreement_heatmap(a_correct, b_correct, args.name_a, args.name_b, output_dir / "paired_outcomes.png")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
