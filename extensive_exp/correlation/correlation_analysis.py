"""
correlation_analysis.py
=======================
Step 2 of the rebuttal experiment: quantify which meta-features
correlate with the divergence index `d` across the 27 datasets.

Input:
    A CSV with columns:
        dataset, d, acc_h_sp, acc_loo,
        <all meta-feature columns from meta_features.py>

Output:
    - A table of Spearman rho, p-value, significance mark per feature.
    - A saved CSV `spearman_results.csv` for direct inclusion in the paper.
"""

import numpy as np
import pandas as pd
from scipy import stats


META_FEATURE_COLS = [
    "missing_ratio",
    "categorical_ratio",
    "scale_heterogeneity",
    "outlier_ratio",
    "feature_sample_ratio",
    "class_imbalance",
    "n_features",
    "skewness_avg",
]


def spearman_table(df: pd.DataFrame, target: str = "d") -> pd.DataFrame:
    """
    Compute Spearman correlation between each meta-feature and the target.

    Returns a DataFrame sorted by |rho| descending, with columns:
        feature, spearman_rho, p_value, n, significant_0.05, significant_0.01
    """
    rows = []
    for feat in META_FEATURE_COLS:
        if feat not in df.columns:
            continue
        x = df[feat].values
        y = df[target].values
        # Drop NaN pairs
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 3:
            continue
        rho, pval = stats.spearmanr(x[mask], y[mask])
        rows.append({
            "feature": feat,
            "spearman_rho": rho,
            "p_value": pval,
            "n": int(mask.sum()),
            "sig_0.05": "*" if pval < 0.05 else "",
            "sig_0.01": "**" if pval < 0.01 else "",
        })
    out = pd.DataFrame(rows)
    out = out.reindex(out["spearman_rho"].abs().sort_values(ascending=False).index)
    return out.reset_index(drop=True)


def pretty_print(table: pd.DataFrame) -> None:
    """Print a nicely formatted table for inclusion in the paper."""
    print(f"\n{'Feature':<24}{'Spearman ρ':>12}{'p-value':>12}{'n':>6}  sig")
    print("-" * 64)
    for _, r in table.iterrows():
        marker = r["sig_0.01"] or r["sig_0.05"]
        print(
            f"{r['feature']:<24}"
            f"{r['spearman_rho']:>12.3f}"
            f"{r['p_value']:>12.4f}"
            f"{r['n']:>6}  {marker}"
        )
    print("-" * 64)
    print("  *  p < 0.05      **  p < 0.01\n")


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-correlation among meta-features themselves (to detect redundancy)."""
    cols = [c for c in META_FEATURE_COLS if c in df.columns]
    return df[cols].corr(method="spearman").round(3)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="dataset_meta.csv",
        help="CSV with dataset name, d, and meta-feature columns",
    )
    parser.add_argument("--target", default="d")
    parser.add_argument("--out", default="spearman_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} datasets from {args.input}")
    print(f"Columns: {list(df.columns)}")

    table = spearman_table(df, target=args.target)
    pretty_print(table)
    table.to_csv(args.out, index=False)
    print(f"Saved Spearman results -> {args.out}")

    print("\nCross-correlation among meta-features:")
    print(correlation_matrix(df))
