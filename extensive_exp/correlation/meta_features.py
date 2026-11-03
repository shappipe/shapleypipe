"""
meta_features.py
================
Compute low-cost dataset meta-features that can predict the divergence
index `d` (between PipeShapley and LOO contributions) a priori — i.e.,
BEFORE running any H-ShapleyPipe or LOO pipeline.

All features are O(n*d) or cheaper. No model training involved.

Usage:
    from meta_features import compute_meta_features
    features = compute_meta_features(X, y)
    # X: pandas DataFrame, y: array-like labels
"""

import numpy as np
import pandas as pd
from scipy import stats


def _is_categorical(series: pd.Series, max_unique_ratio: float = 0.05) -> bool:
    """Heuristic: treat as categorical if dtype is object/category,
    or if it has few unique values relative to length."""
    if series.dtype == object or str(series.dtype) == "category":
        return True
    n_unique = series.nunique(dropna=True)
    if n_unique <= 20:
        return True
    if n_unique / max(len(series), 1) <= max_unique_ratio:
        return True
    return False


def compute_meta_features(X: pd.DataFrame, y) -> dict:
    """
    Compute 8 meta-features for a tabular classification dataset.

    Parameters
    ----------
    X : pd.DataFrame  (shape [n_samples, n_features], may contain NaN)
    y : array-like    (shape [n_samples])

    Returns
    -------
    dict with keys:
        missing_ratio, categorical_ratio, scale_heterogeneity,
        outlier_ratio, feature_sample_ratio, class_imbalance,
        n_features, skewness_avg
    """
    n_samples, n_features = X.shape
    y = np.asarray(y)

    # ---- 1. missing_ratio ---------------------------------------------------
    # Fraction of NaN cells across the whole table.
    total_cells = n_samples * n_features
    missing_ratio = float(X.isna().sum().sum()) / max(total_cells, 1)

    # ---- 2. categorical_ratio ----------------------------------------------
    cat_mask = [_is_categorical(X[c]) for c in X.columns]
    categorical_ratio = float(np.mean(cat_mask))

    # Identify numeric columns for remaining numeric-only features.
    num_cols = [c for c, is_cat in zip(X.columns, cat_mask) if not is_cat]
    X_num = X[num_cols].apply(pd.to_numeric, errors="coerce")

    # ---- 3. scale_heterogeneity --------------------------------------------
    # Coefficient of variation of per-feature standard deviations.
    # High value => features live on very different scales => Scaling matters.
    if len(num_cols) >= 2:
        stds = X_num.std(skipna=True).replace(0, np.nan).dropna().values
        if len(stds) >= 2 and np.mean(stds) > 0:
            scale_heterogeneity = float(np.std(stds) / np.mean(stds))
        else:
            scale_heterogeneity = 0.0
    else:
        scale_heterogeneity = 0.0

    # ---- 4. outlier_ratio --------------------------------------------------
    # Fraction of values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR], averaged over
    # numeric features. High => Robust scalers / QuantileTransformer matter.
    if len(num_cols) >= 1:
        outlier_frac = []
        for c in num_cols:
            col = X_num[c].dropna().values
            if len(col) < 4:
                continue
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            frac = float(np.mean((col < lower) | (col > upper)))
            outlier_frac.append(frac)
        outlier_ratio = float(np.mean(outlier_frac)) if outlier_frac else 0.0
    else:
        outlier_ratio = 0.0

    # ---- 5. feature_sample_ratio -------------------------------------------
    # d / n. High => feature engineering / selection likely position-sensitive.
    feature_sample_ratio = float(n_features) / max(n_samples, 1)

    # ---- 6. class_imbalance ------------------------------------------------
    # Ratio of most-frequent class count to least-frequent class count.
    _, counts = np.unique(y, return_counts=True)
    if len(counts) >= 2 and counts.min() > 0:
        class_imbalance = float(counts.max() / counts.min())
    else:
        class_imbalance = 1.0

    # ---- 7. n_features -----------------------------------------------------
    n_features_out = int(n_features)

    # ---- 8. skewness_avg ---------------------------------------------------
    # Mean |skewness| across numeric features. High => Power / Quantile
    # Transformers at different positions can behave very differently.
    if len(num_cols) >= 1:
        skew_vals = []
        for c in num_cols:
            col = X_num[c].dropna().values
            if len(col) < 3 or np.std(col) == 0:
                continue
            skew_vals.append(abs(float(stats.skew(col))))
        skewness_avg = float(np.mean(skew_vals)) if skew_vals else 0.0
    else:
        skewness_avg = 0.0

    return {
        "missing_ratio": missing_ratio,
        "categorical_ratio": categorical_ratio,
        "scale_heterogeneity": scale_heterogeneity,
        "outlier_ratio": outlier_ratio,
        "feature_sample_ratio": feature_sample_ratio,
        "class_imbalance": class_imbalance,
        "n_features": n_features_out,
        "skewness_avg": skewness_avg,
    }


def compute_meta_features_batch(datasets: dict) -> pd.DataFrame:
    """
    Compute meta-features for a batch of datasets.

    Parameters
    ----------
    datasets : dict[str, (X, y)]
        Mapping from dataset name to (X, y) pair.

    Returns
    -------
    pd.DataFrame  indexed by dataset name, with one column per meta-feature.
    """
    rows = []
    for name, (X, y) in datasets.items():
        feats = compute_meta_features(X, y)
        feats["dataset"] = name
        rows.append(feats)
    df = pd.DataFrame(rows).set_index("dataset")
    return df


if __name__ == "__main__":
    # Smoke test on a toy dataset.
    rng = np.random.RandomState(0)
    n = 500
    X = pd.DataFrame({
        "num1": rng.randn(n) * 100,
        "num2": rng.randn(n) * 0.01,
        "num3_skewed": rng.exponential(1.0, n),
        "cat1": rng.choice(["a", "b", "c"], n),
        "cat2": rng.choice([0, 1], n),
    })
    # Inject some missingness
    X.loc[rng.choice(n, 50, replace=False), "num1"] = np.nan
    y = rng.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1])

    feats = compute_meta_features(X, y)
    for k, v in feats.items():
        print(f"  {k:25s} = {v:.4f}" if isinstance(v, float) else f"  {k:25s} = {v}")
