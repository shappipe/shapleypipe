"""
apply_saga_pipeline.py
=======================
A scikit-learn reimplementation of SAGA's data-CLEANING operators plus a wrapper
(`SagaPipeline`) that turns a SAGA pipeline -- a list of operator-name strings, e.g.
["forward_fill", "outlierBySd", "imputeByMedian", "winsorize", "underSampling", "mice"]
-- into a fitted cleaner with the CORRECT train/test semantics.

WHY THIS FILE EXISTS
--------------------
SAGA (Siddiqi, Kern & Boehm, "SAGA: a scalable framework for optimizing data
cleaning pipelines for ML applications", PACMMOD 2023) optimizes *data-cleaning*
pipelines. Its operators are NOT the 25 feature-transformation operators of the
ShapleyPipe library (Table 2). They live in SAGA's own codebase, which is why
there is no `apply_saga_pipeline` to import from the ShapleyPipe repo. This module
re-creates the 14 operators that appear in `saga_pipelines_full.json`.

CRITICAL SEMANTIC POINT (do not skip)
-------------------------------------
Some SAGA operators change ROWS or LABELS, so they are TRAIN-ONLY:
    underSampling, tomeklink   -> resample rows  (need y; change the row count)
    flipLabels                 -> rewrite y
    outlierBySd / outlierByIQR -> repair outliers (see OUTLIER_MODE below)
You can never resample / relabel / drop the TEST set, because you must emit a
prediction for every test row. Therefore the naive `pipe.transform(X)` interface
is wrong. The correct usage is:

    from apply_saga_pipeline import SagaPipeline
    cleaner = SagaPipeline(op_names, seed=0)
    Xtr_c, ytr_c = cleaner.fit_resample(X_tr, y_tr)   # TRAIN: row/label ops applied
    Xte_c        = cleaner.transform(X_te)            # TEST : column transforms only
    clf.fit(Xtr_c, ytr_c)
    acc = accuracy_score(y_te, clf.predict(Xte_c))

CAVEATS / APPROXIMATIONS (state these in any rebuttal that uses this code)
-------------------------------------------------------------------------
* imputeByFd, flipLabels, tomeklink reproduce SAGA's *intent* with standard,
  documented heuristics. They are faithful re-implementations, not SAGA's exact
  internal code.
* The `hyper_params` vectors in the JSON are SAGA's opaque internal encoding and
  are NOT decoded here. This module uses SAGA-typical defaults, all exposed as the
  constants below so you can tune them in one place.
* Scaling/outlier/winsorize steps are NaN-aware (NaN flows through and is cleaned
  by a final median-impute guard), so a pipeline that scales before imputing -- or
  never imputes at all, e.g. vehicle = ["normalize","normalize"] -- will not crash.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (enables IterativeImputer)
from sklearn.impute import IterativeImputer
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

# ----------------------------------------------------------------------------- #
# Tunable defaults (SAGA hyper_params are not decoded; these are sensible values)
# ----------------------------------------------------------------------------- #
SD_K           = 3.0      # outlierBySd: flag |x - mean| > SD_K * std
IQR_FACTOR     = 1.5      # outlierByIQR: fences = [Q1 - f*IQR, Q3 + f*IQR]
WINSOR_LOW_P   = 5.0      # winsorize: lower percentile
WINSOR_HIGH_P  = 95.0     # winsorize: upper percentile
MICE_MAX_ITER  = 10       # mice (IterativeImputer) iterations
FLIP_THRESHOLD = 0.90     # flipLabels: only relabel when model prob > this AND disagrees
FLIP_KNN_K     = 5        # flipLabels: k for the kNN auditor
TOMEK_NN_K     = 1        # tomeklink: nearest-neighbour count (excluding self)

# outlierBySd / outlierByIQR repair strategy:
#   "clip" (default): clip to the learned fence on BOTH train and test (shape-preserving,
#                     deterministic, and never deletes the dataset).
#   "drop"          : on TRAIN, drop any row outside the fence (SAGA's row-removal variant);
#                     on TEST, fall back to clipping (you cannot drop test rows).
OUTLIER_MODE   = "clip"


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
def _to_float_2d(X):
    """Coerce X (DataFrame / ndarray / list) to a float ndarray, factorizing any
    non-numeric columns. Returns (array, columns) so DataFrame output is possible."""
    if isinstance(X, pd.DataFrame):
        df = X.copy()
        cols = list(df.columns)
    else:
        df = pd.DataFrame(np.asarray(X))
        cols = list(df.columns)
    for c in df.columns:
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.factorize(df[c])[0].astype(float)
            df[c] = df[c].replace(-1, np.nan)  # factorize maps NaN -> -1
    return df.to_numpy(dtype=float), cols


# ----------------------------------------------------------------------------- #
# Operator steps. Each step implements:
#   fit(X, y)            -> learn parameters from the current TRAIN matrix
#   apply_train(X, y)    -> (X, y) for training (may drop / resample / relabel rows)
#   apply_test(X)        -> X for test (column transforms only; row/label ops are no-ops)
# ----------------------------------------------------------------------------- #
class _Step:
    is_train_only = False
    def fit(self, X, y):            return self
    def apply_train(self, X, y):    return self.apply_test(X), y
    def apply_test(self, X):        return X


# ---- imputation (stateful column fills; applied to train AND test) ---------- #
class _ImputeByMean(_Step):
    def fit(self, X, y):
        self.fill_ = np.nanmean(X, axis=0)
        self.fill_ = np.where(np.isnan(self.fill_), 0.0, self.fill_)
        return self
    def apply_test(self, X):
        X = X.copy(); idx = np.where(np.isnan(X))
        X[idx] = np.take(self.fill_, idx[1]); return X

class _ImputeByMedian(_Step):
    def fit(self, X, y):
        self.fill_ = np.nanmedian(X, axis=0)
        self.fill_ = np.where(np.isnan(self.fill_), 0.0, self.fill_)
        return self
    def apply_test(self, X):
        X = X.copy(); idx = np.where(np.isnan(X))
        X[idx] = np.take(self.fill_, idx[1]); return X

class _FillDefault(_Step):
    """SAGA fillDefault: replace missing numeric with a constant default (0.0)."""
    def apply_test(self, X):
        X = X.copy(); X[np.isnan(X)] = 0.0; return X

class _ForwardFill(_Step):
    """Last-observation-carried-forward (then back-fill leading NaN). Any value still
    missing afterwards is filled with the train median learned at fit()."""
    def fit(self, X, y):
        self.med_ = np.nanmedian(X, axis=0)
        self.med_ = np.where(np.isnan(self.med_), 0.0, self.med_)
        return self
    def apply_test(self, X):
        df = pd.DataFrame(X).ffill(axis=0).bfill(axis=0)
        out = np.array(df.to_numpy(dtype=float), copy=True)  # ensure writable (pandas 3.0)
        idx = np.where(np.isnan(out))
        out[idx] = np.take(self.med_, idx[1])
        return out

class _ImputeByFd(_Step):
    """Heuristic stand-in for SAGA's functional-dependency imputation. For each
    column with missing values, find the most-correlated fully/least-missing column,
    bin it into deciles, and fill from the per-bin median of the target column.
    Falls back to the global median when no usable predictor exists. APPROXIMATION."""
    N_BINS = 10
    def fit(self, X, y):
        n, d = X.shape
        self.global_med_ = np.nanmedian(X, axis=0)
        self.global_med_ = np.where(np.isnan(self.global_med_), 0.0, self.global_med_)
        self.maps_ = {}  # target col -> (src col, bin_edges, per-bin median)
        nan_frac = np.isnan(X).mean(axis=0)
        for c in range(d):
            if nan_frac[c] == 0:
                continue
            best_src, best_corr = None, 0.0
            for s in range(d):
                if s == c:
                    continue
                mask = ~np.isnan(X[:, c]) & ~np.isnan(X[:, s])
                if mask.sum() < max(10, int(0.1 * n)):
                    continue
                a, b = X[mask, c], X[mask, s]
                if np.std(a) == 0 or np.std(b) == 0:
                    continue
                corr = abs(np.corrcoef(a, b)[0, 1])
                if np.isfinite(corr) and corr > best_corr:
                    best_corr, best_src = corr, s
            if best_src is None:
                continue
            s = best_src
            mask = ~np.isnan(X[:, c]) & ~np.isnan(X[:, s])
            src_vals = X[mask, s]
            edges = np.unique(np.nanpercentile(src_vals, np.linspace(0, 100, self.N_BINS + 1)))
            if edges.size < 2:
                continue
            bins = np.clip(np.digitize(src_vals, edges[1:-1]), 0, len(edges) - 2)
            per_bin = {}
            for bnum in np.unique(bins):
                per_bin[int(bnum)] = float(np.median(X[mask, c][bins == bnum]))
            self.maps_[c] = (s, edges, per_bin)
        return self
    def apply_test(self, X):
        X = X.copy()
        for c in range(X.shape[1]):
            miss = np.isnan(X[:, c])
            if not miss.any():
                continue
            if c in self.maps_:
                s, edges, per_bin = self.maps_[c]
                src = X[:, c * 0 + s]
                for i in np.where(miss)[0]:
                    sv = src[i]
                    if np.isnan(sv):
                        X[i, c] = self.global_med_[c]
                    else:
                        b = int(np.clip(np.digitize([sv], edges[1:-1])[0], 0, len(edges) - 2))
                        X[i, c] = per_bin.get(b, self.global_med_[c])
            else:
                X[miss, c] = self.global_med_[c]
        return X

class _Mice(_Step):
    """Multiple Imputation by Chained Equations == sklearn IterativeImputer."""
    def fit(self, X, y):
        self.imp_ = IterativeImputer(max_iter=MICE_MAX_ITER, random_state=0, sample_posterior=False)
        # IterativeImputer needs >=1 observed value per column; guard all-NaN columns.
        self.allnan_ = np.isnan(X).all(axis=0)
        Xf = X.copy()
        if self.allnan_.any():
            Xf[:, self.allnan_] = 0.0
        self.imp_.fit(Xf)
        return self
    def apply_test(self, X):
        Xf = X.copy()
        if self.allnan_.any():
            Xf[:, self.allnan_] = 0.0
        return self.imp_.transform(Xf)


# ---- outlier handling (bounds learned on train) ---------------------------- #
class _OutlierBounds(_Step):
    """Base: learn [lo, hi] per column; clip on test. On train, clip or drop per OUTLIER_MODE."""
    def _bounds(self, X):
        raise NotImplementedError
    def fit(self, X, y):
        self.lo_, self.hi_ = self._bounds(X)
        return self
    def apply_test(self, X):
        return np.clip(X, self.lo_, self.hi_)
    def apply_train(self, X, y):
        if OUTLIER_MODE == "drop":
            inside = np.ones(X.shape[0], dtype=bool)
            with np.errstate(invalid="ignore"):
                for j in range(X.shape[1]):
                    col = X[:, j]
                    ok = np.isnan(col) | ((col >= self.lo_[j]) & (col <= self.hi_[j]))
                    inside &= ok
            if inside.sum() >= max(10, int(0.5 * X.shape[0])):  # never nuke the dataset
                return X[inside], (y[inside] if y is not None else None)
        return np.clip(X, self.lo_, self.hi_), y

class _OutlierBySd(_OutlierBounds):
    def _bounds(self, X):
        mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
        mu = np.where(np.isnan(mu), 0.0, mu); sd = np.where(np.isnan(sd) | (sd == 0), 1.0, sd)
        return mu - SD_K * sd, mu + SD_K * sd

class _OutlierByIQR(_OutlierBounds):
    def _bounds(self, X):
        q1 = np.nanpercentile(X, 25, axis=0); q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        return q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr

class _Winsorize(_Step):
    """Clip to [p_low, p_high] percentiles learned on train. Applied to train AND test."""
    def fit(self, X, y):
        self.lo_ = np.nanpercentile(X, WINSOR_LOW_P, axis=0)
        self.hi_ = np.nanpercentile(X, WINSOR_HIGH_P, axis=0)
        return self
    def apply_test(self, X):
        return np.clip(X, self.lo_, self.hi_)


# ---- scaling (NaN-aware; applied to train AND test) ------------------------- #
class _Scale(_Step):
    """Standardize: (x - mean) / std, computed ignoring NaN. NaN flows through."""
    def fit(self, X, y):
        self.mu_ = np.nanmean(X, axis=0); self.sd_ = np.nanstd(X, axis=0)
        self.mu_ = np.where(np.isnan(self.mu_), 0.0, self.mu_)
        self.sd_ = np.where(np.isnan(self.sd_) | (self.sd_ == 0), 1.0, self.sd_)
        return self
    def apply_test(self, X):
        return (X - self.mu_) / self.sd_

class _Normalize(_Step):
    """SAGA 'normalize' == min-max scaling to [0, 1] (NOT row L2). NaN flows through."""
    def fit(self, X, y):
        self.mn_ = np.nanmin(X, axis=0); mx = np.nanmax(X, axis=0)
        self.mn_ = np.where(np.isnan(self.mn_), 0.0, self.mn_)
        rng = mx - self.mn_
        self.rng_ = np.where(np.isnan(rng) | (rng == 0), 1.0, rng)
        return self
    def apply_test(self, X):
        return (X - self.mn_) / self.rng_


# ---- resampling & label repair (TRAIN-ONLY; test is identity) --------------- #
class _UnderSampling(_Step):
    is_train_only = True
    def __init__(self, rng): self.rng = rng
    def apply_train(self, X, y):
        if y is None: return X, y
        classes, counts = np.unique(y, return_counts=True)
        target = counts.min()
        keep = []
        for c in classes:
            idx = np.where(y == c)[0]
            keep.append(self.rng.choice(idx, size=target, replace=False) if len(idx) > target else idx)
        keep = np.concatenate(keep); self.rng.shuffle(keep)
        return X[keep], y[keep]

class _TomekLink(_Step):
    """Remove Tomek links (mutual 1-NN pairs of opposite class); drop the majority member.
    Uses imblearn if present, else a pure-sklearn fallback. TRAIN-ONLY."""
    is_train_only = True
    def __init__(self, rng): self.rng = rng
    def apply_train(self, X, y):
        if y is None or len(np.unique(y)) < 2:
            return X, y
        Xc = np.nan_to_num(X, nan=0.0)  # NN needs finite values
        try:
            from imblearn.under_sampling import TomekLinks
            Xr, yr = TomekLinks().fit_resample(Xc, y)
            # map back to original (un-nan'd) rows via row matching is fragile; instead
            # recompute the mask ourselves below for consistency.
        except Exception:
            pass
        nn = NearestNeighbors(n_neighbors=TOMEK_NN_K + 1).fit(Xc)
        _, ind = nn.kneighbors(Xc)
        nbr = ind[:, 1]
        classes, counts = np.unique(y, return_counts=True)
        majority = classes[np.argmax(counts)]
        drop = set()
        for i in range(len(y)):
            j = nbr[i]
            if nbr[j] == i and y[i] != y[j]:          # mutual NN, opposite class -> Tomek link
                drop.add(i if y[i] == majority else j)  # remove the majority member
        keep = np.array([i for i in range(len(y)) if i not in drop])
        if keep.size < max(10, int(0.5 * len(y))):
            return X, y
        return X[keep], y[keep]

class _FlipLabels(_Step):
    """Conservative label-noise repair: a kNN auditor relabels a training sample only
    when it confidently disagrees with the given label. TRAIN-ONLY. APPROXIMATION."""
    is_train_only = True
    def __init__(self, rng): self.rng = rng
    def apply_train(self, X, y):
        if y is None or len(np.unique(y)) < 2:
            return X, y
        Xc = np.nan_to_num(X, nan=0.0)
        k = min(FLIP_KNN_K, max(1, len(y) - 1))
        try:
            knn = KNeighborsClassifier(n_neighbors=k).fit(Xc, y)
            proba = knn.predict_proba(Xc)
            classes = knn.classes_
            pred = classes[proba.argmax(axis=1)]
            conf = proba.max(axis=1)
        except Exception:
            return X, y
        y_new = y.copy()
        flip = (pred != y) & (conf >= FLIP_THRESHOLD)
        y_new[flip] = pred[flip]
        return X, y_new



class _Abstain(_Step):
    """SAGA 'abstain' operator: a no-op that passes data through unchanged.
    SAGA uses this as a learnable 'skip' primitive — it abstains from applying
    any transformation, allowing the optimizer to effectively shorten the pipeline
    or leave certain steps unused."""
    def fit(self, X, y):            return self
    def apply_train(self, X, y):    return X, y
    def apply_test(self, X):        return X


class _SMOTE(_Step):
    """Synthetic Minority Over-sampling Technique. Generates synthetic samples for
    the minority class(es) by interpolating between a minority sample and its k
    nearest neighbours. TRAIN-ONLY. Uses imblearn if available; falls back to an
    inline kNN-based implementation."""
    is_train_only = True
    def __init__(self, rng): self.rng = rng
    def apply_train(self, X, y):
        if y is None or len(np.unique(y)) < 2:
            return X, y
        Xc = np.nan_to_num(X, nan=0.0)
        try:
            from imblearn.over_sampling import SMOTE as _ImblearnSMOTE
            Xr, yr = _ImblearnSMOTE(random_state=self.rng.integers(2**31)).fit_resample(Xc, y)
            return Xr.astype(X.dtype), yr
        except Exception:
            pass
        # Fallback: basic SMOTE using pure sklearn
        classes, counts = np.unique(y, return_counts=True)
        majority_n = counts.max()
        X_res, y_res = [Xc], [y]
        k = min(5, Xc.shape[0] - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Xc)
        _, ind = nn.kneighbors(Xc)
        for cls in classes:
            n_minority = (y == cls).sum()
            if n_minority >= majority_n:
                continue
            n_synth = majority_n - n_minority
            idx = np.where(y == cls)[0]
            if len(idx) < 2:
                continue
            synth = []
            for _ in range(n_synth):
                pick = idx[self.rng.integers(len(idx))]
                nbr_idx = ind[pick, 1:]  # exclude self
                nbr = nbr_idx[self.rng.integers(len(nbr_idx))]
                diff = Xc[nbr] - Xc[pick]
                gap = self.rng.random()
                synth.append(Xc[pick] + gap * diff)
            if synth:
                X_res.append(np.array(synth))
                y_res.append(np.full(len(synth), cls))
        Xr = np.concatenate(X_res, axis=0).astype(X.dtype)
        yr = np.concatenate(y_res, axis=0)
        return Xr, yr


# ----------------------------------------------------------------------------- #
# registry
# ----------------------------------------------------------------------------- #
def _build_step(name, rng):
    name = name.strip()
    simple = {
        "imputeByMean":   _ImputeByMean,  "imputeByMedian": _ImputeByMedian,
        "fillDefault":    _FillDefault,   "forward_fill":   _ForwardFill,
        "imputeByFd":     _ImputeByFd,    "mice":           _Mice,
        "outlierBySd":    _OutlierBySd,   "outlierByIQR":   _OutlierByIQR,
        "winsorize":      _Winsorize,     "scale":          _Scale,
        "normalize":      _Normalize,     "abstain":        _Abstain,
    }
    if name in simple:
        return simple[name]()
    needs_rng = {"underSampling": _UnderSampling, "tomeklink": _TomekLink, "flipLabels": _FlipLabels, "SMOTE": _SMOTE}
    if name in needs_rng:
        return needs_rng[name](rng)
    raise ValueError(f"Unknown SAGA operator: {name!r}. "
                     f"Known: {sorted(list(simple) + list(needs_rng))}")


# ----------------------------------------------------------------------------- #
# the wrapper
# ----------------------------------------------------------------------------- #
class SagaPipeline:
    """Fit a SAGA cleaning pipeline on train, then transform train (fit_resample) and
    test (transform) with the correct, asymmetric semantics. See module docstring."""

    def __init__(self, op_names, seed: int = 0):
        self.op_names = list(op_names)
        self.seed = seed

    def fit_resample(self, X, y):
        """Fit each step on the running train matrix and apply it. Row/label-altering
        steps (undersampling, tomek, flip, outlier-drop) modify (X, y). Returns the
        cleaned (X_train, y_train) and stores the fitted steps for `transform`."""
        rng = np.random.default_rng(self.seed)
        Xc, self._cols = _to_float_2d(X)
        yc = None if y is None else np.asarray(y)
        self._steps = []
        for name in self.op_names:
            step = _build_step(name, rng)
            step.fit(Xc, yc)               # learn params from current train data
            Xc, yc = step.apply_train(Xc, yc)
            self._steps.append(step)
        # Final guard: clean any residual NaN with the post-pipeline train medians.
        self._final_med = np.nanmedian(Xc, axis=0)
        self._final_med = np.where(np.isnan(self._final_med), 0.0, self._final_med)
        Xc = self._fill_residual(Xc)
        return Xc, yc

    def transform(self, X):
        """Apply the fitted COLUMN transforms to test data. Row/label-altering steps
        are skipped (you cannot resample or relabel the test set); outlier steps clip."""
        if not hasattr(self, "_steps"):
            raise RuntimeError("Call fit_resample(X_train, y_train) before transform().")
        Xc, _ = _to_float_2d(X)
        for step in self._steps:
            if step.is_train_only:
                continue
            Xc = step.apply_test(Xc)
        return self._fill_residual(Xc)

    def _fill_residual(self, X):
        idx = np.where(np.isnan(X))
        if idx[0].size:
            X = X.copy(); X[idx] = np.take(self._final_med, idx[1])
        return X


def apply_saga_pipeline(op_names, X_train, y_train, seed: int = 0):
    """Convenience wrapper matching the call shape in your harness.

    Returns (cleaner, X_train_clean, y_train_clean). Because SAGA pipelines resample
    the TRAINING data, the cleaned (X, y) are returned here directly -- do NOT call
    cleaner.transform(X_train) to recover them (that would skip resampling). For TEST
    data, call cleaner.transform(X_test).
    """
    cleaner = SagaPipeline(op_names, seed=seed)
    Xtr_c, ytr_c = cleaner.fit_resample(X_train, y_train)
    return cleaner, Xtr_c, ytr_c
