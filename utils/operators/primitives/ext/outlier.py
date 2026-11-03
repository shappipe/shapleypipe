'''
Outlier Handling
'''

import os

import numpy as np
import pandas as pd
from loguru import logger

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from ..primitive import Primitive
from ..encoders import split_cat_and_num_cols


CONTAM = float(os.getenv('SHAPLEY_OUTLIER_CONTAM', '0.05'))
LOF_MAX_ROWS = int(os.getenv('SHAPLEY_LOF_MAX_ROWS', '30000'))
MIN_KEEP_RATIO = float(os.getenv('SHAPLEY_OUTLIER_MIN_KEEP', '0.5'))


def _numeric_matrix(train_x: pd.DataFrame):
    if train_x is None or train_x.empty or train_x.shape[1] == 0:
        return None
    _, num_x = split_cat_and_num_cols(train_x)
    if num_x.shape[1] == 0:
        return None
    mat = num_x.to_numpy(dtype=float, copy=True)
    if not np.isfinite(mat).all():
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat


def _safe_row_filter(train_x: pd.DataFrame, train_y: pd.Series, keep_mask: np.ndarray):
    n = len(train_x)
    if keep_mask.sum() < max(2, int(n * MIN_KEEP_RATIO)):
        return train_x, train_y

    y_arr = np.asarray(train_y)
    if len(y_arr) != n:
        return train_x, train_y

    before = set(pd.unique(y_arr))
    after = set(pd.unique(y_arr[keep_mask]))
    if before != after:
        return train_x, train_y

    new_x = train_x.loc[keep_mask].reset_index(drop=True)
    new_y = pd.Series(y_arr[keep_mask]).reset_index(drop=True)
    new_y.name = getattr(train_y, 'name', None)
    return new_x, new_y


class _OutlierPrimBase(Primitive):
    def can_accept(self, data):
        return self.can_accept_a(data)

    def is_needed(self, data):
        return True

    def _apply(self, train_x, test_x, train_y):
        raise NotImplementedError

    def transform_1(self, train_x, test_x, train_y):
        try:
            train_x.columns = train_x.columns.astype(str)
            test_x.columns = test_x.columns.astype(str)
            return self._apply(train_x, test_x, train_y)
        except Exception as e:
            logger.error(f'{self.get_name()} failed: {e}')
            return train_x, test_x, train_y

    def transform(self, data: pd.DataFrame, y=None, **kwargs):
        return data


# ------------------------------------------------------------------
# ID 25: IsolationForest
# ------------------------------------------------------------------
class IsolationForestFilterPrim(_OutlierPrimBase):
    def __init__(self, random_state=0):
        super(IsolationForestFilterPrim, self).__init__(name='IsolationForestFilter')
        self.id = 1
        self.gid = 25
        self.hyperparams = []
        self.type = 'outlier handling'
        self.description = (
            "Removes anomalous training rows detected by an Isolation Forest. "
            "Only the training split is filtered; the test split is left intact."
        )
        self.random_state = random_state
        self.accept_type = 'a'
        self.need_y = True

    def _apply(self, train_x, test_x, train_y):
        mat = _numeric_matrix(train_x)
        if mat is None or mat.shape[0] < 20:
            return train_x, test_x, train_y

        det = IsolationForest(
            n_estimators=100,
            contamination=CONTAM,
            random_state=self.random_state,
            n_jobs=1,
        )
        pred = det.fit_predict(mat)
        keep = pred == 1
        new_x, new_y = _safe_row_filter(train_x, train_y, keep)
        return new_x, test_x, new_y


# ------------------------------------------------------------------
# ID 26: LocalOutlierFactor
# ------------------------------------------------------------------
class LOFFilterPrim(_OutlierPrimBase):
    def __init__(self, random_state=0):
        super(LOFFilterPrim, self).__init__(name='LOFFilter')
        self.id = 2
        self.gid = 26
        self.hyperparams = []
        self.type = 'outlier handling'
        self.description = (
            "Removes training rows flagged as local density outliers by LOF. "
            "Skipped on very large training sets to bound kNN cost."
        )
        self.random_state = random_state
        self.accept_type = 'a'
        self.need_y = True

    def _apply(self, train_x, test_x, train_y):
        mat = _numeric_matrix(train_x)
        if mat is None or mat.shape[0] < 30:
            return train_x, test_x, train_y
        if mat.shape[0] > LOF_MAX_ROWS:
            logger.info(f'LOFFilter skipped: {mat.shape[0]} rows > {LOF_MAX_ROWS}')
            return train_x, test_x, train_y

        n_neighbors = int(min(20, max(5, mat.shape[0] // 10)))
        det = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=CONTAM,
            n_jobs=1,
        )
        pred = det.fit_predict(mat)
        keep = pred == 1
        new_x, new_y = _safe_row_filter(train_x, train_y, keep)
        return new_x, test_x, new_y


# ------------------------------------------------------------------
# ID 27: Winsorization
# ------------------------------------------------------------------
class WinsorizerPrim(_OutlierPrimBase):
    def __init__(self, random_state=0, lower=0.01, upper=0.99):
        super(WinsorizerPrim, self).__init__(name='Winsorizer')
        self.id = 3
        self.gid = 27
        self.hyperparams = []
        self.type = 'outlier handling'
        self.description = (
            "Clips every numeric column to the [1%, 99%] quantile range estimated "
            "on the training split, then applies the same bounds to the test split."
        )
        self.lower = lower
        self.upper = upper
        self.accept_type = 'a'
        self.need_y = False

    def _apply(self, train_x, test_x, train_y):
        cat_train, num_train = split_cat_and_num_cols(train_x)
        if num_train.shape[1] == 0 or num_train.empty:
            return train_x, test_x, train_y

        cols = list(num_train.columns)
        num_test = test_x[cols] if set(cols).issubset(set(test_x.columns)) else None
        if num_test is None:
            return train_x, test_x, train_y

        lo = num_train.quantile(self.lower)
        hi = num_train.quantile(self.upper)
        active = [c for c in cols if pd.notna(lo[c]) and pd.notna(hi[c]) and lo[c] < hi[c]]
        if not active:
            return train_x, test_x, train_y

        num_train = num_train.copy()
        num_test = num_test.copy()
        num_train[active] = num_train[active].clip(lower=lo[active], upper=hi[active], axis=1)
        num_test[active] = num_test[active].clip(lower=lo[active], upper=hi[active], axis=1)

        cat_test = test_x[list(cat_train.columns)]
        new_train = pd.concat(
            [cat_train.reset_index(drop=True), num_train.reset_index(drop=True)], axis=1)
        new_test = pd.concat(
            [cat_test.reset_index(drop=True), num_test.reset_index(drop=True)], axis=1)
        return new_train, new_test, train_y

    def transform(self, data: pd.DataFrame, y=None, **kwargs):
        try:
            cat_x, num_x = split_cat_and_num_cols(data)
            if num_x.shape[1] == 0:
                return data
            lo, hi = num_x.quantile(self.lower), num_x.quantile(self.upper)
            active = [c for c in num_x.columns
                      if pd.notna(lo[c]) and pd.notna(hi[c]) and lo[c] < hi[c]]
            if not active:
                return data
            num_x = num_x.copy()
            num_x[active] = num_x[active].clip(lower=lo[active], upper=hi[active], axis=1)
            return pd.concat(
                [cat_x.reset_index(drop=True), num_x.reset_index(drop=True)], axis=1)
        except Exception as e:
            logger.error(f'Winsorizer.transform failed: {e}')
            return data


# ------------------------------------------------------------------
# ID 28: Z-score clipping
# ------------------------------------------------------------------
class ZScoreClipperPrim(_OutlierPrimBase):
    def __init__(self, random_state=0, k=3.0):
        super(ZScoreClipperPrim, self).__init__(name='ZScoreClipper')
        self.id = 4
        self.gid = 28
        self.hyperparams = []
        self.type = 'outlier handling'
        self.description = (
            "Clips every numeric column to mean +/- 3 standard deviations estimated "
            "on the training split, then applies the same bounds to the test split."
        )
        self.k = k
        self.accept_type = 'a'
        self.need_y = False

    def _apply(self, train_x, test_x, train_y):
        cat_train, num_train = split_cat_and_num_cols(train_x)
        if num_train.shape[1] == 0 or num_train.empty:
            return train_x, test_x, train_y

        cols = list(num_train.columns)
        if not set(cols).issubset(set(test_x.columns)):
            return train_x, test_x, train_y
        num_test = test_x[cols]

        mu = num_train.mean()
        sd = num_train.std(ddof=0)
        active = [c for c in cols if pd.notna(sd[c]) and sd[c] > 0]
        if not active:
            return train_x, test_x, train_y

        lo = mu[active] - self.k * sd[active]
        hi = mu[active] + self.k * sd[active]

        num_train = num_train.copy()
        num_test = num_test.copy()
        num_train[active] = num_train[active].clip(lower=lo, upper=hi, axis=1)
        num_test[active] = num_test[active].clip(lower=lo, upper=hi, axis=1)

        cat_test = test_x[list(cat_train.columns)]
        new_train = pd.concat(
            [cat_train.reset_index(drop=True), num_train.reset_index(drop=True)], axis=1)
        new_test = pd.concat(
            [cat_test.reset_index(drop=True), num_test.reset_index(drop=True)], axis=1)
        return new_train, new_test, train_y

    def transform(self, data: pd.DataFrame, y=None, **kwargs):
        try:
            cat_x, num_x = split_cat_and_num_cols(data)
            if num_x.shape[1] == 0:
                return data
            mu, sd = num_x.mean(), num_x.std(ddof=0)
            active = [c for c in num_x.columns if pd.notna(sd[c]) and sd[c] > 0]
            if not active:
                return data
            num_x = num_x.copy()
            num_x[active] = num_x[active].clip(
                lower=mu[active] - self.k * sd[active],
                upper=mu[active] + self.k * sd[active], axis=1)
            return pd.concat(
                [cat_x.reset_index(drop=True), num_x.reset_index(drop=True)], axis=1)
        except Exception as e:
            logger.error(f'ZScoreClipper.transform failed: {e}')
            return data
