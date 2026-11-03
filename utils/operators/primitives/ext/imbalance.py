'''
Imbalance Handling
'''

import os

import numpy as np
import pandas as pd
from loguru import logger

from sklearn.neighbors import NearestNeighbors

from ..primitive import Primitive


MAX_ROWS_MULT = float(os.getenv('SHAPLEY_IMB_MAX_MULT', '3.0'))
MAX_ABS_ROWS = int(os.getenv('SHAPLEY_IMB_MAX_ROWS', '200000'))
IMBALANCE_MIN_RATIO = float(os.getenv('SHAPLEY_IMB_MIN_RATIO', '1.5'))
SMOTE_MAX_FEAT = int(os.getenv('SHAPLEY_SMOTE_MAX_FEAT', '500'))


def _prepare(train_x: pd.DataFrame, train_y: pd.Series):
    if train_x is None or train_x.empty or train_x.shape[1] == 0:
        return None, None, False
    y_arr = np.asarray(train_y)
    if len(y_arr) != len(train_x) or len(y_arr) < 10:
        return None, None, False

    labels, counts = np.unique(y_arr, return_counts=True)
    if len(labels) < 2:
        return None, None, False
    if counts.max() / max(counts.min(), 1) < IMBALANCE_MIN_RATIO:
        return None, None, False
    return y_arr, dict(zip(labels.tolist(), counts.tolist())), True


def _row_budget(n_now: int):
    return int(min(n_now * MAX_ROWS_MULT, MAX_ABS_ROWS))


def _rebuild(train_x: pd.DataFrame, idx: np.ndarray, y_arr: np.ndarray,
             extra_x: np.ndarray = None, extra_y: np.ndarray = None,
             num_cols=None, y_name=None):
    base_x = train_x.iloc[idx].reset_index(drop=True)
    base_y = pd.Series(y_arr[idx]).reset_index(drop=True)

    if extra_x is not None and len(extra_x) > 0:
        syn = pd.DataFrame(extra_x, columns=num_cols)
        base_x = pd.concat([base_x, syn], axis=0, ignore_index=True)
        base_y = pd.concat([base_y, pd.Series(extra_y)], axis=0, ignore_index=True)

    base_y.name = y_name
    return base_x.reset_index(drop=True), base_y.reset_index(drop=True)


class _ImbalancePrimBase(Primitive):
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
# ID 29: SMOTE
# ------------------------------------------------------------------
class SMOTEPrim(_ImbalancePrimBase):
    def __init__(self, random_state=0, k_neighbors=5):
        super(SMOTEPrim, self).__init__(name='SMOTE')
        self.id = 1
        self.gid = 29
        self.hyperparams = []
        self.type = 'imbalance handling'
        self.description = (
            "Synthetic Minority Over-sampling: interpolates new minority-class rows "
            "between a sample and its k nearest same-class neighbours. Training split only."
        )
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.accept_type = 'c'
        self.need_y = True

    def _apply(self, train_x, test_x, train_y):
        y_arr, counts, ok = _prepare(train_x, train_y)
        if not ok:
            return train_x, test_x, train_y

        num_x = train_x.select_dtypes('number')
        if num_x.shape[1] != train_x.shape[1] or num_x.shape[1] == 0:
            return train_x, test_x, train_y
        mat = num_x.to_numpy(dtype=float, copy=True)
        if not np.isfinite(mat).all():
            return train_x, test_x, train_y
        if mat.shape[1] > SMOTE_MAX_FEAT:
            return RandomOverSamplerPrim(self.random_state)._apply(train_x, test_x, train_y)

        rng = np.random.RandomState(self.random_state)
        n_now = len(train_x)
        budget = _row_budget(n_now) - n_now
        if budget <= 0:
            return train_x, test_x, train_y

        target = max(counts.values())
        cols = list(num_x.columns)
        syn_x, syn_y = [], []

        for label, cnt in sorted(counts.items(), key=lambda kv: kv[1]):
            need = target - cnt
            if need <= 0 or budget <= 0:
                continue
            need = int(min(need, budget))

            member_idx = np.where(y_arr == label)[0]
            if len(member_idx) < 2:
                continue
            sub = mat[member_idx]
            k = int(min(self.k_neighbors, len(member_idx) - 1))
            if k < 1:
                continue

            nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=1).fit(sub)
            _, nbr = nn.kneighbors(sub)         

            base = rng.randint(0, len(sub), size=need)
            pick = rng.randint(1, k + 1, size=need)
            partner = nbr[base, pick]
            gap = rng.rand(need, 1)
            syn = sub[base] + gap * (sub[partner] - sub[base])

            syn_x.append(syn)
            syn_y.append(np.full(need, label, dtype=y_arr.dtype))
            budget -= need

        if not syn_x:
            return train_x, test_x, train_y

        extra_x = np.vstack(syn_x)
        extra_y = np.concatenate(syn_y)
        new_x, new_y = _rebuild(
            train_x, np.arange(n_now), y_arr,
            extra_x=extra_x, extra_y=extra_y,
            num_cols=cols, y_name=getattr(train_y, 'name', None))
        return new_x, test_x, new_y


# ------------------------------------------------------------------
# ID 30: RandomUnderSampler
# ------------------------------------------------------------------
class RandomUnderSamplerPrim(_ImbalancePrimBase):
    def __init__(self, random_state=0, floor_ratio=0.2):
        super(RandomUnderSamplerPrim, self).__init__(name='RandomUnderSampler')
        self.id = 2
        self.gid = 30
        self.hyperparams = []
        self.type = 'imbalance handling'
        self.description = (
            "Randomly down-samples majority classes toward the minority class size. "
            "A floor keeps the training split from collapsing. Training split only."
        )
        self.random_state = random_state
        self.floor_ratio = floor_ratio
        self.accept_type = 'a'
        self.need_y = True

    def _apply(self, train_x, test_x, train_y):
        y_arr, counts, ok = _prepare(train_x, train_y)
        if not ok:
            return train_x, test_x, train_y

        rng = np.random.RandomState(self.random_state)
        n_now = len(train_x)
        minority = min(counts.values())
        target = max(minority, int(np.ceil(n_now * self.floor_ratio / len(counts))))

        keep = []
        for label, cnt in counts.items():
            member_idx = np.where(y_arr == label)[0]
            if cnt <= target:
                keep.append(member_idx)
            else:
                keep.append(rng.choice(member_idx, size=target, replace=False))
        idx = np.sort(np.concatenate(keep))

        if len(idx) < max(10, int(n_now * 0.1)):
            return train_x, test_x, train_y

        new_x, new_y = _rebuild(train_x, idx, y_arr,
                                y_name=getattr(train_y, 'name', None))
        return new_x, test_x, new_y


# ------------------------------------------------------------------
# ID 31: RandomOverSampler
# ------------------------------------------------------------------
class RandomOverSamplerPrim(_ImbalancePrimBase):
    def __init__(self, random_state=0):
        super(RandomOverSamplerPrim, self).__init__(name='RandomOverSampler')
        self.id = 3
        self.gid = 31
        self.hyperparams = []
        self.type = 'imbalance handling'
        self.description = (
            "Randomly duplicates minority-class rows toward the majority class size, "
            "under a hard row budget. Training split only."
        )
        self.random_state = random_state
        self.accept_type = 'a'
        self.need_y = True

    def _apply(self, train_x, test_x, train_y):
        y_arr, counts, ok = _prepare(train_x, train_y)
        if not ok:
            return train_x, test_x, train_y

        rng = np.random.RandomState(self.random_state)
        n_now = len(train_x)
        budget = _row_budget(n_now) - n_now
        if budget <= 0:
            return train_x, test_x, train_y

        target = max(counts.values())
        idx_parts = [np.arange(n_now)]
        for label, cnt in sorted(counts.items(), key=lambda kv: kv[1]):
            need = target - cnt
            if need <= 0 or budget <= 0:
                continue
            need = int(min(need, budget))
            member_idx = np.where(y_arr == label)[0]
            idx_parts.append(rng.choice(member_idx, size=need, replace=True))
            budget -= need

        idx = np.concatenate(idx_parts)
        if len(idx) == n_now:
            return train_x, test_x, train_y

        new_x, new_y = _rebuild(train_x, idx, y_arr,
                                y_name=getattr(train_y, 'name', None))
        return new_x, test_x, new_y
