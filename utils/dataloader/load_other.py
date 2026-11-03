import os
import numpy as np
import pandas as pd
from .load_diffprep import basic_preprocess


def load_other_dataset(name, t='numpy', sample_size=None, preprocess=True):
    '''
    name: credit, diabetes, employee
    '''
    BASE_PATH = '/home/cj/projects/autoprep/datasets/others'
    df = pd.read_csv(os.path.join(BASE_PATH, f'{name}.csv'))
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    X = df.iloc[:, :-1]
    if preprocess:
        X = basic_preprocess(X)
    y = pd.Series(pd.factorize(df.iloc[:, -1])[0])
    if t == 'numpy':
        X = X.to_numpy()
        y = y.to_numpy()
    return X, y

def inject_nan(X: np.ndarray, nan_rate: float = 0.1, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    mask = rng.random(X.shape) < nan_rate
    X = X.astype(float)
    X[mask] = np.nan
    return X
