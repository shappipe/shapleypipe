import pandas as pd
import random

def clamped_df(df: pd.DataFrame, max_rows=1000):
    if df.shape[0] > max_rows:
        return df.head(max_rows)
    else:
        return df

def safe_div(numerator: float, denominator: float, default=0.):
    if denominator == 0:
        return default
    else:
        return numerator / denominator

def rand_int(start, stop):
    '''
    Returns a random integer between [start, stop).
    '''
    return start + int(random.random() * (stop - start))
