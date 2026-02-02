from typing import Literal, Union, Dict, TypedDict
import pandas as pd

# DatasetType = Dict[Literal['train', 'test', 'target', 'target_test'], Union[pd.DataFrame, Dict[str, pd.DataFrame]]]

class DatasetType(TypedDict):
    train: pd.DataFrame
    test: pd.DataFrame
    target: pd.Series
    target_test: pd.Series
