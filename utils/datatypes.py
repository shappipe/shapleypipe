from typing import Literal, Union, Dict, List, Tuple, Optional, TypedDict
import pandas as pd

# typedefs
class DatasetType(TypedDict):
    train: pd.DataFrame
    test: pd.DataFrame
    target: pd.Series
    target_test: pd.Series
