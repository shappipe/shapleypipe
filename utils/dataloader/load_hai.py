import dataclasses
import json
import os
import dataclasses_json
import pandas as pd
import sys

from ..seed import RANDOM_SEED, TRAIN_TEST_SPLIT_SEED
from sklearn.model_selection import train_test_split

@dataclasses_json.dataclass_json
@dataclasses.dataclass
class DatasetMeta:
    dataset: str
    label: str
    csv_file: str

sys.path.append('/home/cj/projects/code_analysis')

def load_haipipe_csv_and_labels():
    dataset_dir = '/SSD/00/cj/haipipe_datasets'
    with open('aipipe_reprod/dqn/dataloader/available_datasets.json', 'r') as f:
        available_datasets: list[DatasetMeta] = [DatasetMeta.from_dict(d) for d in json.load(f)]
        csv_files = [os.path.join(dataset_dir, d.dataset, d.csv_file) for d in available_datasets]
        labels = [int(d.label) for d in available_datasets]

    return csv_files, labels

def path_to_df_dict(csv_file: str, label: int):
    d = pd.read_csv(csv_file)
    d = d.dropna(subset=[d.columns[label]])
    if d.shape[0] > 1500:
        d = d.sample(1500, random_state=RANDOM_SEED)
    y = d.iloc[:, label]
    y = pd.Series(pd.factorize(y)[0])
    X_train, X_test, y_train, y_test = train_test_split(
        d.drop(columns=[d.columns[label]]), 
        y, 
        test_size=0.2, random_state=TRAIN_TEST_SPLIT_SEED)
    df = {
        'train': X_train.reset_index(drop=True),
        'test': X_test.reset_index(drop=True),
        'target': y_train.reset_index(drop=True),
        'target_test': y_test.reset_index(drop=True),
    }
    return df
