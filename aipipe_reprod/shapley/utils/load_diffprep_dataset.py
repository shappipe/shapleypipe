import json
import os
from typing import TypedDict

class DiffPrepDatasetMeta(TypedDict):
    name: str
    path: str
    label: str

def load_diffprep_meta():
    dataset_names = sorted(os.listdir('datasets/diffprep_dataset'))
    meta: list[DiffPrepDatasetMeta] = []
    for dataset_name in dataset_names:
        # ... (same as before)
        dataset_path = os.path.join('datasets/diffprep_dataset', dataset_name, 'data.csv')
        info_path = os.path.join('datasets/diffprep_dataset', dataset_name, 'info.json')
        with open(info_path, 'r') as f: info = json.load(f)
        label = info['label']
        meta.append({'name': dataset_name, 'path': dataset_path, 'label': label})
    return meta
