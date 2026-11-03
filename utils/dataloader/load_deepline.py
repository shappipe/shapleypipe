import json
import os
from typing import TypedDict

import pandas as pd

deepline_path = '/SSD/00/cj/deepline_dataset'

class Meta(TypedDict):
    name: str
    path: str
    label_idx: int
    label: str

def get_deepline_metadatas() -> list[Meta]:
    metadatas = []
    csv_names = sorted(os.listdir(deepline_path))

    for csv_name in csv_names:
        if csv_name == 'analcatdata_germangss': continue
        csv_path = os.path.join(deepline_path, csv_name, 'data.csv')
        with open(os.path.join(deepline_path, csv_name, 'info.json'), 'r') as f:
            info = json.load(f)
            label = info['label']
        metadata: Meta = {
            'name': csv_name.replace('.csv', ''),
            'path': csv_path,
            'label_idx': -1,
            'label': label
        }
        metadatas.append(metadata)

    return metadatas
