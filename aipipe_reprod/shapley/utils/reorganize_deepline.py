import json
import sys
import os
import openml
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR'))

base_dir = 'datasets/deepline_datasets'
csv_names = sorted(os.listdir(base_dir))

for csv_name in csv_names:
    with open(os.path.join(base_dir, csv_name), 'r') as f:
        df = pd.read_csv(f, nrows=5)
    label_name = df.columns[-1]
    meta = { 'label': label_name }
    clean_csv_name = csv_name.replace('.csv', '').replace(' ', '').replace('(', '').replace(')', '')
    os.makedirs(os.path.join(base_dir, clean_csv_name), exist_ok=True)
    os.rename(os.path.join(base_dir, csv_name), os.path.join(base_dir, clean_csv_name, f'data.csv'))
    with open(os.path.join(base_dir, clean_csv_name, 'info.json'), 'w') as f:
        json.dump(meta, f, indent=2)
