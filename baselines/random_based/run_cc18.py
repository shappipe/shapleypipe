import os
import json
import random
import pandas as pd
import sys
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import argparse
import warnings
warnings.filterwarnings('ignore')
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ''))
from utils.operators.q_action_provider import QActionProvider
from utils.dataloader.load_cc18 import load_cc18

selected_dataset_names = [
    'cmc',
    'diabetes',
    'jm1',
    'kc1',
    'kc2',
    'pc4',
    'phoneme',
    'qsar-biodeg',
    'vehicle',
]

def preprocess(dataset):
    string_cols = dataset['train'].select_dtypes(include=['object']).columns
    ordinal_encoder = OrdinalEncoder()
    dataset['train'][string_cols] = ordinal_encoder.fit_transform(dataset['train'][string_cols])
    dataset['test'][string_cols] = ordinal_encoder.transform(dataset['test'][string_cols])

    le = LabelEncoder()
    dataset['target'] = le.fit_transform(dataset['target'])
    dataset['target_test'] = le.transform(dataset['target_test'])
    return dataset

def process_one_dataset(dataset_name: str):
    dataset = load_cc18(dataset_name)
    dataset = preprocess(dataset)

    pipe = random.choices(range(len(QActionProvider.methods)), k=5)

    for op_id in pipe:
        op = QActionProvider.get(op_id)
        dataset['train'], dataset['test'], dataset['target'] = op.transform_1(dataset['train'], dataset['test'], dataset['target'])
    
    lr = QActionProvider.lr()
    y_pred = lr.transform_1(dataset['train'], dataset['target'], dataset['test'])
    acc = accuracy_score(dataset['target_test'], y_pred)

    print(f'{dataset_name} acc: {acc}')
    return {'dataset_name': dataset_name, 'acc': acc}

if __name__ == '__main__':
    results = []
    for dataset_name in selected_dataset_names:
        results.append(process_one_dataset(dataset_name))

    os.makedirs('baselines/random_based/saves', exist_ok=True)
    pd.DataFrame(results).to_csv('baselines/random_based/saves/cc18_random_search.csv', index=False, mode='a')
