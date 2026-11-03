import json
import sys
import os
import openml
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ''))

datasetIds = [
    # (37,"Diabetes","binary",5),
    # (23,"CMC","multiclass",3), 
    # (50,"Tic-Tac-Toe","binary",6),        
    # # (42343,"KDD98","binary",210),    
    # (1509,"Walking-Activity","multiclass",20),
    # (1476,"Gas-Drift","multiclass",34),
    # (41166,"Volkert","multiclass",35)
    (41672, 'airlines', 'binary', 9),
    (4532, 'higgs', 'binary', 10),
    (150, 'covertype', 'multiclass', 10),
]

for dataset_id, dataset_name, task_type, dataset_index in datasetIds:
    print(f'Downloading dataset {dataset_name}, id={dataset_id}')
    dataset = openml.datasets.get_dataset(dataset_id, download_qualities=False)
    dataset_description = dataset.description
    data, y, categorical_indicator, attribute_names = dataset.get_data()
    target_attribute = dataset.default_target_attribute

    n_classes = data[dataset.default_target_attribute].nunique() # type:ignore

    print(dataset_name)
    print(f'{n_classes=}')
    print(f'{data.shape=}')
    
    df = pd.DataFrame(data) # type:ignore

    missing_cnt = df.isna().sum().sum()
    print(f'{missing_cnt=}')
    
    num_cols_without_na = df.select_dtypes(include=['number']).dropna(axis=1).columns
    Xa = df[num_cols_without_na]
    mean = Xa.mean()
    std = Xa.std()

    outlier_cnt = ((Xa > mean + 3 * std) | (Xa < mean - 3 * std)).sum().sum()
    print(f'{outlier_cnt=}')

    metadata = {
        'dataset_name': dataset_name,
        'n_classes': n_classes,
        'label': dataset.default_target_attribute
    }

    os.makedirs(f'datasets/cc18/{dataset_name}', exist_ok=True)
    df.to_csv(f'datasets/cc18/{dataset_name}/data.csv', index=False)
    with open(f'datasets/cc18/{dataset_name}/info.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    train_x, test_x, train_y, test_y = train_test_split(df.drop(columns=[target_attribute]), df[target_attribute], test_size=0.2, random_state=0)
    train = pd.concat([train_x, train_y], axis=1)
    test = pd.concat([test_x, test_y], axis=1)
    train.to_csv(os.path.join(f'datasets/cc18/{dataset_name}', 'train.csv'), index=False)
    test.to_csv(os.path.join(f'datasets/cc18/{dataset_name}', 'test.csv'), index=False)
    train_x.to_csv(os.path.join(f'datasets/cc18/{dataset_name}', 'train_x.csv'), index=False)
    test_x.to_csv(os.path.join(f'datasets/cc18/{dataset_name}', 'test_x.csv'), index=False)
    train_y.to_csv(os.path.join(f'datasets/cc18/{dataset_name}', 'train_y.csv'), index=False)
    test_y.to_csv(os.path.join(f'datasets/cc18/{dataset_name}', 'test_y.csv'), index=False)
