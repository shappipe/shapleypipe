import pandas as pd
import json
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from loguru import logger
from ..datatypes import DatasetType

from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ''))

BASE_DIR = './datasets/cc18'

def list_cc18():
    return os.listdir(BASE_DIR)

def available_cc18():
    return [
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


def cc18_to_Xy(dataset_name: str, factorize=True, force_factorize=False):
    BASE_URL_2 = 'datasets/cc18_2'
    csv_file_path = os.path.join(BASE_URL_2, dataset_name, 'data.csv')
    info_file_path = os.path.join(BASE_URL_2, dataset_name, 'info.json')
    with open(info_file_path, 'r') as f:
        label_name = json.load(f)['label']
    
    data = pd.read_csv(csv_file_path)
    X = data.drop(columns=[label_name])
    y = data[label_name]
    if force_factorize or (factorize and str(y.dtype).lower() in ['object', 'bool']):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=y.name)
    return X, y


def load_cc18(dataset_name: str, factorize=True, force_factorize=False, limit_train=None, limit_all=None):
    dataset_dir = os.path.join(BASE_DIR, dataset_name)
    train_X = pd.read_csv(os.path.join(dataset_dir, 'train_x.csv'))
    train_y: pd.Series = pd.read_csv(os.path.join(dataset_dir, 'train_y.csv')).iloc[:, 0]

    if limit_train is not None and limit_train > 0 and limit_train < len(train_X):
        # train_X = train_X.iloc[:limit_train, :]
        # train_y = train_y.iloc[:limit_train]
        train_X, _, train_y, _ = train_test_split(train_X, train_y, train_size=limit_train, random_state=0)
        train_X = train_X.reset_index(drop=True)
        train_y = train_y.reset_index(drop=True)
    test_X = pd.read_csv(os.path.join(dataset_dir, 'test_x.csv'))
    test_y: pd.Series = pd.read_csv(os.path.join(dataset_dir, 'test_y.csv')).iloc[:, 0]

    if force_factorize or (factorize and str(train_y.dtype).lower() in ['object', 'bool']):
        le = LabelEncoder()
        train_y = pd.Series(le.fit_transform(train_y), name=train_y.name)
        test_y = pd.Series(le.transform(test_y), name=train_y.name)
        logger.info(f'Factorize label into int with {le}')

    if limit_all is not None:
        if type(limit_all) == int and limit_all < len(train_X) + len(test_X):
            rate = limit_all / (len(train_X) + len(test_X))
        elif type(limit_all) == float:
            rate = limit_all
        else:
            raise ValueError(f'Error type of limit_all {limit_all} {type(limit_all)}')
        train_X, _, train_y, _ = train_test_split(train_X, train_y, train_size=rate, random_state=0)
        test_X, _, test_y, _ = train_test_split(test_X, test_y, train_size=rate, random_state=0)

    return DatasetType({
        'train': train_X.reset_index(drop=True),
        'test': test_X.reset_index(drop=True),
        'target': train_y.reset_index(drop=True),
        'target_test': test_y.reset_index(drop=True),
    })

