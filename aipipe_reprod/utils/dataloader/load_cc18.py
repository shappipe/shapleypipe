import pandas as pd
import os
import sys
from ...shapley.datatypes import DatasetType

from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR'))

BASE_DIR = './datasets/cc18'

def list_cc18():
    return os.listdir(BASE_DIR)

def load_cc18(dataset_name: str):
    dataset_dir = os.path.join(BASE_DIR, dataset_name)
    train_X = pd.read_csv(os.path.join(dataset_dir, 'train_x.csv'))
    train_y: pd.Series = pd.read_csv(os.path.join(dataset_dir, 'train_y.csv')).iloc[:, 0]
    train_y = pd.Series(pd.factorize(train_y)[0])
    test_X = pd.read_csv(os.path.join(dataset_dir, 'test_x.csv'))
    test_y: pd.Series = pd.read_csv(os.path.join(dataset_dir, 'test_y.csv')).iloc[:, 0]
    test_y = pd.Series(pd.factorize(test_y)[0])
    return DatasetType({
        'train': train_X,
        'test': test_X,
        'target': train_y,
        'target_test': test_y,
    })

