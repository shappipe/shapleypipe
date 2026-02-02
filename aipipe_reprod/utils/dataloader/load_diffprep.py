import csv
import sys
import datetime
import os
import traceback
import pandas as pd
import json
import numpy as np
import random
import time

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

from ...shapley.datatypes import DatasetType

from sklearn.model_selection import train_test_split
from ..seed import TRAIN_TEST_SPLIT_SEED
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR'))

BASE_DIR = './datasets/diffprep_dataset'

def load_diffprep_csv_and_labels():
    dataset_path = BASE_DIR
    test_set_dirs = sorted(os.listdir(dataset_path))
    test_set_csvs = [os.path.join(dataset_path, d, 'data.csv') for d in test_set_dirs]
    test_set_label_names: list[str] = []
    for d in test_set_dirs:
        with open(os.path.join(dataset_path, d, 'info.json'), 'r') as f:
            test_set_label_names.append(json.load(f)['label'])

    return test_set_dirs, test_set_csvs, test_set_label_names


def basic_preprocess(X: pd.DataFrame):
    string_columns = X.select_dtypes(include=['object']).columns.tolist()

    if X[string_columns].isnull().any().any():
        imp = SimpleImputer(strategy='most_frequent')
        X[string_columns] = imp.fit_transform(X[string_columns])
    
    if len(string_columns) > 0:
        X[string_columns] = X[string_columns].astype('category')
        ord_enc = OrdinalEncoder()
        X[string_columns] = ord_enc.fit_transform(X[string_columns])

    # if X.isna().any().any():
    #     imp = SimpleImputer(strategy='mean')
    #     X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    return X


def name_to_Xy(name: str, t='pandas', sample_size=None, preprocess=False):
    dataset_path = BASE_DIR
    csv_file = os.path.join(dataset_path, name, 'data.csv')
    label = json.load(open(os.path.join(dataset_path, name, 'info.json'), 'r'))['label']
    df = pd.read_csv(csv_file)
    X = df.drop(columns=[label])
    y = df[label]
    y = pd.Series(pd.factorize(y)[0])
    if sample_size is not None and sample_size < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=sample_size, random_state=TRAIN_TEST_SPLIT_SEED)

    if preprocess:
        X = basic_preprocess(X)

    if t == 'pandas':
        return X, y
    elif t == 'numpy':
        return X.values, y.values
    else:
        raise ValueError(f"Unknown type {t}")


def path_to_df_dict(csv_file: str, label: str, limit_train=None):
    df = pd.read_csv(csv_file)
    y = df[label]
    y = pd.Series(pd.factorize(y)[0])
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[label]), y, test_size=0.2, random_state=TRAIN_TEST_SPLIT_SEED)

    if limit_train is not None and limit_train < len(X_train):
        X_train = X_train.sample(n=limit_train, random_state=TRAIN_TEST_SPLIT_SEED)
        y_train = y_train.loc[X_train.index]

    # drop 掉只有1个样本的类别
    valid_classes = y.value_counts()[y.value_counts() > 1].index
    X_train = X_train[y_train.isin(valid_classes)]
    y_train = y_train[y_train.isin(valid_classes)]
    X_test = X_test[y_test.isin(valid_classes)]
    y_test = y_test[y_test.isin(valid_classes)]

    # 将标签转换为从0开始的整数
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    df: DatasetType = {
        'train': X_train.reset_index(drop=True),
        'test': X_test.reset_index(drop=True),
        'target': pd.Series(y_train),
        'target_test': pd.Series(y_test),
    }   
    return df

def path_and_label_idx_to_dict(csv_file: str, label_idx: int):
    df = pd.read_csv(csv_file)
    label = df.columns[label_idx]
    y = df.iloc[:, label_idx]
    y = pd.Series(pd.factorize(y)[0])
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[label]), y, test_size=0.2, random_state=TRAIN_TEST_SPLIT_SEED)
    df: DatasetType = {
        'train': X_train.reset_index(drop=True),
        'test': X_test.reset_index(drop=True),
        'target': y_train.reset_index(drop=True),
        'target_test': y_test.reset_index(drop=True),
    }
    return df
