from collections import Counter
from datetime import datetime
import random
import traceback
from loguru import logger
import numpy as np
import time
from itertools import permutations, product
import psutil
import multiprocessing as mp
from typing import Callable, Optional, TypedDict, List

from tqdm import tqdm
import sys
import os
import json
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ''))

from utils.dataloader.load_diffprep import name_to_df, ALL_DATASET_NAMES
from utils.dataloader.load_cc18 import load_cc18
from utils.dataloader.load_cc18 import list_cc18, load_cc18
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.operators.q_action_provider import QActionProvider
from utils.datatypes import DatasetType

def exec_pipe(dataset_name: str, pipeline: list[int], limit_train=None):
    if dataset_name in ALL_DATASET_NAMES:
        data = name_to_df(dataset_name, limit_train=limit_train)
    else:
        try:
            data = load_cc18(dataset_name, limit_train=limit_train)
        except Exception as e:
            traceback.print_exc()
            raise ValueError('Cannot load dataset ' + dataset_name + ' ' + str(e))

    for op_idx in pipeline:
        if op_idx == -1:
            continue
        op = QActionProvider.get(op_idx)
        data['train'], data['test'], data['target'] = op.transform_1(
            data['train'], data['test'], data['target'])
    
    lr = QActionProvider.lr()
    y_pred = lr.transform_1(data['train'], data['target'], data['test'])
    acc = float(accuracy_score(data['target_test'], y_pred))
    prec = float(precision_score(data['target_test'], y_pred, average='weighted'))
    rec = float(recall_score(data['target_test'], y_pred, average='weighted'))
    f1 = float(f1_score(data['target_test'], y_pred, average='weighted'))

    print(f'{dataset_name} {pipeline} Accuracy : {acc} '
          f'Precision: {prec} '
          f'Recall: {rec} '
          f'F1: {f1} ')

    return acc, prec, rec, f1

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--pipe', nargs='+', type=int)
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()

    exec_pipe(args.dataset, args.pipe, limit_train=args.limit)
