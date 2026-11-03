from itertools import product
import sys
import os
import json
import time
from dotenv import load_dotenv
from typing import TypedDict
from loguru import logger
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ''))

from utils.operators.q_action_provider import QActionProvider
from utils.datatypes import DatasetType
import random
random.seed(42)
np.random.seed(42)

DATA_META_PATH = 'pretrain_mab/fine_meta.json'
SAVE_DIR = 'saves/mab_pretrain'

class Metadata(TypedDict):
    folder: str
    file: str
    label_idx: int


def get_all_dataset_meta() -> list[Metadata]:
    with open(DATA_META_PATH, 'r') as f:
        meta = json.load(f)

    return list(map(
        lambda x: {'folder': x['dataset'], 
                   'file': x['csv_file'], 
                   'label_idx': int(x['label'])}, 
                   meta.values()))


def read_dataset(meta: Metadata):
    p = os.path.join('datasets', 'haipipe', meta['folder'], meta['file'])
    df = pd.read_csv(p)
    label = df.columns[meta['label_idx']]
    X = df.drop(columns=[label])
    y = df[label]
    return X, y


def split_dataset(X, y) -> DatasetType:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return {'train':X_train, 
            'test': X_test, 
            'target': y_train, 
            'target_test': y_test}


def check_pipe(X: pd.DataFrame, pipe: list[int]):
    counter = Counter(pipe)
    if counter.get(23, 0) >= 2: return False  # RandomTreesEmbedding
    if len(X.columns) >= 15 and counter.get(23, 0) + counter.get(15, 0) + counter.get(16, 0) >= 2: return False 
    
    for i in range(len(pipe)-1):
        if pipe[i] == pipe[i+1] and (pipe[i] in {24, 7, 8, 9, 10, 13}): 
            return False
        if pipe[i] in {7, 8, 9, 10} and pipe[i+1] in {7, 8, 9, 10}: 
            return False
        if pipe[i] in {23,15,16} and pipe[i+1] in {23, 15, 16}: 
            return False
        if pipe[i] in {17, 18, 19, 20, 21, 22} and pipe[i+1] in {17, 18, 19, 20, 21, 22}:
            return False

    return True


def enum_filled_valid_pipelines(X: pd.DataFrame, max_len: int):
    m = max_len
    cat_miss_span = [-1]
    num_miss_span = [-1]
    cat_enc_span = [-1]
    has_cat_miss = any(X[col].isnull().any() for col in X.select_dtypes(include=['object']).columns.tolist())
    has_cat = len(X.select_dtypes(include=['object']).columns.tolist()) > 0
    has_num_miss = any(X.select_dtypes(include=['number']).isnull().any())

    if has_cat_miss:
        cat_miss_span = QActionProvider.spans[0].copy()
    
    if has_cat:
        cat_enc_span = QActionProvider.spans[2].copy()
    
    if has_num_miss:
        num_miss_span = QActionProvider.spans[1].copy()
    

    feature_spans = [*QActionProvider.spans[3], 
                        *QActionProvider.spans[4], 
                        *QActionProvider.spans[5]]
    
    all_pipelines = []

    for cat_miss_op in cat_miss_span:
        for cat_enc_op in cat_enc_span:
            for num_miss_op in num_miss_span:
                base_pipe = [cat_miss_op, num_miss_op, cat_enc_op]
                for combo in product(feature_spans, repeat=m-3):
                    pipe = [*base_pipe, *combo]
                    if check_pipe(X, pipe) and tuple(pipe) not in all_pipelines:
                        all_pipelines.append(tuple(pipe))

    return list(map(lambda x: list(filter(lambda y: y >= 0, x)), all_pipelines))


def copy_data(data: DatasetType) -> DatasetType:
    return {
        'train': data['train'].copy(),
        'test': data['test'].copy(),
        'target': data['target'].copy(),
        'target_test': data['target_test'].copy()
    }


def evaluate_pipeline_worker(dataset_name: str, data: DatasetType, 
                             pipeline: list[int], pipe_id: int, caches={}):
    try:
        d = copy_data(data)
        accuracy = 0.
        cumu_pipe = []

        for op in pipeline:
            if op < 0: continue
            cumu_pipe.append(op)

            # if op <= QActionProvider.spans[2][-1]: 
            #     continue
            key = f'{dataset_name}:{cumu_pipe}'
            if key in caches:
                accuracy = float(caches[key])
                if accuracy != 0:
                    continue
            o = QActionProvider.get(op)
            d['train'], d['test'], d['target'] = o.transform_1(d['train'], d['test'], d['target'])
            lr = QActionProvider.lr()
            y_pred = lr.transform_1(d['train'], d['target'], d['test'])
            accuracy = float(accuracy_score(d['target_test'], y_pred))
            logger.info(f'{pipe_id:>4d} evaluate {dataset_name} {cumu_pipe} {accuracy}')
            caches[key] = accuracy
        # The logger will be sparse here as it's from workers, but that's okay.
        # Main process logger will provide batch-level info.
        if len(pipeline) == 0:
            if f'{dataset_name}:[]' in caches:
                accuracy = float(caches[f'{dataset_name}:[]'])
            else:
                lr = QActionProvider.lr()
                y_pred = lr.transform_1(d['train'], d['target'], d['test'])
                accuracy = float(accuracy_score(d['target_test'], y_pred))
                caches[f'{dataset_name}:{pipeline}'] = accuracy
                logger.info(f'{pipe_id:>4d} evaluate {dataset_name} {pipeline} {accuracy}')

        return accuracy, caches
    except Exception as e:
        logger.error(f'{pipe_id:>4d} Worker evaluation error for pipeline {pipeline}: {e}')
        # It's better to log the exception with traceback in the worker log
        # but for simplicity, we keep it concise here.
        # traceback.print_exc()
        return 0.0, caches


def get_cache_of_one_dataset(meta: Metadata, max_len: int, sample_cnt=50):
    X, y = read_dataset(meta)
    data = split_dataset(X, y)
    all_pipelines = random.sample(enum_filled_valid_pipelines(X, max_len), k=sample_cnt)
    if os.path.exists(os.path.join(SAVE_DIR, f'{meta["folder"]}.json')):
        with open(os.path.join(SAVE_DIR, f'{meta["folder"]}.json'), 'r') as f:
            caches = json.load(f)
        logger.info(f'Loaded cache for {meta["folder"]}')
    else:
        caches = {}
    rewards = []
    for pid, pipe in enumerate(all_pipelines):
        for pos in range(len(pipe)):
            target_pipe = pipe[:]
            base_pipe = pipe[:pos] + pipe[pos+1:]

            acc_target, caches = evaluate_pipeline_worker(meta['folder'], data, target_pipe, pid, caches)
            acc_base, caches = evaluate_pipeline_worker(meta['folder'], data, base_pipe, pid, caches)

            reward = acc_target - acc_base
            rewards.append({
                'dataset_folder': meta['folder'],
                'dataset_file': meta['file'],
                'dataset_label_idx': meta['label_idx'],
                'op': pipe[pos],
                'reward': reward,
            })

    return rewards, caches


def get_all_dataset_cache_and_reward():
    metas = get_all_dataset_meta()
    rewards = []
    os.makedirs(SAVE_DIR, exist_ok=True)
    time_start = time.time()
    try:
        for meta in metas:
            try:
                r, c = get_cache_of_one_dataset(meta, 6)
                rewards.extend(r)
                with open(os.path.join(SAVE_DIR, f'{meta["folder"]}.json'), 'w') as f:
                    json.dump(c, f)
                pd.DataFrame(rewards).to_csv(os.path.join(SAVE_DIR, 'rewards.csv'), index=False)
            except Exception as e:
                logger.error(f'Error processing {meta["folder"]}/{meta["file"]}: {e}')
    
    finally:
        pd.DataFrame(rewards).to_csv(os.path.join(SAVE_DIR, 'rewards.csv'), index=False)
        logger.info(f'Finished in {time.time() - time_start:.2f}s')
    
    return rewards


if __name__ == '__main__':
    get_all_dataset_cache_and_reward()

