from collections import Counter
from itertools import product
import os
import sys
import json
from loguru import logger
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ''))
import numpy as np
import pandas as pd
from utils.operators.q_action_provider import QActionProvider
from utils.dataloader.load_cc18 import load_cc18
from utils.dataloader.load_diffprep import name_to_df, ALL_DATASET_NAMES
from sklearn.metrics import accuracy_score
import argparse
from deprecated import deprecated
import random
random.seed(42)
np.set_printoptions(precision=6)

def load_data(dataset_name: str, limit_train=None):
    if dataset_name in ALL_DATASET_NAMES:
        return name_to_df(dataset_name, sample_size=limit_train)
    else:
        return load_cc18(dataset_name, limit_train=limit_train)


def pipe_to_names(pipe: list[int]):
    res = '['
    for i, op in enumerate(pipe):
        if op < 0: 
            res += 'None'
        else:
            res += QActionProvider.get(op).get_name()
        if i < len(pipe) - 1:
            res += ','
    return res + ']'

class InteractionByDatasetOperatorPosition:
    def __init__(self, dataset_name, max_len=6):
        self.dataset_name = dataset_name
        self.data = load_data(dataset_name)
        cache_path = f"saves/cache_25-6/lr/{dataset_name}.json"
        with open(cache_path, 'r') as f:
            self.cache = json.load(f)
        self.max_len = max_len
        string_cols = self.data['train'].select_dtypes(exclude=['number']).columns
        self.has_cat_miss = self.data['train'][string_cols].isnull().any().any()
        self.has_cat = len(string_cols) > 0
        self.has_num_miss = any(self.data['train'].select_dtypes(include=['number']).isnull().any())
        # self.grid = np.load(f'saves/interaction/grids/{dataset_name}.npy')
        
    def evaluate_pipeline_worker(self, pipeline: list[int]):
        # Note: The signature has changed. It now only takes the pipeline.
        # It gets data and dataset_name from global variables.
        try:
            # Crucially, we perform a deep copy *inside the worker*
            # to ensure each task has its own isolated data copy.
            d = {
                'train': self.data['train'].copy(deep=True),
                'target': self.data['target'].copy(deep=True),
                'test': self.data['test'].copy(deep=True),
                'target_test': self.data['target_test'].copy(deep=True),
            }

            caches = {}
            accuracy = 0.
            cumu_pipe = []

            for op in pipeline:
                if op < 0: continue
                cumu_pipe.append(op)
                o = QActionProvider.get(op)
                d['train'], d['test'], d['target'] = o.transform_1(d['train'], d['test'], d['target'])

                # if op <= QActionProvider.spans[2][-1]: 
                #     continue
                key = f'{self.dataset_name}:{cumu_pipe}'
                if key in self.cache:
                    accuracy = float(self.cache[key])
                    if accuracy != 0:
                        continue
                lr = QActionProvider.lr()
                y_pred = lr.transform_1(d['train'], d['target'], d['test'])
                accuracy = float(accuracy_score(d['target_test'], y_pred))
                caches[key] = accuracy
            # The logger will be sparse here as it's from workers, but that's okay.
            # Main process logger will provide batch-level info.
            if len(pipeline) == 0:
                lr = QActionProvider.lr()
                y_pred = lr.transform_1(d['train'], d['target'], d['test'])
                accuracy = float(accuracy_score(d['target_test'], y_pred))
                caches[f'{self.dataset_name}:[]'] = accuracy

            if accuracy == 0:
                print('debugger')

            logger.info(f'evaluate {self.dataset_name} {pipeline} {accuracy}')
            return accuracy, caches
        except Exception as e:
            logger.error(f'Worker evaluation error for pipeline {pipeline}: {e}')
            # It's better to log the exception with traceback in the worker log
            # but for simplicity, we keep it concise here.
            # traceback.print_exc()
            return 0.0, {}

    def get_util(self, pipeline: list[int]):
        p = list(filter(lambda x: x >= 0, pipeline))
        key = f'{self.dataset_name}:{str(p)}'
        if key in self.cache and self.cache[key] != 0:
            return self.cache[key]
        else:
            acc, caches = self.evaluate_pipeline_worker(p)
            self.cache.update(caches)
            return acc
        
    def calc_xi(self, pipeline: list[int] | tuple[int], j: int):
        p = list(pipeline)
        uP = self.get_util(p[:])
        uP_1j = self.get_util(p[:j+1])
        uP_nj = self.get_util(p[:j] + p[j+1:])
        uP_ltj = self.get_util(p[:j])
        xi = abs(uP - uP_1j - uP_nj + uP_ltj)
        return xi

    def on_destroy(self):
        cache_path = f"saves/cache_25-6/lr/{self.dataset_name}.json"
        with open(cache_path, 'w') as f:
            json.dump(self.cache, f)
        logger.info('saved cache')

    def check_pipe(self, pipe: list[int]):
        counter = Counter(pipe)
        if counter.get(23, 0) >= 2: return False  # RandomTreesEmbedding
        if len(self.data['train'].columns) >= 15 and counter.get(23, 0) + counter.get(15, 0) + counter.get(16, 0) >= 2: return False
        
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
    
    def enum_pipelines(self):
        m = self.max_len
        cat_miss_span = [-1]
        num_miss_span = [-1]
        cat_enc_span = [-1]

        if self.has_cat_miss:
            cat_miss_span = QActionProvider.spans[0].copy()
        
        if self.has_cat:
            cat_enc_span = QActionProvider.spans[2].copy()
        
        if self.has_num_miss:
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
                        if self.check_pipe(pipe) and tuple(pipe) not in all_pipelines:
                            all_pipelines.append(tuple(pipe))

        # operator_grids = { op_id: [] for op_id in range(len(QActionProvider.methods)) }

        logger.info(f'{self.dataset_name} with {len(all_pipelines)} pipelines')
        # for pipe_id, p in enumerate(all_pipelines):
        #     values = []
        #     for j in range(m):
        #         if p[j] >= 0:
        #             values.append(self.calc_xi(p, j))
        #         else:
        #             values.append(0)

        operator_position_grid = np.zeros((len(QActionProvider.methods), m))
        for op_id in range(len(QActionProvider.methods)):
            for j in range(m):
                values = []
                for pipe in all_pipelines:
                    if pipe[j] == op_id:
                        values.append(self.calc_xi(pipe, j))
                operator_position_grid[op_id, j] = np.mean(values) if len(values) > 0 else 0
        
        return operator_position_grid

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--max_len', type=int, default=6)

    return parser.parse_args()


def one_method(grid: np.ndarray, max_len, dataset):
    row_result = []
    for col_id in range(max_len):
        sel_op_id = np.abs(grid[:, col_id]).argmax()
        row_result.append({
            'pos': col_id,
            'operator_id': sel_op_id,
            'interaction_value': abs(grid[sel_op_id, col_id])
        })
    
    base_path = 'saves/interaction/max_inter_by_op'
    os.makedirs(base_path, exist_ok=True)
    pd.DataFrame(row_result).to_csv(os.path.join(base_path, f'{dataset}.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    args = parse_args()
    logger.add(f'saves/interaction/logs/interaction.log', rotation='10MB')
    for dataset in args.datasets:
        e = InteractionByDatasetOperatorPosition(dataset, args.max_len)
        grid = e.enum_pipelines()
        # version 2
        # one_method(grid, args.max_len, dataset)
        # version 1
        os.makedirs('saves/interaction/inter_grid_by_op_j', exist_ok=True)
        np.save(f'saves/interaction/inter_grid_by_op_j/{dataset}.npy', grid)
