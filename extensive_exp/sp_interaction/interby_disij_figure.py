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


class InteractionEvaluator:
    def __init__(self, dataset_name, max_len=6, limit_train=None, mcts=False):
        self.dataset_name = dataset_name
        self.data = load_data(dataset_name)
        self.limit_train = limit_train
        self.mcts = mcts

        cache_path = f"saves/cache_25-6/lr/{dataset_name}.json"
        with open(cache_path, 'r') as f:
            self.cache = json.load(f)
        
        string_cols = self.data['train'].select_dtypes(exclude=['number']).columns
        self.has_cat_miss = self.data['train'][string_cols].isnull().any().any()
        self.has_cat = len(string_cols) > 0
        self.has_num_miss = any(self.data['train'].select_dtypes(include=['number']).isnull().any())
        self.max_len = max_len

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
    
    @deprecated
    def enum_all_valid_pipelines(self, given_pipe: list[int], j: int):
        m = self.max_len
        cat_miss_span = [-1]
        num_miss_span = [-1]
        cat_enc_span = [-1]

        if self.has_cat_miss:
            cat_miss_span = QActionProvider.spans[0].copy()
        m -= 1
        if self.has_cat:
            cat_enc_span = QActionProvider.spans[2].copy()
        m -= 1
        if self.has_num_miss:
            num_miss_span = QActionProvider.spans[1].copy()
        m -= 1

        feature_spans = [*QActionProvider.spans[3], 
                         *QActionProvider.spans[4], 
                         *QActionProvider.spans[5]]
        
        all_pipelines = []

        for cat_miss_op in cat_miss_span:
            for cat_enc_op in cat_enc_span:
                for num_miss_op in num_miss_span:
                    base_pipe = [cat_miss_op, num_miss_op, cat_enc_op]
                    for r in range(m + 1):
                        if r == 0:
                            res = list(filter(lambda x: x != -1, base_pipe))
                            if len(res) > j:
                                res[:j] = given_pipe[:j]
                                if res not in all_pipelines:
                                    all_pipelines.append(tuple(res))
                            continue

                        for combo in product(feature_spans, repeat=r):
                            counter = Counter(combo)
                            if counter.get(23, 0) >= 2: continue  # RandomTreesEmbedding
                            if len(self.data['train'].columns) >= 15 and counter.get(23, 0) + counter.get(15, 0) + counter.get(16, 0) >= 2: continue 

                            flag = False
                            res = list(filter(lambda x: x != -1, base_pipe + list(combo)))
                            if len(res) > j:
                                res[:j] = given_pipe[:j]
                                for i in range(len(res)-1):
                                    if res[i] == res[i+1] and (res[i] in {24, 7, 8, 9, 10, 13}): 
                                        flag = True
                                        break
                                    if res[i] in {7, 8, 9, 10} and res[i+1] in {7, 8, 9, 10}: 
                                        flag = True
                                        break
                                    if res[i] in {23} and res[i+1] in {23, 15, 16}:
                                        flag = True
                                        break
                                    if res[i] in {17, 18, 19, 20, 21, 22} and res[i+1] in {17, 18, 19, 20, 21, 22}:
                                        flag = True
                                        break
                                
                                if len(set(res) & set(QActionProvider.spans[1])) > 1 or len(set(res) & set(QActionProvider.spans[2])) > 1:
                                    continue
                                if flag: continue
                                if res not in all_pipelines:
                                    all_pipelines.append(tuple(res))

        return all_pipelines

    def enum_filled_valid_pipelines(self):
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

        logger.info(f'Totally {len(all_pipelines)} pipelines')
        grid = np.zeros((len(all_pipelines), self.max_len))
        for i, p in enumerate(all_pipelines):
            for j in range(self.max_len):
                grid[i, j] = self.calc_xi(p, j)

        return grid

    @deprecated
    def mcts_any_pipelines(self):
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
                    k = 0
                    base_pipe = list(filter(lambda x: x >= 0, [cat_miss_op, num_miss_op, cat_enc_op]))
                    for guard in range(1000):
                        combo = np.random.choice(feature_spans, size=m-len(base_pipe)).tolist()
                    # for combo in product(feature_spans, repeat=m-len(base_pipe)):
                        pipe = [*base_pipe, *combo]
                        if self.check_pipe(pipe) and tuple(pipe) not in all_pipelines:
                            all_pipelines.append(tuple(pipe))
                            k += 1
                            if k >= 100:
                                break

        logger.info(f'Totally {len(all_pipelines)} pipelines')
        if self.mcts:
            all_pipelines = random.sample(all_pipelines, 400)
        grid = np.zeros((len(all_pipelines), self.max_len))
        for i, p in enumerate(all_pipelines):
            for j in range(self.max_len):
                grid[i, j] = self.calc_xi(p, j)

        return grid

    def check_all_pipes(self, pipe: list[int]):
        pipes = []
        for j in range(self.max_len):
            pipes.extend([pipe[:], pipe[:j+1], pipe[:j] + pipe[j+1:], pipe[:j]])

        for p in pipes:
            if not self.check_pipe(p):
                return False
        return True

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

    @deprecated
    def enum_mcts_pipelines(self, given_pipe: list[int], j: int):
        m = self.max_len
        cat_miss_span = [-1]
        num_miss_span = [-1]
        cat_enc_span = [-1]
        target_op = given_pipe[j]

        if self.has_cat_miss:
            cat_miss_span = QActionProvider.spans[0].copy()
        m -= 1
        if self.has_cat:
            cat_enc_span = QActionProvider.spans[2].copy()
        m -= 1
        if self.has_num_miss:
            num_miss_span = QActionProvider.spans[1].copy()
        m -= 1

        feature_spans = [*QActionProvider.spans[3], 
                         *QActionProvider.spans[4], 
                         *QActionProvider.spans[5]]
        
        k = 0
        repeat = 0
        for cat_miss_op in cat_miss_span:
            for num_miss_op in num_miss_span:
                for cat_enc_op in cat_enc_span:
                    base_pipe = [cat_miss_op, num_miss_op, cat_enc_op]
                    while k < 100 and repeat < 10000:
                        repeat += 1
                        remaining_len = np.random.choice(m+1)
                        if remaining_len == 0:
                            res = list(filter(lambda x: x != -1, base_pipe))
                            if len(res) > j:
                                res[:j] = given_pipe[:j]
                                k += 1
                                yield res
                        else:
                            remaining_ops = np.random.choice(feature_spans, size=remaining_len, replace=True).tolist()
                            
                            counter = Counter(remaining_ops)
                            if counter.get(23, 0) >= 2: continue  # RandomTreesEmbedding
                            if len(self.data['train'].columns) >= 15 and counter.get(23, 0) + counter.get(15, 0) + counter.get(16, 0) >= 2: continue 
                            
                            res = list(filter(lambda x: x != -1, base_pipe + list(remaining_ops)))
                            if len(res) > j:
                                res[:j] = given_pipe[:j]
                                
                                flag = False
                                for i in range(len(res)-1):
                                    if res[i] == res[i+1] and (res[i] in {24, 7, 8, 9, 10, 13}): 
                                        flag = True
                                        break
                                    if res[i] in {7, 8, 9, 10} and res[i+1] in {7, 8, 9, 10}: 
                                        flag = True
                                        break
                                    if res[i] in {23} and res[i+1] in {23, 15, 16}: 
                                        flag = True
                                        break
                                    if res[i] in {17, 18, 19, 20, 21, 22} and res[i+1] in {17, 18, 19, 20, 21, 22}:
                                        flag = True
                                        break
                                
                                if len(set(res) & set(QActionProvider.spans[1])) > 1 or len(set(res) & set(QActionProvider.spans[2])) > 1:
                                    continue
                                if flag: continue
                                k += 1
                                yield res

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

    @deprecated
    def calc_e_xi(self, j: int, given_pipe: list[int]):
        xi_list = []
        if self.mcts:
            pipelines = list(self.enum_mcts_pipelines(given_pipe, j))
        else:
            pipelines = list(self.enum_all_valid_pipelines(given_pipe, j))
        logger.info(f'Totally {len(pipelines)} pipelines')
        for p in pipelines:
            xi_list.append(self.calc_xi(list(p), j))
        logger.info(f'len={len(xi_list)}')
        e_xi = float(np.mean(xi_list))
        return e_xi, xi_list

    @deprecated
    def calc_ee_xi(self, given_pipe: list[int]):
        xi_list = []
        for j in range(len(given_pipe)):
            e_xi, xi_list_j = self.calc_e_xi(j, given_pipe)
            xi_list.append(e_xi)
        return xi_list

    def on_destroy(self):
        cache_path = f"saves/cache_25-6/lr/{self.dataset_name}.json"
        with open(cache_path, 'w') as f:
            json.dump(self.cache, f)
        logger.info('saved cache')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--max_len', type=int, default=6)
    parser.add_argument('--limit_train', type=int, default=None)
    parser.add_argument('--mcts', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger.add(f'saves/interaction/{"mcts_" if args.mcts else ""}logs/{args.dataset}.log', rotation='10MB')
    evaluator = InteractionEvaluator(args.dataset, args.max_len, args.limit_train, args.mcts)
    try:
        if not args.mcts:
            grid_xi = evaluator.enum_filled_valid_pipelines()
        else:
            grid_xi = evaluator.mcts_any_pipelines()
        os.makedirs(f'saves/interaction/{"mcts_" if args.mcts else ""}grids', exist_ok=True)
        np.save(os.path.join(f'saves/interaction/{"mcts_" if args.mcts else ""}grids', f'{args.dataset}.npy'), grid_xi)
        os.makedirs(f'saves/interaction/{"mcts_" if args.mcts else ""}values', exist_ok=True)
        avg_xi = grid_xi.mean(axis=0)
        np.save(os.path.join(f'saves/interaction/{"mcts_" if args.mcts else ""}values', f'{args.dataset}.npy'), avg_xi)
    finally:
        evaluator.on_destroy()

