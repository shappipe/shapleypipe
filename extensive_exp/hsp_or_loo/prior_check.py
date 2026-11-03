from collections import Counter, defaultdict
from itertools import product
import os
import sys
import json
import time
from loguru import logger
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ''))
import numpy as np
import pandas as pd
from utils.operators.q_action_provider import QActionProvider
from utils.dataloader.load_cc18 import load_cc18, available_cc18
from utils.dataloader.load_diffprep import name_to_df, ALL_DATASET_NAMES
from sklearn.metrics import accuracy_score
import argparse
from deprecated import deprecated
import random

np.set_printoptions(precision=6)

def load_data(dataset_name: str, limit_train=None, percent=None):
    if dataset_name in ALL_DATASET_NAMES:
        return name_to_df(dataset_name, sample_size=percent)
    else:
        return load_cc18(dataset_name, limit_all=percent)

class PriorCheck:
    def __init__(self, dataset_name, max_len=6, limit_train=None, percent=0.1):
        self.dataset_name = dataset_name
        self.data = load_data(dataset_name, limit_train=limit_train, percent=percent)
        self.limit_train = limit_train
        self.percent = percent

        cache_path = f"extensive_exp/hsp_or_loo/saves/cache_small/lr/{dataset_name}.json"
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

        string_cols = self.data['train'].select_dtypes(exclude=['number']).columns
        self.has_cat_miss = self.data['train'][string_cols].isnull().any().any()
        self.has_cat = len(string_cols) > 0
        self.has_num_miss = any(self.data['train'].select_dtypes(include=['number']).isnull().any())
        self.max_len = max_len

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
        
        return all_pipelines
    
    def partition_pipelines(self, all_pipelines: list[tuple[int, ...]]):
        result = {0: {i: [] for i in range(7, len(QActionProvider.methods))}, 
                  1: {i: [] for i in range(7, len(QActionProvider.methods))}, 
                  2: {i: [] for i in range(7, len(QActionProvider.methods))}}
        for p in all_pipelines:
            sub_p = p[-3:]
            for i, op in enumerate(sub_p):
                result[i][op].append(p)
        
        return result

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
    
    def get_util(self, pipeline: list[int] | tuple[int, ...]):
        p = list(filter(lambda x: x >= 0, pipeline))
        key = f'{self.dataset_name}:{str(p)}'
        if key in self.cache and self.cache[key] != 0:
            return self.cache[key]
        else:
            acc, caches = self.evaluate_pipeline_worker(p)
            self.cache.update(caches)
            return acc

    def calc_single_op_marginal(self, op: int, partitions: dict[int, dict[int, list[tuple[int, ...]]]]):
        res = {}
        for j in partitions:
            pipes_with = partitions[j][op]
            j_offset = len(pipes_with[0]) - 3
            margin = 0.
            for p in pipes_with:
                pipe_without = p[:j+j_offset] + p[j+j_offset+1:]
                margin += self.get_util(p) - self.get_util(pipe_without)
            margin = margin / len(pipes_with) if len(pipes_with) > 0 else margin
            res[j] = margin
        return res
    
    def calc_all_op_marginal(self):
        all_pipelines = self.enum_filled_valid_pipelines()
        partitions = self.partition_pipelines(all_pipelines)
        res = []
        for op in range(7, len(QActionProvider.methods)):
            single_op_margin = self.calc_single_op_marginal(op, partitions)
            std = np.std(list(single_op_margin.values()))
            res.append({'op': op, **single_op_margin, 'std': std})
        
        return res
    
    def shift_x_to_end(self, tup, x):
        indices = [i for i, v in enumerate(tup) if v == x]
        if not indices:
            return [tup] 
        idx = indices[0] 
        result = []
        for shift in range(0, len(tup) - idx):
            lst = list(tup)
            del lst[idx]
            lst.insert(idx + shift, x)
            result.append(tuple(lst))
        return result
    
    def std_of_different_perms_given_base(self, given_base: list[tuple[int, ...]], op: int):
        results = []
        for base in given_base:
            perms = self.shift_x_to_end(base, op)
            utils = [self.get_util(p) for p in perms]
            res = {
                'utils': utils,
                'mean': np.mean(utils),
                'std': np.std(utils) if len(utils) > 1 else 0.,
                'part_rho': np.std(utils) / np.mean(utils) if len(utils) > 1 else 0.
            }
            results.append(res)
        total_results = {
            'mean_part_rho': np.mean([res['part_rho'] for res in results]),
            'res': results,
        }
        return total_results

    def std_of_different_permus(self, all_pipelines: list[tuple[int, ...]], op: int):
        def rule(pipe: tuple[int, ...]):
            if op not in pipe: return False
            if set(pipe[:pipe.index(op)]) & set(range(7,25)): return False
            return True
        candidates = list(filter(rule, all_pipelines))
        sampled = random.sample(candidates, k=min(5, len(candidates)))
        results = []
        for i, p1 in enumerate(sampled):
            perms = self.shift_x_to_end(p1, op)
            utils = [self.get_util(p) for p in perms]
            res = {
                'utils': utils,
                'mean': np.mean(utils),
                'std': np.std(utils) if len(utils) > 1 else 0.,
                'part_rho': np.std(utils) / np.mean(utils) if len(utils) > 1 else 0.
            }
            results.append(res)
        total_results = {
            'mean_part_rho': np.mean([res['part_rho'] for res in results]),
            'res': results,
        }
        return total_results
    
    def all_std(self):
        res = []
        all_pipelines = self.enum_filled_valid_pipelines()
        for op in range(7, len(QActionProvider.methods)):
            r = self.std_of_different_permus(all_pipelines, op)
            res.append({'op': op, **r})
        return res

    def error_from_sample(self, all_pipelines: list[tuple[int, ...]], op: int) -> dict[str, float]:
        pos_to_pipelines = defaultdict(list)
        for p in all_pipelines:
            indices = [i for i, x in enumerate(p) if x == op]
            if len(indices) != 1:  # we want exactly one occurrence for simplicity
                continue
            pos = indices[0]
            # Remove op to get the rest of the pipeline (keeping order)
            other = tuple(x for i, x in enumerate(p) if i != pos)
            pos_to_pipelines[pos].append((p, other))
        
        # Only consider positions that actually appear
        positions = sorted(pos_to_pipelines.keys())
        if len(positions) < 2:
            return {'v1': 0.0, 'v2': 0.0, 'v3': 0.0, 'v4': 0.0, 'v5': 0.0, 'avg': 0.0}
        
        # For each sample (v1..v5), we collect several pair differences and average
        num_samples = 5
        sample_results = []
        
        for _ in range(num_samples):
            differences = []
            # Try to form pairs between early and late positions (e.g., first half vs second half)
            early_positions = positions[:len(positions)//2]
            late_positions = positions[len(positions)//2:]
            if not early_positions or not late_positions:
                continue
            
            # For each pair of positions (early, late), try to find pipelines with identical "other" operators
            for ep in early_positions:
                for lp in late_positions:
                    # Build mapping from "other" tuple to list of pipelines for each position
                    early_by_other = {other: pipe for pipe, other in pos_to_pipelines[ep]}
                    late_by_other = {other: pipe for pipe, other in pos_to_pipelines[lp]}
                    
                    common_others = set(early_by_other.keys()) & set(late_by_other.keys())
                    if not common_others:
                        continue
                    # Pick one random common "other"
                    other = random.choice(list(common_others))
                    pipe_early = early_by_other[other]
                    pipe_late = late_by_other[other]
                    
                    util_early = self.get_util(pipe_early)
                    util_late = self.get_util(pipe_late)
                    differences.append(abs(util_early - util_late))  # positive if early is better
                    break
            
            if differences:
                sample_results.append(np.mean(differences))
            else:
                sample_results.append(0.0)
        
        # Ensure we have exactly 5 values (pad with zeros)
        while len(sample_results) < 5:
            sample_results.append(0.0)
        
        result = {f'v{i+1}': sample_results[i] for i in range(5)}
        result['avg'] = np.mean(sample_results)
        return result

    def all_error(self):
        res = []
        all_pipelines = self.enum_filled_valid_pipelines()
        for op in range(7, len(QActionProvider.methods)):
            r = self.error_from_sample(all_pipelines, op)
            res.append({'op': op, **r})
        return res


if __name__ == '__main__':
    all_std_mean = []
    for dataset_name in [*ALL_DATASET_NAMES, *available_cc18()]:
        random.seed(42)
        pc = PriorCheck(dataset_name)
        start = time.time()
        os.makedirs('extensive_exp/hsp_or_loo/saves/std_a_time', exist_ok=True)
        all_std = pc.all_std()
        df = pd.DataFrame(all_std)
        df.to_csv(f'extensive_exp/hsp_or_loo/saves/std_a_time/{dataset_name}.csv', index=False, mode='a')
        dataset_std = [item['mean_part_rho'] for item in all_std]
        with open(f'extensive_exp/hsp_or_loo/saves/std_a_time/{dataset_name}.csv', mode='a') as f:
            f.write(f'rho={float(np.mean(dataset_std))},time={time.time()-start}\n')

# def calc_all_error():
    # all_mean = []
    # for dataset_name in []:
    #     all_margin = pc.all_error()
    #     os.makedirs('hsp_or_loo/saves/early_late_error', exist_ok=True)
    #     df = pd.DataFrame(all_margin)
    #     df.to_csv(f'hsp_or_loo/saves/early_late_error/{dataset_name}.csv', index=False, mode='a')
    #     for _ in range(5):
    #         all_mean.append({'dataset': dataset_name, 'avg': np.mean(df.iloc[random.sample(range(25-7), k=5), -1])})
    
    # pd.DataFrame(all_mean).to_csv('hsp_or_loo/saves/early_late_error/z_all2.csv', index=False, mode='a')
