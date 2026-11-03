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
from multiprocessing import Pool, Manager
from tqdm import tqdm
import sys
import os
import json
from dotenv import load_dotenv
from multiprocessing.pool import Pool as ProcessPool



load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ''))

if os.getenv('SHAPLEY_EXT_OPS', '0') != '1':
    raise SystemExit(
        'SHAPLEY_EXT_OPS=1 is not set. This program requires the extended operator library (N=32, K=8)。\n'
        'Please set it first: export SHAPLEY_EXT_OPS=1'
    )

from utils.dataloader.load_diffprep import name_to_df, ALL_DATASET_NAMES
from utils.dataloader.load_cc18 import load_cc18
from utils.dataloader.load_cc18 import list_cc18, load_cc18
# from aipipe_reprod.shapley.q_action_provider_simple import QActionProviderSimple
# from aipipe_reprod.shapley.q_action_provider import QActionProvider
from utils.operators.q_action_provider import QActionProvider
from utils.datatypes import DatasetType
from sklearn.metrics import accuracy_score
from utils.hierarchy.mab import UCB1_Bandit
from utils.operators.primitives.predictors import PredictorPrimitive
from utils.exec_pipe_full import exec_pipe
from utils.operators import op_constraints_ext as OC


np.set_printoptions(precision=4, suppress=True)
OP_USE = 'op32'
MCTS_CNT = 75
MAB_SAMPLES = 1000
DT = 'lr'
SEED = 42

RAND_MAB = False

EXHAUST_ENUM_OP_IN_CATE = False
EARLY_STOP = True
LIMIT_TRAIN = None
LIMIT_ALL = None
NO_DUMP = False
EXEC_FULL = False
CACHE_LABEL=''

dt_dict: dict[str, Callable[[], PredictorPrimitive]] = {
    'lr': lambda: QActionProvider.lr(),
    'rf': lambda: QActionProvider.rf(),
    'gb': lambda: QActionProvider.xg(),
    'mlp': lambda: QActionProvider.mlp(),
}

# --- START: 新增辅助类 ---
class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(mp.get_context())):
    Process = NoDaemonProcess


class NonDaemonicPool(ProcessPool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NonDaemonicPool, self).__init__(*args, **kwargs)


class StatsDict(TypedDict):
    total_evaluations: int
    cache_hits: int
    cache_misses: int
    slot_times: List[float]
    theoretical_evaluations: int


worker_data = None
worker_dataset_name = None
worker_cache = None


def init_worker(data: DatasetType, dataset_name: str, shared_cache: dict):
    """Initializer for each worker process in the pool."""
    global worker_data, worker_dataset_name, worker_cache
    worker_data = data
    worker_dataset_name = dataset_name  
    worker_cache = dict(shared_cache)
    logger.info(f"Worker process {os.getpid()} initialized for dataset {dataset_name}")


# from aipipe_reprod.primitives import ImputerCatPrim, ImputerMeanPrim, LabelEncoderPrim

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

def evaluate_pipeline_worker(pipeline: List[int]):
    # Note: The signature has changed. It now only takes the pipeline.
    # It gets data and dataset_name from global variables.
    global worker_data, worker_dataset_name, worker_cache, DT
    if (k :=f'{worker_dataset_name}:{pipeline}') in worker_cache:
        return float(worker_cache[k]), {}
    try:
        # Crucially, we perform a deep copy *inside the worker*
        # to ensure each task has its own isolated data copy.
        d = {
            'train': worker_data['train'].copy(deep=True),
            'target': worker_data['target'].copy(deep=True),
            'test': worker_data['test'].copy(deep=True),
            'target_test': worker_data['target_test'].copy(deep=True),
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
            # key = f'{worker_dataset_name}:{cumu_pipe}'
            # if key in worker_cache:
            #     accuracy = float(worker_cache[key])
            #     continue
        key = f'{worker_dataset_name}:{cumu_pipe}'
        lr = dt_dict[DT]()  # QActionProvider.lr()
        y_pred = lr.transform_1(d['train'], d['target'], d['test'])
        accuracy = float(accuracy_score(d['target_test'], y_pred))
        caches[key] = accuracy

        if len(pipeline) == 0:
            key = f'{worker_dataset_name}:[]'
            if key in worker_cache:
                accuracy = float(worker_cache[key])
                return accuracy, {}
            lr = dt_dict[DT]()  # QActionProvider.lr()
            y_pred = lr.transform_1(d['train'], d['target'], d['test'])
            accuracy = float(accuracy_score(d['target_test'], y_pred))
            caches[key] = accuracy
        # The logger will be sparse here as it's from workers, but that's okay.
        # Main process logger will provide batch-level info.
        logger.info(f'evaluate {worker_dataset_name} {pipeline} {accuracy} {pipe_to_names(pipeline)}')
        return accuracy, caches
    except Exception as e:
        logger.error(f'Worker evaluation error for pipeline {pipeline}: {e}')
        # It's better to log the exception with traceback in the worker log
        # but for simplicity, we keep it concise here.
        # traceback.print_exc()
        return 0.0, {}


class _BanditUpdate(TypedDict):
    category: int
    arm: int
    reward: float


def check_constraint(pipe: List[int] | tuple[int]):
    counter = Counter(pipe)
    if counter.get(23, 0) + counter.get(15, 0) + counter.get(16, 0) >= 2: 
        return False 
    if (counter.get(17, 0) + counter.get(18, 0) + counter.get(19, 0) 
        + counter.get(20, 0) + counter.get(21, 0) + counter.get(22, 0) >= 3): return False 
    if counter.get(24, 0) >= 3: return False 

    for i in range(len(pipe)-1):
        if pipe[i] == pipe[i+1] and (pipe[i] in {24, 7, 8, 9, 10, 13}): 
            return False
        if pipe[i] in {7, 8, 9, 10} and pipe[i+1] in {7, 8, 9, 10}:
            return False
        if pipe[i] in {23} and pipe[i+1] in {23, 15, 16}: 
            return False
        if pipe[i] in {17, 18, 19, 20, 21, 22} and pipe[i+1] in {17, 18, 19, 20, 21, 22}:
            return False

    return True


def check_constraint_ext(pipe: List[int] | tuple[int], n_feature: int):
    if OC.blocked_counter(pipe, n_feature):
        return False
    if OC.blocked_chain(pipe, strict_chain=True):
        return False
    return True

def eval_cate_pipe_work_enum_ops_in_cate(args: tuple[list[int], dict[int, UCB1_Bandit]]):
    pipeline, bandits = args
    # Note: The signature has changed. It now only takes the pipeline.
    # It gets data and dataset_name from global variables.
    global worker_dataset_name, worker_data #, worker_cache #, worker_data, 
    try:
        ops_to_product = []
        for cate_idx in pipeline:
            if cate_idx < 0:
                continue
            arms = bandits[cate_idx].arms[:]
            ops_to_product.append(arms)
        prds = list(product(*ops_to_product))
        all_pipelines = list(filter(lambda x: check_constraint_ext(x, worker_data['train'].shape[1]), prds))
        logger.info('all pipelines enumerated')
        accs = []
        all_caches = {}
        for concrete_pipeline in all_pipelines:
            acc, caches = evaluate_pipeline_worker(list(concrete_pipeline))
            accs.append(acc)
            all_caches.update(caches)

        avg_acc = sum(accs) / len(accs) if len(accs) > 0 else 0  # max(accs) if len(accs) > 0 else 0.
        logger.info(f'evaluate {worker_dataset_name} category {pipeline} {avg_acc}')
        return {
            'pipeline': pipeline,
            'acc': avg_acc,
            'bandit_updates': [],
            'caches': all_caches,
            'error': None
        }
    
    except Exception as e:
        logger.error(f'Worker category eval error for pipeline {pipeline}: {e}')
        traceback.print_exc()
        return {
            'pipeline': pipeline,
            'acc': 0.0,
            'bandit_updates': [],
            'caches': {},
            'error': str(e)
        }

def evaluate_cate_pipeline_worker(args: tuple[list[int], dict[int, UCB1_Bandit]]):
    pipeline, bandits = args
    # Note: The signature has changed. It now only takes the pipeline.
    # It gets data and dataset_name from global variables.
    global worker_dataset_name #, worker_cache, worker_data, 
    global RAND_MAB
    try:
        concrete_pipeline = []
        bandit_updates: list[_BanditUpdate] = []

        for cate_idx in pipeline:
            if cate_idx < 0:
                concrete_pipeline.append(cate_idx)
            else:
                # category = IDX_TO_CATEGORY[cate_idx]
                bandit: UCB1_Bandit = bandits[cate_idx]
                if not RAND_MAB:
                    # selected_op = bandit.select_arm()
                    selected_op = bandit.get_best_arm()
                    bandit_updates.append({'category': cate_idx, 'arm': selected_op, 'reward': 0.})
                else:
                    selected_op = bandit.select_rand()
                
                concrete_pipeline.append(int(selected_op))

        acc, caches = evaluate_pipeline_worker(concrete_pipeline)
        cumu = []
        for update in bandit_updates:
            last_cumu = cumu.copy()
            last_acc = caches.get(f'{worker_dataset_name}:{last_cumu}', 0.)
            cumu.append(update['arm'])
            update['reward'] = caches.get(f'{worker_dataset_name}:{cumu}', 0.) - last_acc

        return {
            'pipeline': pipeline,
            'acc': acc,
            'bandit_updates': bandit_updates,
            'caches': caches,
            'error': None 
        }
    except Exception as e:
        logger.error(f'Worker category eval error for pipeline {pipeline}: {e}')
        return {
            'pipeline': pipeline,
            'acc': 0.0,
            'bandit_updates': [],
            'caches': {},
            'error': str(e)
        }

class ParallelPipelineEvaluator:
    def __init__(self,
                 dataset_name: str,
                 data: DatasetType,
                 operations: List[int],
                 n_processes: int | None,
                 cache_path: str = None,
                 dump_interval=10):
        self.dataset_name = dataset_name
        # self.data is no longer stored directly, it's passed to workers
        self.operations = operations
        self.n_processes = n_processes
        self.cache_path = cache_path
        self.dump_interval = dump_interval
        self.batch_counter = 0

        # shared memory (Manager is necessary for inter-process sharing)
        self.manager = Manager()
        self.shared_cache = self.manager.dict()
        self.cache_lock = self.manager.Lock()

        if self.cache_path and os.path.exists(self.cache_path):
            self.load_cache()

        self.stats: StatsDict = {
            'total_evaluations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'slot_times': [],
            'theoretical_evaluations': 0
        }

        # maxtasksperchild=1 helps with memory leaks in long-running tasks
        self.pool = Pool(
            processes=self.n_processes,
            initializer=init_worker,
            initargs=(data, dataset_name, self.shared_cache),
            maxtasksperchild=100  # Periodically restart workers to free memory
        )

    def load_cache(self):
        try:
            with open(self.cache_path, 'r') as f:
                cached_data = json.load(f)
            with self.cache_lock:
                self.shared_cache.update(cached_data)
            logger.info(f"Loaded {len(cached_data)} entries from cache file")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def dump_cache(self):
        if NO_DUMP: return
        if not self.cache_path: return
        try:
            with self.cache_lock:
                cache_data = dict(self.shared_cache.copy())
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'w') as f:
                json.dump(cache_data, f)
            logger.info(f'tail cache {list(cache_data.items())[-5:]}')
            logger.info(f"Saved {len(cache_data)} entries to cache file")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

        self.stats: StatsDict = {
            'total_evaluations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'slot_times': [],
            'theoretical_evaluations': 0
        }

    def evaluate_pipelines_batch(self, pipelines):
        results = {}
        uncached_pipelines = []
        hits = 0
        misses = 0

        with self.cache_lock:
            for pipeline in pipelines:
                pipeline_key = f'{self.dataset_name}:{str(pipeline)}'  # Use str(pipeline) for safety
                if pipeline_key in self.shared_cache:
                    results[pipeline_key] = self.shared_cache[pipeline_key]
                    hits += 1
                else:
                    uncached_pipelines.append(pipeline)
                    misses += 1

        self.stats['cache_hits'] += hits
        self.stats['cache_misses'] += misses

        if not uncached_pipelines:
            logger.info(f"      all {len(pipelines)} pipelines hit cache, no evaluation needed.")
            return results

        logger.info(f"      evaluate {len(uncached_pipelines):,} un-cached pipelines...")

        # pool.imap_unordered is often faster as it yields results as they complete
        # We wrap it in tqdm for a progress bar.
        batch_results_iterator = self.pool.imap_unordered(evaluate_pipeline_worker, uncached_pipelines)

        batch_results = list(tqdm(batch_results_iterator,
                                  total=len(uncached_pipelines),
                                  desc="      Progress",
                                  leave=False))

        with self.cache_lock:
            for pipeline, (performance, cache) in zip(uncached_pipelines, batch_results):
                pipeline_key = f'{self.dataset_name}:{str(pipeline)}'
                # self.shared_cache[pipeline_key] = performance
                self.shared_cache.update(cache)
                results[pipeline_key] = performance

        self.batch_counter += 1
        if self.batch_counter % self.dump_interval == 0:
            self.dump_cache()
            # self.batch_counter is not shared, this is fine

        self.stats['total_evaluations'] += len(uncached_pipelines)
        logger.info(f"      completed {len(uncached_pipelines)} pipeline evaluations, hit rate: {hits / (hits + misses):.2%}")

        return results

    def cleanup(self):
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()  
        if hasattr(self, 'manager'):
            self.manager.shutdown()

    def dump_final(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            cache = dict(self.shared_cache.copy())
            json.dump(cache, f)


class CategoricalParallelPipelineEvaluator(ParallelPipelineEvaluator):
    def __init__(self, dataset_name: str,
                 data: DatasetType,
                 operations: List[int],
                 n_processes: int | None,
                 bandits: dict[int, UCB1_Bandit],
                 cache_path: str = None,
                 dump_interval=10):
        super().__init__(dataset_name, data, operations, n_processes, cache_path, dump_interval)
        self.bandits = bandits
        # self.cate_cache = self.manager.dict()

    def get_cate_pipe_key(self, pipeline: list[int]):
        p = list(filter(lambda x: x >= 0, pipeline))
        return f'cat:{self.dataset_name}:{str(p)}'

    def eval_cate_batched(self, pipelines: list[list[int]]):
        global EXHAUST_ENUM_OP_IN_CATE
        if not pipelines: return {}
        results = {}
        try:
            args_list = [(pipe, self.bandits) for pipe in pipelines]
            logger.info(f"Evaluating {len(pipelines)} category pipelines with {self.n_processes} processes...")
            logger.info(f"Process pool initialized: {self.pool is not None}")
            eval_results = []

            if EXHAUST_ENUM_OP_IN_CATE:
                for result in tqdm(self.pool.imap_unordered(eval_cate_pipe_work_enum_ops_in_cate, args_list),
                                total=len(pipelines), desc='Eval cate pipelines with enum ops in cate'):
                    eval_results.append(result)
            else:
                for result in tqdm(self.pool.imap_unordered(evaluate_cate_pipeline_worker, args_list),
                                total=len(pipelines), desc='Eval cate pipelines'):
                    eval_results.append(result)
            logger.info(f"Completed evaluation of {len(pipelines)} category pipelines")

            for result in eval_results:
                if result['error'] is None:
                    pipeline = result['pipeline']
                    acc = result['acc']
                    bandit_updates: list[_BanditUpdate] = result['bandit_updates']
                    caches = result['caches']
                    pipeline_key = self.get_cate_pipe_key(pipeline)
                    results[pipeline_key] = acc
                    
                    if bandit_updates:
                        for update in bandit_updates:
                            try:
                                self.bandits[update['category']].update(
                                    update['arm'], update['reward']
                                )
                            except Exception as e:
                                logger.error(f"Failed to update bandit: {e}")
                    
                    if caches:
                        with self.cache_lock:
                            self.shared_cache.update(caches)

                else:
                    pipeline = result['pipeline']
                    pipeline_key = self.get_cate_pipe_key(pipeline)
                    results[pipeline_key] = 0.0
                    logger.error(f"Pipeline evaluation failed: {result.get('error', 'Unknown error')}")

            return results

        except Exception as e:
            logger.error(f'Worker category eval error for pipelines: {e}')
            return {}


class CateShap:
    def __init__(self,
                 dataset_name: str,
                 dataset: DatasetType,
                 bandits: dict[int, UCB1_Bandit],
                 slot_num=3,
                 n_processes=None,
                 force_large_computation=False,
                 operations=list(range(len(QActionProvider.spans))),
                 dump_interval: int = 10,
                 cache_path: str = None,
                 output_shapley_values_path: str = None):
        self.dataset_name = dataset_name
        self.dataset: DatasetType = {
            'train' : dataset['train'].copy(),
            'test' : dataset['test'].copy(),
            'target' : dataset['target'].copy(),
            'target_test' : dataset['target_test'].copy(),
        }
        self.operations = operations
        self.baseline_idx = -1

        self.shapley_values = {}
        self.N = len(operations)
        self.M = slot_num
        self.n_processes = n_processes or min(mp.cpu_count(), 16)
        self.force_large_computation = force_large_computation
        self.output_shapley_values_path = output_shapley_values_path

        self.evaluator = CategoricalParallelPipelineEvaluator(
            self.dataset_name,
            self.dataset,
            self.operations,
            self.n_processes,
            bandits=bandits,
            cache_path=cache_path,
            dump_interval=dump_interval
        )

        self.shapley_cache = []
        self.local_set = set()

        self.stats: StatsDict = {
            'total_evaluations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'slot_times': [],
            'theoretical_evaluations': 0
        }
    
    def run_full_enum_cate(self):
        logger.info(f"\n{'=' * 60}")
        logger.info("STARTING CATE ENUMERATION ALGORITHM")
        logger.info("WARNING: This will evaluate ALL possible CATE coalitions")
        global OP_USE, EXHAUST_ENUM_OP_IN_CATE
        logger.info(f"OP_USE: {OP_USE}")
        logger.info(f"{'=' * 60}")
        algorithm_start = time.time()

        selected_pipeline: list[int] = [-1 for _ in range(self.M)]

        string_columns = self.dataset['train'].select_dtypes(exclude=['number']).columns
        has_cate_enc = len(string_columns) > 0
        has_cate_miss = self.dataset['train'][string_columns].isnull().any().any()
        has_num_miss = self.dataset['train'].select_dtypes(include=['number']).isnull().any().any()

        start = 0
        if has_cate_miss:
            selected_pipeline.append(-1)
            selected_pipeline[start] = 0  # QActionProvider.spans[0], cate imputer
            start += 1
        
        if has_cate_enc:
            selected_pipeline.append(-1)
            selected_pipeline[start] = 2  # QActionProvider.spans[2], cate encoder
            start += 1

        if has_num_miss:
            selected_pipeline.append(-1)
            selected_pipeline[start] = 1  # QActionProvider.spans[1], num imputer
            start += 1
        
        eval_total_cnt = 0
        while -1 in selected_pipeline:
            shap_grid = np.ones((self.N, self.M + start)) * (-99)
            logger.info(f'Current selected cate pipeline: {selected_pipeline}')
            slot_start = time.time()

            for slot_pos in range(start, self.M + start):
                if selected_pipeline[slot_pos] != -1:
                    continue
                for cate_idx in range(3, self.N):   # skipped 0~2 (imputer and encoder)
                    if not EXHAUST_ENUM_OP_IN_CATE:
                        shapley_val, eval_cnt = self.calc_cell_shap(slot_pos, cate_idx, selected_pipeline, start)
                    else:
                        shapley_val, eval_cnt = self.calc_cell_shap_exhaust(slot_pos, cate_idx, selected_pipeline, start)
                    shap_grid[cate_idx][slot_pos] = shapley_val
                    eval_total_cnt += eval_cnt
            
            best_shapley = float('-inf')
            best_op = None
            best_slot = None
            for slot_pos in range(start, self.M + start):
                for cate_idx in range(self.N):
                    if shap_grid[cate_idx][slot_pos] > best_shapley:
                        best_shapley = shap_grid[cate_idx, slot_pos]
                        best_op = cate_idx
                        best_slot = slot_pos
            
            assert best_op is not None and best_slot is not None
            selected_pipeline[best_slot] = best_op

            slot_time = time.time() - slot_start
            self.stats['slot_times'].append(slot_time)

            self.evaluator.dump_cache()
        
        total_time = time.time() - algorithm_start
        logger.info(f"\n{'=' * 60}")
        logger.info(f"CATE ENUMERATION COMPLETED IN {total_time:.2f}s")
        logger.info(f"{'=' * 60}")
        logger.info(f"Pipeline: {selected_pipeline}")

        if self.output_shapley_values_path is not None:
            import pandas as pd
            pd.DataFrame(self.shapley_cache).to_csv(self.output_shapley_values_path.replace('.json', '.csv'), index=False)

        return selected_pipeline, eval_total_cnt

    def calc_cell_shap_exhaust(self, slot_pos: int, cate_idx: int, selected_pipeline: list[int], prep_start: int):
        feat_spans = [*QActionProvider.spans[3],
                      *QActionProvider.spans[4],
                      *QActionProvider.spans[5], -1]
        to_product = []
        for c in selected_pipeline:
            if c >= 0:
                to_product.append(self.evaluator.bandits[c].arms[:])
            else:
                to_product.append(feat_spans[:])
        ops_in_cate = self.evaluator.bandits[cate_idx].arms[:]
        best_sv = -float('inf')
        cache_cnt = 0
        for op in ops_in_cate:
            to_product[slot_pos] = [op]
            all_pipelines = []
            for p in product(*to_product):
                if check_constraint_ext(p, self.dataset['train'].shape[1]):
                    all_pipelines.append(list(p))
            eval_set = []
            for p in all_pipelines:
                clean_p = tuple(filter(lambda x: x>=0, p))
                if clean_p not in eval_set:
                    eval_set.append(clean_p)
                pwo = tuple(filter(lambda x:x>=0, p[:slot_pos] + p[slot_pos+1:]))
                if pwo not in eval_set:
                    eval_set.append(pwo)
            results = self.evaluator.evaluate_pipelines_batch(list(map(list, eval_set)))
            cache_cnt += len(results)
            sv = 0.
            for p in all_pipelines:
                clean_p = list(filter(lambda x: x>=0, p))
                tk = f'{self.dataset_name}:{clean_p}'
                clean_pwo = list(filter(lambda x: x>=0, p[:slot_pos]+p[slot_pos+1:]))
                bk = f'{self.dataset_name}:{clean_pwo}'
                sv += results[tk] - results[bk]
            sv = sv / len(all_pipelines) if len(all_pipelines) > 0 else 0.
            if sv > best_sv:
                best_sv = sv
        return best_sv, cache_cnt


    def calc_cell_shap(self, slot_pos: int, cate_idx: int, selected_pipeline: list[int], prep_start: int):
        global EXHAUST_ENUM_OP_IN_CATE
        coalitions = self.generate_cate_coalitions_cell(selected_pipeline, slot_pos, cate_idx, prep_start)
        coalition_cnt = len(coalitions)
        pipelines_to_evaluate = []
        for coalition in coalitions:
            pipeline_base = tuple(filter(lambda x: x != -1, coalition.copy()))
            if pipeline_base not in pipelines_to_evaluate:
                pipelines_to_evaluate.append(pipeline_base)
                self.local_set.add(pipeline_base)

            pipeline_target = coalition.copy()
            pipeline_target[slot_pos] = cate_idx
            pipeline_target = tuple(filter(lambda x: x != -1, pipeline_target))
            if pipeline_target not in pipelines_to_evaluate:
                pipelines_to_evaluate.append(pipeline_target)
                self.local_set.add(pipeline_target)

        if coalition_cnt == 0:
            logger.info(f"        No pipelines to evaluate for slot {slot_pos} with target {cate_idx}")
            eval_time = 0.
            results = {}
        else:
            pipelines_to_evaluate = list(map(list, pipelines_to_evaluate))
            logger.info(f"        Total cate pipelines to evaluate: {len(pipelines_to_evaluate):,}")
            start_time = time.time()
            results = self.evaluator.eval_cate_batched(pipelines_to_evaluate)
            eval_time = time.time() - start_time

        shapley_value = 0.0
        if coalition_cnt > 0:
            weight = 1.0 / coalition_cnt
        else:
            weight = 0.

        for coalition in coalitions:
            base_key = self.evaluator.get_cate_pipe_key(list(filter(lambda x: x != -1, coalition.copy())))
            target_pipe = coalition.copy()
            target_pipe[slot_pos] = cate_idx
            target_key = self.evaluator.get_cate_pipe_key(list(filter(lambda x: x != -1, target_pipe)))
            shapley_value += (results[target_key] - results[base_key]) * weight

        self.stats['total_evaluations'] += len(pipelines_to_evaluate) * (MCTS_CNT if EXHAUST_ENUM_OP_IN_CATE else 1)
        logger.info(f"        Shapley value: {shapley_value:.6f} (time: {eval_time:.1f}s)")

        if self.output_shapley_values_path is not None:
            self.shapley_cache.append({
                'dataset': self.dataset_name,
                'cate_idx': cate_idx,
                'slot': slot_pos + 1,
                'shapley_value': shapley_value,
                'selected_pipe': selected_pipeline[:],
            })
        return shapley_value, len(results)


    def generate_cate_coalitions_cell(self, selected_pipeline: list[int], slot_pos: int, cate_idx: int, prep_start: int):
        coalitions = []
        prev_sel_positions = [i for i, x in enumerate(selected_pipeline) if x != -1]
        occupied_positions = set(prev_sel_positions + [slot_pos])
        remaining_slots = len(selected_pipeline) - len(occupied_positions)
        # This is used to adjust the starting index for generating coalitions.
        prev_occupied_pos_cnt_before_slot = len([pos for pos in prev_sel_positions if pos < slot_pos])
        if slot_pos >= prep_start + prev_occupied_pos_cnt_before_slot:
            start = slot_pos - prep_start - prev_occupied_pos_cnt_before_slot
        else:
            start = 0
        for r in range(start, remaining_slots + 1):
            if r < 0:
                logger.error(f"r={r} is negative, which is not expected.")
                continue
            elif r == 0:
                final_pipe = selected_pipeline.copy()
                coalitions.append(final_pipe)
            else:
                f_span = [3,4,5] # QActionProvider.spans: [3,4,5]
                for combo in product(f_span, repeat=r):
                    final_pipe = selected_pipeline.copy()
                    empty_positions = [i for i in range(len(final_pipe)) if final_pipe[i] == -1 and i != slot_pos]
                    for pos, cate in zip(empty_positions, combo):
                        final_pipe[pos] = cate
                    coalitions.append(final_pipe)
        if len(coalitions) > MCTS_CNT:
            return random.sample(coalitions, k=MCTS_CNT)
        return coalitions


class PermShapley:
    def __init__(self,
                 dataset_name: str,
                 dataset: DatasetType,
                 slot_num=3,
                 n_processes=None,
                 force_large_computation=False,
                 cache_manager='dict',
                 operations=list(range(len(QActionProvider.methods))),
                 cache_path: str = None,
                 dump_interval: int = 10,
                 output_shapley_values_path=None): 
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.operations = operations
        self.baseline_idx = -1

        self.shapley_values = {}
        self.N = len(operations)
        self.M = slot_num

        self.n_processes = n_processes or min(mp.cpu_count(), 16)
        self.force_large_computation = force_large_computation
        self.evaluator = ParallelPipelineEvaluator(
            self.dataset_name,
            self.dataset,
            self.operations,
            self.n_processes,
            cache_path=cache_path, 
            dump_interval=dump_interval 
        )

        self.output_shapley_values_path = output_shapley_values_path
        self.shapley_cache = []

        self.stats: StatsDict = {
            'total_evaluations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'slot_times': [],
            'theoretical_evaluations': 0
        }

        self.local_set = set()

        # self._analyze_full_complexity()

    def _analyze_full_complexity(self):
        logger.info(f"\nFULL COMPLEXITY ANALYSIS:")
        logger.info(f"{'Slot':<6} {'Remaining':<9} {'Coalitions/Op':<12} {'Evals/Op':<10} {'Total Evals':<12}")
        logger.info("-" * 60)

        total_evaluations = 0
        slot_details = []

        for slot in range(self.M):
            remaining_slots = self.M - slot - 1
            coalitions_per_operation = self.N ** remaining_slots
            evaluations_per_operation = coalitions_per_operation * 2  # target + baseline
            slot_evaluations = self.N * evaluations_per_operation  # exclude baseline op
            total_evaluations += slot_evaluations

            slot_details.append({
                'slot': slot + 1,
                'remaining_slots': remaining_slots,
                'coalitions_per_op': coalitions_per_operation,
                'evaluations_per_op': evaluations_per_operation,
                'slot_total': slot_evaluations
            })

            logger.info(
                f"{slot + 1:<6} {remaining_slots:<9} {coalitions_per_operation:<12,} {evaluations_per_operation:<10,} {slot_evaluations:<12,}")

        logger.info("-" * 60)
        logger.info(f"TOTAL EVALUATIONS: {total_evaluations:,}")

        self.stats['theoretical_evaluations'] = total_evaluations

        est_time_per_eval = 0.05  # 50ms per evaluation
        sequential_time = total_evaluations * est_time_per_eval
        parallel_time = sequential_time / self.n_processes * 1.3  # overhead

        logger.info(f"\nTIME ESTIMATES:")
        logger.info(
            f"Sequential: {sequential_time:,.1f}s = {sequential_time / 3600:.1f} hours = {sequential_time / 86400:.1f} days")
        logger.info(
            f"Parallel ({self.n_processes} cores): {parallel_time:,.1f}s = {parallel_time / 3600:.1f} hours = {parallel_time / 86400:.1f} days")

        if total_evaluations > 1000000:  # 1M evaluations
            logger.info(f"\n{'!' * 50}")
            logger.info(f"EXTREME COMPLEXITY WARNING!")
            logger.info(f"This computation requires {total_evaluations:,} evaluations")
            logger.info(f"Estimated time: {parallel_time / 3600:.1f} hours")
            logger.info(f"Consider reducing operations or pipeline length")
            if not self.force_large_computation:
                logger.info(f"Set force_large_computation=True to proceed anyway")
                logger.info(f"{'!' * 50}")
                raise ValueError("Computation too large - aborted for safety")
        elif total_evaluations > 50000:  # 50K evaluations
            logger.info(f"\nHIGH COMPLEXITY WARNING!")
            logger.info(f"This computation requires {total_evaluations:,} evaluations")
            logger.info(f"Estimated time: {parallel_time / 60:.1f} minutes")
        else:
            logger.info(f"\nComplexity acceptable: {total_evaluations:,} evaluations")

        return total_evaluations


    def generate_constrainted_coalitions_cell(self, previous_sels: list[int], slot_pos: int, target_op: int, 
                                              has_cate_miss, has_cate_enc, has_num_miss, num_dim, cate_dim, prep_start,
                                              selected_cate_pipe):
        global OP_USE
        cat_miss_span = [-1]
        num_miss_span = [-1]
        cat_enc_span = [-1]
        if has_cate_miss:
            cat_miss_span = list(set(QActionProvider.spans[0]))
        if has_cate_enc:
            cat_enc_span = list(set(QActionProvider.spans[2]))
        if has_num_miss:
            num_miss_span = list(set(QActionProvider.spans[1]))

        feature_spans = [op for span in QActionProvider.spans[3:] for op in span]
        
        cnt = 0
        cnt_guard, GUARD_MAX = 0, 2000
        prev_sel_positions = [i for i, x in enumerate(previous_sels) if x != -1]
        coalitions = []
        try:
            for cate_miss_op in cat_miss_span:
                for num_miss_op in num_miss_span:
                    for cat_enc_op in cat_enc_span:
                        base_pipe = list(filter(lambda x: x != -1, [cate_miss_op, cat_enc_op, num_miss_op]))
                        # logger.debug(f'slot {slot_pos}, base_pipe: {base_pipe}')
                        occupied_positions = set(list(range(len(base_pipe))) + [slot_pos] + prev_sel_positions) 
                        # logger.debug(f'occupied_positions: {occupied_positions}')
                        remaining_slots = self.M + prep_start - len(occupied_positions)
                        # |B      B      _      S     P      _ |
                        #  ^base1 ^base2 ^empty ^slot ^prev  ^empty
                        prev_occupied_pos_cnt_before_slot = len([pos for pos in prev_sel_positions if pos < slot_pos])
                        if slot_pos >= len(base_pipe) + prev_occupied_pos_cnt_before_slot:
                            start = slot_pos - len(base_pipe) - prev_occupied_pos_cnt_before_slot
                        else:
                            start = 0
                        
                        while True:
                            cnt_guard += 1
                            if cnt_guard > GUARD_MAX:
                                raise StopIteration
                            size = np.random.choice(list(range(start, remaining_slots + 1)))
                            f_spans = feature_spans #list(set(feature_spans) - set(previous_sels) - set([target_op]))
                            # combo = np.random.choice(f_spans, size=size, replace=False).tolist()
                            if size == 0:
                                final_pipe = previous_sels.copy()
                                for pos, b in enumerate(base_pipe):
                                    final_pipe[pos] = b
                                # yield final_pipe
                                if tuple(final_pipe) not in coalitions:
                                    coalitions.append(tuple(final_pipe))
                                    cnt += 1
                                    if cnt >= MCTS_CNT:
                                        raise StopIteration
                            else:
                                final_pipe = [-1 for _ in range(len(previous_sels))]
                                for pos, op in enumerate(base_pipe): final_pipe[pos] = op
                                for pos, prev in enumerate(previous_sels):
                                    if prev != -1: final_pipe[pos] = prev
                                empty_positions = [i for i in range(len(previous_sels)) if final_pipe[i] == -1 and i != slot_pos]
                                combo = []
                                # logger.debug(f'empty_positions: {empty_positions}, f_spans: {f_spans}')
                                for ep in empty_positions:
                                    # combo.append(int(np.random.choice(list(set(f_spans) & set(QActionProvider.spans[selected_cate_pipe[ep]])))))
                                    combo.append(int(np.random.choice(f_spans)))
                                for pos, op in zip(empty_positions, combo):
                                    final_pipe[pos] = op
                                
                                final_target_pipe = final_pipe.copy()
                                final_target_pipe[slot_pos] = target_op
                                if not check_constraint_ext(final_target_pipe, self.dataset['train'].shape[1]):
                                    continue

                                if tuple(final_pipe) not in coalitions:
                                    # yield final_pipe
                                    coalitions.append(tuple(final_pipe))
                                    cnt += 1
                                    if cnt >= MCTS_CNT:
                                        raise StopIteration
        except StopIteration:
            logger.info(f"Generated {cnt} pipelines, iteration done")

        return list(map(list, coalitions))


    def calc_cell_shap(self, slot_pos: int, target_op: int, previous_sels: list[int], 
                       has_cate_miss, has_cate_enc, has_num_miss, num_dim, cate_dim, prep_start,
                       selected_cate_pipe):
        has_cate_miss = has_cate_miss and len(set(previous_sels) & set(QActionProvider.spans[0])) == 0 and target_op not in QActionProvider.spans[0]
        has_cate_enc = has_cate_enc and len(set(previous_sels) & set(QActionProvider.spans[2])) == 0 and target_op not in QActionProvider.spans[2]
        has_num_miss = has_num_miss and len(set(previous_sels) & set(QActionProvider.spans[1])) == 0 and target_op not in QActionProvider.spans[1]

        coalitions = self.generate_constrainted_coalitions_cell(
            previous_sels, slot_pos, target_op, has_cate_miss, has_cate_enc, has_num_miss, num_dim, cate_dim, prep_start, selected_cate_pipe)
        coalition_cnt = len(coalitions)
        pipelines_to_evaluate = []
        for coalition in coalitions:
            pipeline_base = tuple(filter(lambda x: x != -1, coalition.copy()))
            if pipeline_base not in pipelines_to_evaluate:
                pipelines_to_evaluate.append(pipeline_base)
                self.local_set.add(pipeline_base)

            pipeline_target = coalition.copy()
            pipeline_target[slot_pos] = target_op
            pipeline_target = tuple(filter(lambda x: x != -1, pipeline_target))
            if pipeline_target not in pipelines_to_evaluate:
                pipelines_to_evaluate.append(pipeline_target)
                self.local_set.add(pipeline_target)

        if coalition_cnt == 0:
            logger.info(f"        No pipelines to evaluate for slot {slot_pos} with target {target_op}")
            eval_time = 0.
            results = {}
        else:
            pipelines_to_evaluate = list(map(list, pipelines_to_evaluate))
            logger.info(f"        Total pipelines to evaluate: {len(pipelines_to_evaluate):,}")
            start_time = time.time()
            results = self.evaluator.evaluate_pipelines_batch(pipelines_to_evaluate)
            eval_time = time.time() - start_time
        shapley_value = 0.0
        if coalition_cnt > 0:
            weight = 1.0 / coalition_cnt
        else:
            weight = 0.

        for coalition in coalitions:
            base_key = f'{self.dataset_name}:{list(filter(lambda x: x != -1, coalition.copy()))}'
            target_pipe = coalition.copy()
            target_pipe[slot_pos] = target_op
            target_key = f'{self.dataset_name}:{list(filter(lambda x: x != -1, target_pipe))}'
            shapley_value += (results[target_key] - results[base_key]) * weight

        self.stats['total_evaluations'] += len(pipelines_to_evaluate)
        logger.info(f"        Shapley value: {shapley_value:.6f} (time: {eval_time:.1f}s)")

        if self.output_shapley_values_path is not None:
            self.shapley_cache.append({
                'dataset': self.dataset_name,
                'operation': target_op,
                'slot': slot_pos + 1,
                'shapley_value': shapley_value,
                'selected_pipe': previous_sels[:],
            })
        return shapley_value


    def run_full_enumerate_algorithm(self, selected_cate_pipe: list[int]):
        logger.info(f"\n{'=' * 60}")
        logger.info("STARTING FULL ENUMERATION ALGORITHM")
        logger.info("WARNING: This will evaluate ALL possible coalitions")
        global OP_USE
        logger.info(f"OP_USE: {OP_USE}")
        logger.info(f"{'=' * 60}")

        algorithm_start = time.time()
        selected_pipeline: list[int] = [-1 for _ in range(self.M)]

        string_columns = self.dataset['train'].select_dtypes(exclude=['number']).columns
        has_cate_enc = len(string_columns) > 0
        has_cate_miss = self.dataset['train'][string_columns].isnull().any().any()
        has_num_miss = self.dataset['train'].select_dtypes(include=['number']).isnull().any().any()

        feature_spans = [op for span in QActionProvider.spans[3:] for op in span]

        start = 0
        if has_cate_miss:
            selected_pipeline.append(-1)
            selected_pipeline[start] = QActionProvider.spans[0][0]
            start += 1

        if has_cate_enc:
            shaps: dict[int, float] = {}
            selected_pipeline.append(-1)
            selected_cate_pipe_copy = selected_cate_pipe[:start+1] + selected_cate_pipe[-self.M:]
            # if len(selected_cate_pipe) > self.M + 1:
            #     selected_cate_pipe_copy = selected_cate_pipe[:-self.M-1] + selected_cate_pipe[-self.M:]
            for op_idx in list(set(QActionProvider.spans[2])):
                shaps[op_idx] = self.calc_cell_shap(start, op_idx, selected_pipeline, has_cate_miss, has_cate_enc, has_num_miss, 
                                                    len(feature_spans), len(string_columns), start, selected_cate_pipe_copy)
            selected_op, shap = max(shaps.items(), key=lambda x: x[1])
            selected_pipeline[start] = selected_op
            start += 1

        if has_num_miss:
            shaps: dict[int, float] = {}
            selected_pipeline.append(-1)
            for op_idx in list(set(QActionProvider.spans[1])):
                shaps[op_idx] = self.calc_cell_shap(start, op_idx, selected_pipeline, has_cate_miss, has_cate_enc, has_num_miss, 
                                                    len(feature_spans), len(string_columns), start, selected_cate_pipe)
            selected_op, shap = max(shaps.items(), key=lambda x: x[1])
            selected_pipeline[start] = selected_op
            start += 1

        while -1 in selected_pipeline:
            shap_grid = np.ones((self.N, self.M + start)) * (-99)
            logger.info(f'Current selected pipeline: {selected_pipeline}')
            slot_start = time.time()
            
            for slot_pos in range(start, self.M + start):
                if selected_pipeline[slot_pos] != -1:
                    continue
                for op_idx in list(set(range(self.N)) & set(feature_spans) & set(QActionProvider.spans[selected_cate_pipe[slot_pos]])):
                    # if op_idx in selected_pipeline:
                    #     continue
                    if (
                        (not has_cate_enc and op_idx in QActionProvider.spans[2]) 
                        or (op_idx in QActionProvider.spans[2] and len(set(selected_pipeline) & set(QActionProvider.spans[2])) > 0) 
                    ):
                        continue
                    if (
                        (not has_num_miss and op_idx in QActionProvider.spans[1]) 
                        or (op_idx in QActionProvider.spans[1] and len(set(selected_pipeline) & set(QActionProvider.spans[1])) > 0) 
                    ):
                        continue
                    if (
                        (not has_cate_miss and op_idx in QActionProvider.spans[0])  
                        or (op_idx in QActionProvider.spans[0] and len(set(selected_pipeline) & set(QActionProvider.spans[0])) > 0) 
                    ):
                        continue

                    selected_positions = [i for i, x in enumerate(selected_pipeline) if x != -1]
                    fl = False
                    for sel_pos in selected_positions:
                        if abs(slot_pos - sel_pos) == 1 and OC.blocked_pair_lease(selected_pipeline[sel_pos], op_idx): 
                            fl = True
                            break

                    if fl: continue

                    shapley_val = self.calc_cell_shap(slot_pos, op_idx, selected_pipeline, has_cate_miss, has_cate_enc, has_num_miss, 
                        self.dataset['train'].shape[1]-len(string_columns), len(string_columns), start, selected_cate_pipe)
                    
                    shap_grid[op_idx][slot_pos] = shapley_val
            
            best_shapley = float('-inf')
            best_op = None
            best_slot = None
            for slot_pos in range(start, self.M + start):
                for op_idx in range(self.N):
                    if shap_grid[op_idx][slot_pos] > best_shapley:
                        best_shapley = shap_grid[op_idx, slot_pos]
                        best_op = op_idx
                        best_slot = slot_pos

            assert best_op is not None and best_slot is not None
            slot_time = time.time() - slot_start
            self.stats['slot_times'].append(slot_time)

            self.evaluator.dump_cache()
            if best_shapley <= 0 and best_op not in QActionProvider.spans[5] and EARLY_STOP:
                break
            selected_pipeline[best_slot] = best_op

        total_time = time.time() - algorithm_start
        selected_pipeline = list(filter(lambda x: x != -1, selected_pipeline))

        final_pipeline_key = f'{self.dataset_name}:{str(selected_pipeline)}'
        if final_pipeline_key in self.evaluator.shared_cache:
            final_accuracy = self.evaluator.shared_cache[final_pipeline_key]
        else:
            # This should ideally be in cache, but as a fallback, evaluate it
            final_accuracy, cached = evaluate_pipeline_worker(selected_pipeline)

        logger.info(f"\n{'=' * 60}")
        logger.info("FULL ENUMERATION COMPLETED")
        logger.info(f"{'=' * 60}")
        logger.info(f"Pipeline: {selected_pipeline}")
        logger.info(f"Accuracy: {final_accuracy:.4f}")  # Corrected line
        logger.info(f"Total time: {total_time / 60:.1f} minutes")
        logger.info(f"Evaluations: {self.stats['total_evaluations']:,}")
        logger.info(f"Theoretical: {self.stats['theoretical_evaluations']:,}")
        # cache_efficiency = (self.stats['theoretical_evaluations'] - self.evaluator.stats['total_evaluations']) / \
        #                    self.stats['theoretical_evaluations'] * 100
        # logger.info(f"Cache efficiency: {cache_efficiency:.1f}%")

        if self.output_shapley_values_path is not None:
            import pandas as pd
            pd.DataFrame(self.shapley_cache).to_csv(self.output_shapley_values_path.replace('.json', '.csv'), index=False)

        return selected_pipeline, final_accuracy


class HieShapley:
    def __init__(self, dataset_name: str, data: DatasetType,
                 pipeline_length: int, other_cache_path: str,
                 enable_pretrain = True, n_processes = 4):
        self.dataset_name = dataset_name
        self.data = data
        self.pipeline_length = pipeline_length

        self.other_cache_path = other_cache_path
        if not os.path.exists(self.other_cache_path):
            os.makedirs(os.path.dirname(self.other_cache_path), exist_ok=True)

        self.enable_pretrain = enable_pretrain
        self.num_samples = MCTS_CNT
        self.n_processes = n_processes
        self.mab_sample = MAB_SAMPLES

    def _pretrain_bandits(self, bandits: dict[int, UCB1_Bandit], sample_size=None):
        cache_path = self.other_cache_path
        if CACHE_LABEL.strip() != '':
            cache_path = cache_path.replace(CACHE_LABEL, '')
        if not os.path.exists(cache_path):
            logger.info('Cache file not found, skip pretraining!')
            return
        string_columns = self.data['train'].select_dtypes(exclude=['number']).columns
        has_cate_enc = len(string_columns) > 0
        has_cate_miss = self.data['train'][string_columns].isnull().any().any()
        has_num_miss = self.data['train'].select_dtypes(include=['number']).isnull().any().any()

        actual_len = self.pipeline_length + int(has_cate_enc) + int(has_cate_miss) + int(has_num_miss)

        with open(cache_path, 'r') as f:
            cache = json.load(f)

        cache2 = {}
        for k, v in cache.items():
            dsn, pipe_str = k.split(':')
            pipe: list[int] = json.loads(pipe_str)
            pipe = list(filter(lambda x: x >= 0, pipe))
            if len(pipe) == actual_len or len(pipe) == actual_len - 1:
                cache2[tuple(pipe)] = v
        
        cache2_list = list(cache2.items())
        cache2_Alen_list = [item for item in cache2_list if len(item[0]) == actual_len]

        if self.mab_sample is not None and len(cache2_Alen_list) > self.mab_sample:
            cache2_Alen_list = random.sample(cache2_Alen_list, self.mab_sample)
        del cache

        logger.info(f'Pretrain bandits with {len(cache2_Alen_list)} samples')

        for j in range(actual_len):
            for i in range(len(cache2_Alen_list)):
                pipe, acc = cache2_Alen_list[i]
                pipe_pop_last = list(pipe)
                pipe_pop_last.pop(j)
                pop_acc = cache2.get(tuple(pipe_pop_last), None)
                if pop_acc is None: 
                    # logger.info(f'Pop last {pipe_pop_last} not found in cache, skip')
                    continue
                target_opid = pipe[j]
                cate_id = QActionProvider.idx_to_factory_id(target_opid)
                bandits[cate_id].update(target_opid, acc - pop_acc)

    def run(self):
        dataset_name = self.dataset_name
        bandits = {
            i: UCB1_Bandit(arms=QActionProvider.spans[i])
            for i in range(len(QActionProvider.spans))
        }

        if self.enable_pretrain:
            logger.info('Enable MAB pretrain')
            self._pretrain_bandits(bandits, sample_size=self.mab_sample)
        
        logger.info(('#' * 10) + f' num_samples: {self.num_samples} ' + ('#' * 10))
        logger.info(f"\n{'#' * 60}\n### STAGE 1: MAB-GUIDED SEARCH FOR BEST CATEGORY PIPELINE ###\n{'#' * 60}")
        logger.info(('=' * 10) + ' create stage 1 evaluator ' + ('=' * 10))
        
        # category_evaluator = ...

        t0 = time.time()
        logger.info(('=' * 10) + ' create category shapley ' + ('=' * 10))
        category_shapley = CateShap(
            dataset_name,
            self.data,
            bandits,
            self.pipeline_length,
            cache_path=self.other_cache_path,
            dump_interval=20,
            force_large_computation=True,
            n_processes=self.n_processes,
            output_shapley_values_path=f'save/shapley_value_pos_vis/{self.dataset_name}.oldhsp.stage1.repeat.csv',
        )
        mab_init_time = time.time() - t0
        logger.info(('=' * 10) + ' run category shapley ' + ('=' * 10))
        t1 = time.time()
        best_category_indices, stage_1_evals = category_shapley.run_full_enum_cate()
        stage1_time = time.time() - t1
        logger.info(('=' * 10) + f' got best category indices ' + ('=' * 10) + str(best_category_indices))
        logger.info(('=' * 10) + f'stage 1: {stage_1_evals} evals ' + ('=' * 10))

        # best_category_pipeline = [IDX_TO_CATEGORY[i] for i in best_category_indices]
        logger.info(f"STAGE 1 RESULT: Best category pipeline found: {best_category_indices}")
        category_shapley.evaluator.cleanup()

        logger.info(f"\n{'#' * 60}\n### STAGE 2: MAB-WARM-STARTED GREEDY SEARCH ###\n{'#' * 60}")
        logger.info(('=' * 10) + ' create stage 2 evaluator ' + ('=' * 10))
        t2 = time.time()

        logger.info(('=' * 10) + ' create operator shapley ' + ('=' * 10))
        operator_searcher = PermShapley(
            dataset_name,
            self.data,
            self.pipeline_length,
            cache_path=self.other_cache_path,
            dump_interval=20,
            force_large_computation=True,
            n_processes=self.n_processes,
            output_shapley_values_path=f'save/shapley_value_pos_vis/{self.dataset_name}.oldhsp.stage2.repeat.csv',
        )
        logger.info(('=' * 10) + ' run operator shapley ' + ('=' * 10))
        final_pipeline, final_acc = operator_searcher.run_full_enumerate_algorithm(best_category_indices)
        logger.info(('=' * 10) + ' got best operator pipeline ' + ('=' * 10) + str(final_pipeline))
        stage_2_evals = operator_searcher.stats['total_evaluations']
        logger.info(('=' * 10) + ' total evals in stage 2 ' + ('=' * 10) + str(stage_2_evals))

        stage2_time = time.time() - t2
        # t3 = time.time()
        # acc, f1, recall, precision = operator_searcher.evaluator.single_pipe_multi_metric_nocache(final_pipeline)
        # final_time = time.time() - t3
        logger.info(f"\n{'#' * 60}\n### HIERARCHICAL SEARCH COMPLETED ###\n{'#' * 60}")
        logger.info(f"Final Operator Pipeline: {final_pipeline} {pipe_to_names(final_pipeline)}")
        logger.info(f"Final Accuracy: {final_acc:.4f}")
        operator_searcher.evaluator.cleanup()
        
        return final_pipeline, final_acc, stage_1_evals, stage_2_evals, {
            'mab_init_time': mab_init_time,
            'stage1_time': stage1_time,
            'stage2_time': stage2_time,
            'localset_len': len(category_shapley.local_set | operator_searcher.local_set)
        }
        

def load_diffprep_meta():
    dataset_names = sorted(os.listdir('datasets/diffprep_dataset'))
    meta: list[dict] = []
    for dataset_name in dataset_names:
        dataset_path = os.path.join('datasets/diffprep_dataset', dataset_name, 'data.csv')
        info_path = os.path.join('datasets/diffprep_dataset', dataset_name, 'info.json')
        with open(info_path, 'r') as f:
            info = json.load(f)
        label = info['label']
        meta.append({
            'name': dataset_name,
            'path': dataset_path,
            'label': label,
        })
    return meta


def run_full_enum(_metadatas, use = [], parallel=4, n_processes=4):
    logger.info("FULL ENUMERATION SHAPLEY VALUE DEMONSTRATION (OPTIMIZED CONCURRENT VERSION)")
    if len(use) == 0:
        raise Exception('No input datasets')

    metadatas = []

    for meta in _metadatas:
        if meta['name'] in use:
            metadatas.append(meta)
    
    pipeline_length = 6
    num_datasets = len(metadatas)
    logger.info(f"found {num_datasets} datasets ...")

    max_concurrent_datasets = min(mp.cpu_count() - 1, num_datasets,
                                  parallel)  # Limit to 4 to not overload system with many sub-sub-processes
    logger.info(f"using {max_concurrent_datasets} concurrent dataset processes, each with {n_processes} sub-processes for evaluation.")

    # Prepare arguments for each dataset task
    tasks = [(meta, pipeline_length, n_processes) for meta in metadatas]

    with NonDaemonicPool(processes=max_concurrent_datasets) as pool:
        pool.starmap(process_dataset, tasks)

    logger.info("all datasets processed.")


def process_dataset(dataset_name, pipeline_length, out, n_processes=4):
    global CACHE_LABEL
    log_dir = os.path.join('saves/logs', f'choosekfrom32/hie_{DT}')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{dataset_name}.log')
    logger_id = logger.add(log_path, rotation='10 MB', retention='7 days', enqueue=True)

    shapley = None 
    try:
        logger.info(f"=== start to process: {dataset_name} ===")
        if dataset_name in ALL_DATASET_NAMES:
            data = name_to_df(dataset_name, sample_size=LIMIT_ALL)
        else:
            try:
                data = load_cc18(dataset_name, limit_train=LIMIT_TRAIN, limit_all=LIMIT_ALL)
            except Exception as e:
                traceback.print_exc()
                raise ValueError('Cannot load dataset ' + dataset_name + ' ' + str(e))

        t = time.time()
        shapley = HieShapley(
            dataset_name,
            data,
            pipeline_length,
            other_cache_path=f"saves/cache_32-8/{DT}/{dataset_name}{CACHE_LABEL}.json",
            enable_pretrain=not RAND_MAB and MAB_SAMPLES > 0,
            n_processes=n_processes,
        )

        pipeline, acc, stage_1_evals, stage_2_evals, stats = shapley.run()

        if EXEC_FULL:
            acc, _, _, _ = exec_pipe(dataset_name, pipeline, limit_train=LIMIT_TRAIN)

        with open(out, 'a') as f:
            f.write(f'diffprep\t{dataset_name}\t{pipeline}\t{acc}\t{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                    f'\tsv_mcts_hie\t{OP_USE}\t{stage_1_evals}\t{stage_2_evals}'
                    f'\t{stats["stage1_time"]}\t{stats["stage2_time"]}\t{time.time()-t}\t{DT}'
                    f'\tRAND_MAB{RAND_MAB}\tMAB{MAB_SAMPLES}\tMCTS{MCTS_CNT}\tEXH_CATE_ENUM{EXHAUST_ENUM_OP_IN_CATE}\t'
                    f'{stats['localset_len']}\t{SEED}\t{n_processes}\n')
            f.flush()

        logger.info(f"=== dataset {dataset_name} process complete ===")
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error occurred at {dataset_name}: {str(e)}", exc_info=True)
    finally:
        logger.remove(logger_id)


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=str, default='diffprep', help='input the dataset set, diffprep or cc18')
    parser.add_argument('--use', type=str, required=True, help='input the dataset names, split with comma')
    parser.add_argument('--len', type=int, required=True, help='input the pipeline length')
    parser.add_argument('--mcts', type=int, default=MCTS_CNT, help='input the mcts count')
    parser.add_argument('--mab', type=int, default=MAB_SAMPLES, help='input the mab sample count')
    parser.add_argument('--n_processes', type=int, default=1, help='input the n_processes number')
    parser.add_argument('--dt', type=str, default=DT, choices=dt_dict.keys(), help='input the downstream task type')
    parser.add_argument('--rand_mab', action='store_true', help='input the enable mab flag')
    parser.add_argument('--exhaust_enum_op_in_cate', action='store_true', help='input the exhaust enum op in cate flag')
    parser.add_argument('--disable_early_stop', action='store_false', help='add this flag to disable early stop')
    parser.add_argument('--limit', default=LIMIT_TRAIN, type=int, help='limit the training sample')
    parser.add_argument('--limit_all', default=LIMIT_ALL, type=int, help='limit the total dataset')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saves/hier_shapley_ext32ops.tsv'))
    parser.add_argument('--no_dump', action='store_true', default=False)
    parser.add_argument('--exec_full', action='store_true', default=False)
    parser.add_argument('--cache_label', default=CACHE_LABEL, help='limit the training sample')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # _metadatas = get_deepline_metadatas()
    args = parse_args()
    OP_USE = f'op32'
    MCTS_CNT = args.mcts
    MAB_SAMPLES = args.mab
    DT = args.dt
    RAND_MAB = args.rand_mab
    EXHAUST_ENUM_OP_IN_CATE = args.exhaust_enum_op_in_cate
    EARLY_STOP = args.disable_early_stop
    LIMIT_TRAIN = args.limit
    LIMIT_ALL = args.limit_all
    SEED = args.seed
    NO_DUMP = args.no_dump
    EXEC_FULL = args.exec_full
    CACHE_LABEL = args.cache_label

    random.seed(SEED)
    np.random.seed(SEED)

    logger.info(f'cache label `{CACHE_LABEL}`')
    
    process_dataset(args.use, pipeline_length=args.len-3, out=args.out,
                    n_processes=args.n_processes)
