import random
from datetime import datetime
import traceback
from loguru import logger
import numpy as np
import time
from itertools import product
import pandas as pd
import psutil
import multiprocessing as mp
from typing import TypedDict, List, Dict, Tuple, Optional
from multiprocessing.pool import Pool as ProcessPool
from multiprocessing import Manager

import hashlib
from tqdm import tqdm
import sys
import os
import json
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()
sys.path.append(os.getenv('PROJ_DIR'))

from aipipe_reprod.dqn.dataloader.load_deepline import get_deepline_metadatas
from aipipe_reprod.dqn.dataloader.load_diffprep import path_to_df_dict
from aipipe_reprod.new_ql.q_action_provider import QActionProvider
from operators.types import DatasetType
from sklearn.metrics import accuracy_score
from aipipe_reprod.shapley.hierarchical.operator_info_full import (
    OPERATOR_INFO, OPS_BY_CATEGORY,
    CATEGORIES, CATEGORY_TO_IDX, IDX_TO_CATEGORY
)
from aipipe_reprod.shapley.hierarchical.mab import UCB1_Bandit


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


# --- Worker process setup ---
worker_data = None
worker_dataset_name = None
worker_bandits = None
worker_local_cache = None


def init_eval_worker(data, dataset_name, bandits, shared_cache):
    """初始化评估worker进程"""
    global worker_data, worker_dataset_name, worker_bandits, worker_local_cache
    worker_data = data
    worker_dataset_name = dataset_name
    worker_bandits = bandits
    worker_local_cache = dict(shared_cache)  # 复制共享缓存到本地


def eval_single_cate_pipe_worker(args):
    """Worker函数：评估单个category pipeline"""
    pipeline, dataset_name = args
    global worker_data, worker_dataset_name, worker_bandits, worker_local_cache
    
    try:
        concrete_pipeline = []
        bandit_updates = []
        
        # 使用bandit选择具体操作符
        for op_or_cat_idx in pipeline:
            if op_or_cat_idx < 0:
                concrete_pipeline.append(op_or_cat_idx)  # 空操作
            else:
                category = IDX_TO_CATEGORY[op_or_cat_idx]
                bandit: UCB1_Bandit = worker_bandits[category]
                selected_op = bandit.select_arm()
                
                concrete_pipeline.append(selected_op)
                bandit_updates.append({'category': category, 'arm': selected_op})

        # 评估pipeline
        acc, caches = single_operator_pipe_eval_worker(concrete_pipeline, dataset_name)
        
        # 计算reward
        cumu = []
        for update in bandit_updates:
            last_cumu = cumu.copy()
            last_acc = caches.get(f'{dataset_name}:{last_cumu}', 0.)
            cumu.append(update['arm'])
            update['reward'] = caches.get(f'{dataset_name}:{cumu}', 0.) - last_acc

        return {
            'pipeline': pipeline,
            'acc': acc,
            'bandit_updates': bandit_updates,
            'caches': caches,
            'success': True
        }
    except Exception as e:
        logger.error(f'Worker category eval error for pipeline {pipeline}: {e}')
        return {
            'pipeline': pipeline,
            'acc': 0.0,
            'bandit_updates': [],
            'caches': {},
            'success': False,
            'error': str(e)
        }


def eval_single_op_pipe_worker(args):
    """Worker函数：评估单个operator pipeline"""
    pipeline, dataset_name = args
    global worker_data, worker_dataset_name, worker_local_cache
    
    try:
        acc, caches = single_operator_pipe_eval_worker(pipeline, dataset_name)
        return {
            'pipeline': pipeline,
            'acc': acc,
            'caches': caches,
            'success': True
        }
    except Exception as e:
        logger.error(f'Worker operator eval error for pipeline {pipeline}: {e}')
        return {
            'pipeline': pipeline,
            'acc': 0.0,
            'caches': {},
            'success': False,
            'error': str(e)
        }


def single_operator_pipe_eval_worker(pipeline: list[int], dataset_name: str):
    """Worker中的单个operator pipeline评估函数"""
    global worker_data, worker_local_cache
    
    # 生成pipeline key
    reconstructed_pipe = []
    for idx, op in enumerate(pipeline):
        if op < 0:
            continue
        if op == 0 and reconstructed_pipe.__contains__(0):
            continue
        if op in [1,2,3] and set(reconstructed_pipe) & set([1,2,3]):
            continue
        if op in [4,5,6] and set(reconstructed_pipe) & set([4,5,6]):
            continue
        reconstructed_pipe.append(op)
    
    _k = f'{dataset_name}:{str(reconstructed_pipe)}'
    
    # 检查本地缓存
    if _k in worker_local_cache:
        return worker_local_cache[_k], {}
    
    try:
        # 深拷贝数据以避免修改原始数据
        d = {k: v.copy(deep=True) for k, v in worker_data.items()}

        caches: dict[str, float] = {}
        key = f'{dataset_name}:[]'
        last_acc = 0.
        try:
            lr = QActionProvider.get(QActionProvider.done_action)
            y_pred = lr.transform(d['train'], d['target'], d['test'])
            last_acc = float(accuracy_score(d['target_test'], y_pred))
            caches[key] = last_acc
        except Exception as e:
            caches[key] = 0.

        cumu_pipe = []
        # 应用管道中的每个操作
        for i, op in enumerate(pipeline):
            if op < 0:
                continue

            try:
                o = QActionProvider.get(op)

                d['train'], d['test'], d['target'] = o.transform(
                    d['train'], d['test'], d['target'])

                cumu_pipe.append(op)
                key = f'{dataset_name}:{str(cumu_pipe)}'
                lr = QActionProvider.get(QActionProvider.done_action)
                y_pred = lr.transform(d['train'], d['target'], d['test'])
                last_acc = float(accuracy_score(d['target_test'], y_pred))
                caches[key] = last_acc

            except Exception as e:
                logger.error(f'Transform failed for operator {op} at position {i}: {e}')
                return 0.0, caches

        # 更新本地缓存
        worker_local_cache[_k] = last_acc
        return last_acc, caches

    except Exception as e:
        logger.error(f'Pipeline evaluation failed for {pipeline}: {e}', exc_info=True)
        return 0.0, {}


def pipe_to_names(pipe: list[int]):
    res = '['
    for _idx, i in enumerate(pipe):
        if i < 0: 
            res += 'None'
        else:
            res += QActionProvider.get(i).get_name()
        if _idx < len(pipe) - 1:
            res += ','
    return res + ']'


# --- Parallel Evaluator ---
class ParallelPipelineEvaluator:
    def __init__(self, dataset_name: str, data: DatasetType,
                 bandits: Optional[Dict[str, UCB1_Bandit]] = None,
                 cache_path: Optional[str] = None, dump_interval: int = 10, 
                 other_cache_dir: Optional[str] = None,
                 n_processors=4):
        self.dataset_name = dataset_name
        self.bandits = bandits
        self.is_category_mode = bandits is not None
        self.cache_path = cache_path
        self.dump_interval = dump_interval
        self.batch_counter = 0
        self.data = data
        self.n_processors = n_processors

        # 使用Manager创建共享缓存
        self.manager = Manager()
        self.shared_cache = self.manager.dict()
        self.cache_lock = self.manager.Lock()

        # 本地缓存用于提高性能
        self.local_cache: Dict[str, float] = {}
        self.sync_interval = 100
        self.eval_counter = 0

        if other_cache_dir:
            self.load_other_cache(other_cache_dir)
        elif self.cache_path and os.path.exists(self.cache_path):
            self.load_cache()

        # 初始化进程池 - 用于evaluate_pipelines_cate和evaluate_pipelines_op
        self.eval_pool = None

    def _init_eval_pool(self):
        """延迟初始化评估进程池"""
        if self.eval_pool is None:
            init_args = (self.data, self.dataset_name, self.bandits, self.shared_cache)
            self.eval_pool = ProcessPool(
                processes=self.n_processors,
                initializer=init_eval_worker,
                initargs=init_args,
                maxtasksperchild=50  # 定期重启进程以防止内存泄漏
            )

    def get_pipeline_key(self, pipeline: List[int]) -> str:
        reconstructed_pipe = []
        for idx, op in enumerate(pipeline):
            if op < 0:
                continue
            if op == 0 and reconstructed_pipe.__contains__(0):
                continue
            if op in [1,2,3] and set(reconstructed_pipe) & set([1,2,3]):
                continue
            if op in [4,5,6] and set(reconstructed_pipe) & set([4,5,6]):
                continue
            reconstructed_pipe.append(op)
        return f'{self.dataset_name}:{str(reconstructed_pipe)}'
    
    def get_cate_pipeline_key(self, pipeline: List[int]) -> str:
        p = list(filter(lambda x: x >= 0, pipeline))
        return f'cat:{self.dataset_name}:{str(p)}'

    def load_cache(self):
        """加载缓存文件"""
        try:
            with open(self.cache_path, 'r') as f:
                cached_data = json.load(f)
            with self.cache_lock:
                self.shared_cache.update(cached_data)
            self.local_cache.update(cached_data)
            logger.info(f"Loaded {len(cached_data)} entries from cache at {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def load_other_cache(self, cache_dir: str):
        cache_path = os.path.join(cache_dir, f'{self.dataset_name}.json')
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            cached_data2 = {f'{key}': value for key, value in cached_data.items()}
            for k, v in cached_data.items():
                dsn, p = k.split(':')
                p = json.loads(p)
                if -1 in p:
                    p = list(filter(lambda x: x >= 0, p))
                    cached_data2[f'{dsn}:{str(p)}'] = v

            with self.cache_lock:
                self.shared_cache.update(cached_data2)
            self.local_cache.update(cached_data2)
            logger.info(f"Loaded {len(cached_data2)} entries from cache at {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def save_other_cache(self, cache_dir: str):
        cache_path = os.path.join(cache_dir, f'{self.dataset_name}.json')
        try:
            with self.cache_lock:
                cache_data = dict(self.shared_cache)
            cache_data = {key.replace('op:', '').replace('cat:', ''): value 
                          for key, value in cache_data.items() 
                          if key.startswith('op') or key.startswith('cat')}
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Saved {len(cache_data)} entries to cache at {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def dump_cache(self, path: Optional[str] = None):
        """保存缓存到文件"""
        dump_path = path or self.cache_path
        if not dump_path:
            return

        try:
            # 合并本地和共享缓存
            self._sync_caches()
            with self.cache_lock:
                cache_data = dict(self.shared_cache)

            os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            with open(dump_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Saved {len(cache_data)} entries to cache at {dump_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _sync_caches(self):
        """同步本地缓存到共享缓存"""
        if self.local_cache:
            with self.cache_lock:
                self.shared_cache.update(self.local_cache)

    def evaluate_pipelines_cate(self, pipelines: list[list[int]]) -> dict[str, float]:
        """多进程版本的category pipeline评估"""
        if not pipelines: 
            return {}

        results = {}
        try:
            # 初始化进程池
            self._init_eval_pool()
            
            # 准备参数
            args_list = [(pipe, self.dataset_name) for pipe in pipelines]
            
            # 使用进程池并行评估
            logger.info(f"Evaluating {len(pipelines)} category pipelines with {self.n_processors} processes...")
            eval_results = []
            for result in tqdm(self.eval_pool.imap_unordered(eval_single_cate_pipe_worker, args_list),
                             total=len(pipelines),
                             leave=False,
                             desc="Evaluating category pipelines"):
                eval_results.append(result)
            
            # 处理结果
            for result in eval_results:
                if result['success']:
                    pipeline = result['pipeline']
                    acc = result['acc']
                    bandit_updates = result['bandit_updates']
                    caches = result['caches']
                    
                    # 更新结果
                    pipeline_key = self.get_cate_pipeline_key(pipeline)
                    results[pipeline_key] = acc
                    
                    # 更新bandit
                    if bandit_updates:
                        for update in bandit_updates:
                            try:
                                self.bandits[update['category']].update(
                                    update['arm'], update['reward']
                                )
                            except Exception as e:
                                logger.error(f"Failed to update bandit: {e}")
                    
                    # 更新缓存
                    if caches:
                        self.local_cache.update(caches)
                        with self.cache_lock:
                            self.shared_cache.update(caches)
                else:
                    pipeline = result['pipeline']
                    pipeline_key = self.get_cate_pipeline_key(pipeline)
                    results[pipeline_key] = 0.0
                    logger.error(f"Pipeline evaluation failed: {result.get('error', 'Unknown error')}")

            return results

        except Exception as e:
            logger.error(f'Worker category eval error for pipelines: {e}')
            return {}

    def evaluate_pipelines_op(self, pipelines: list[list[int]]) -> dict[str, float]:
        """多进程版本的operator pipeline评估"""
        if not pipelines:
            return {}

        results = {}
        uncached_pipelines = []
        
        # 首先检查缓存
        for pipeline in pipelines:
            pipeline_key = self.get_pipeline_key(pipeline)

            if pipeline_key in self.local_cache:
                results[pipeline_key] = self.local_cache[pipeline_key]
            else:
                with self.cache_lock:
                    if pipeline_key in self.shared_cache:
                        # 将共享缓存中的结果复制到本地缓存
                        self.local_cache[pipeline_key] = self.shared_cache[pipeline_key]
                        results[pipeline_key] = self.shared_cache[pipeline_key]
                    else:
                        uncached_pipelines.append(pipeline)

        if not uncached_pipelines:
            return results
        
        logger.info(f"Evaluating {len(uncached_pipelines)} uncached pipelines with {self.n_processors} processes...")
        
        try:
            # 初始化进程池
            self._init_eval_pool()
            
            # 准备参数
            args_list = [(pipe, self.dataset_name) for pipe in uncached_pipelines]
            
            # 使用进程池并行评估
            eval_results = []
            for result in tqdm(self.eval_pool.imap_unordered(eval_single_op_pipe_worker, args_list),
                             total=len(uncached_pipelines),
                             leave=False,
                             desc="Evaluating operator pipelines"):
                eval_results.append(result)
            
            # 处理结果
            for result in eval_results:
                if result['success']:
                    pipeline = result['pipeline']
                    perf = result['acc']
                    caches = result['caches']
                    
                    # 更新结果和缓存
                    pipeline_key = self.get_pipeline_key(pipeline)
                    results[pipeline_key] = perf
                    self.local_cache[pipeline_key] = perf
                    
                    with self.cache_lock:
                        self.shared_cache[pipeline_key] = perf
                        if caches:
                            self.shared_cache.update(caches)
                    
                    if caches:
                        self.local_cache.update(caches)
                else:
                    pipeline = result['pipeline']
                    pipeline_key = self.get_pipeline_key(pipeline)
                    results[pipeline_key] = 0.0
                    logger.error(f"Pipeline evaluation failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f'Error in operator pipeline evaluation: {e}')

        return results

    def single_operator_pipe_eval(self, pipeline: list[int]):
        """单进程版本的评估（用于兼容性）"""
        _k = self.get_pipeline_key(pipeline)
        
        # 检查缓存
        if _k in self.local_cache:
            return self.local_cache[_k], {}
        
        with self.cache_lock:
            if _k in self.shared_cache:
                return self.shared_cache[_k], {}
        
        try:
            # 深拷贝数据以避免修改原始数据
            d = {k: v.copy(deep=True) for k, v in self.data.items()}

            caches: dict[str, float] = {}
            key = f'{self.dataset_name}:[]'
            last_acc = 0.
            try:
                lr = QActionProvider.get(QActionProvider.done_action)
                y_pred = lr.transform(d['train'], d['target'], d['test'])
                last_acc = float(accuracy_score(d['target_test'], y_pred))
                caches[key] = last_acc
            except Exception as e:
                caches[key] = 0.

            cumu_pipe = []
            # 应用管道中的每个操作
            for i, op in enumerate(pipeline):
                if op < 0:
                    continue

                try:
                    o = QActionProvider.get(op)

                    d['train'], d['test'], d['target'] = o.transform(
                        d['train'], d['test'], d['target'])

                    cumu_pipe.append(op)
                    key = f'{self.dataset_name}:{str(cumu_pipe)}'
                    lr = QActionProvider.get(QActionProvider.done_action)
                    y_pred = lr.transform(d['train'], d['target'], d['test'])
                    last_acc = float(accuracy_score(d['target_test'], y_pred))
                    caches[key] = last_acc

                except Exception as e:
                    logger.error(f'Transform failed for operator {op} at position {i}: {e}')
                    return 0.0, caches

            logger.info(f'evaluate {self.dataset_name} {pipeline} {last_acc} {pipe_to_names(pipeline)}')
            self.local_cache.update(caches)
            return last_acc, caches

        except Exception as e:
            logger.error(f'Pipeline evaluation failed for {pipeline}: {e}', exc_info=True)
            return 0.0, {}

    def cleanup(self, cache_path: str):
        """清理资源"""
        try:
            # 最终同步缓存
            self._sync_caches()
            self.dump_cache(f'{cache_path}/{self.dataset_name}.json')
            logger.info(f"Cache dumped to {cache_path}/{self.dataset_name}.json")

            # 关闭进程池
            if self.eval_pool is not None:
                self.eval_pool.close()
                self.eval_pool.join()
                self.eval_pool = None
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        finally:
            if hasattr(self, 'manager'):
                try:
                    self.manager.shutdown()
                except:
                    pass


def load_diffprep_meta():
    import glob
    metadatas = []
    paths = glob.glob('data/diffprep/*')
    for path in paths:
        # /root/ai_pipe/ai-data-pipe/data/diffprep/abalone/abalone.csv
        name = path.split('/')[-1]
        csv_path = glob.glob(f'{path}/*.csv')[0]
        label_name = csv_path.split('/')[-1].split('.')[0]
        metadatas.append({'name': name, 'path': csv_path, 'label': label_name})
    return metadatas


def load_deepline_meta():
    names_labels = get_deepline_metadatas()
    metadatas = []
    for name, label in names_labels.items():
        metadatas.append({'name': name, 'path': f'data/deepline/{name}/{name}.csv', 'label': label})
    return metadatas


class HierarchicalShapley:
    def __init__(
        self,
        dataset_name: str,
        data: DatasetType,
        pipeline_length: int = 6,
        cache_path_prefix: Optional[str] = None,
        other_cache_path: Optional[str] = None,
        enable_pretrain: bool = True,
        stage2_method: str = 'shapley',
        num_samples: int = 100,
        mab_pretrain_sample: Optional[int] = None,
        n_processors: int = 4
    ):
        self.dataset_name = dataset_name
        self.data = data
        self.pipeline_length = pipeline_length
        self.cache_path_prefix = cache_path_prefix
        self.enable_pretrain = enable_pretrain
        self.stage2_method = stage2_method
        self.num_samples = num_samples
        self.mab_pretrain_sample = mab_pretrain_sample

        # Stage 1: Category-level bandits
        self.bandits: Dict[str, UCB1_Bandit] = {}
        for category, ops in OPS_BY_CATEGORY.items():
            self.bandits[category] = UCB1_Bandit(arms=ops)

        # Stage 2: Operator-level evaluation
        self.evaluator = ParallelPipelineEvaluator(
            dataset_name=dataset_name,
            data=data,
            bandits=self.bandits,
            cache_path=f"{cache_path_prefix}_cate.json" if cache_path_prefix else None,
            other_cache_dir=other_cache_path,
            n_processors=n_processors
        )

    def initialize_bandits_with_pretraining(self, sample_n: int = 100):
        """使用预训练初始化bandits"""
        logger.info(f"Initializing bandits with {sample_n} samples per category...")
        start_time = time.time()

        for category in CATEGORIES:
            ops = OPS_BY_CATEGORY[category]
            pipelines = [[op] for op in ops]

            # 为每个操作创建sample_n个pipeline
            sampled_pipelines = []
            for op in ops:
                for _ in range(sample_n):
                    sampled_pipelines.append([op])

            # 批量评估
            results = self.evaluator.evaluate_pipelines_cate(sampled_pipelines)

            # 更新bandit
            for op in ops:
                op_results = [results[self.evaluator.get_cate_pipeline_key([op])] 
                            for _ in range(sample_n)]
                avg_reward = np.mean(op_results)
                for _ in range(sample_n):
                    self.bandits[category].update(op, avg_reward)

        init_time = time.time() - start_time
        logger.info(f"Bandit initialization completed in {init_time:.2f}s")
        return init_time

    def stage1_category_selection(self) -> Tuple[List[int], Dict]:
        """Stage 1: 使用UCB1选择category序列"""
        logger.info("Starting Stage 1: Category-level selection with UCB1...")
        start_time = time.time()

        best_pipeline = []
        best_acc = 0.0
        evaluations = 0

        # MCTS-style exploration
        for iteration in range(self.num_samples):
            # 生成category pipeline
            category_pipeline = []
            for _ in range(self.pipeline_length):
                cat_idx = random.choice(range(len(CATEGORIES)))
                category_pipeline.append(cat_idx)

            # 评估
            results = self.evaluator.evaluate_pipelines_cate([category_pipeline])
            key = self.evaluator.get_cate_pipeline_key(category_pipeline)
            acc = results.get(key, 0.0)
            evaluations += 1

            if acc > best_acc:
                best_acc = acc
                best_pipeline = category_pipeline
                logger.info(f"New best category pipeline: {best_pipeline}, acc: {best_acc:.4f}")

        stage1_time = time.time() - start_time
        logger.info(f"Stage 1 completed in {stage1_time:.2f}s with {evaluations} evaluations")

        return best_pipeline, {
            'time': stage1_time,
            'evaluations': evaluations,
            'best_acc': best_acc
        }

    def stage2_operator_refinement(self, category_pipeline: List[int]) -> Tuple[List[int], float, Dict]:
        """Stage 2: Operator-level refinement"""
        logger.info(f"Starting Stage 2: Operator-level refinement using {self.stage2_method}...")
        start_time = time.time()

        if self.stage2_method == 'shapley':
            best_pipeline, best_acc, evaluations = self._shapley_refinement(category_pipeline)
        elif self.stage2_method == 'greedy':
            best_pipeline, best_acc, evaluations = self._greedy_refinement(category_pipeline)
        else:
            raise ValueError(f"Unknown stage2_method: {self.stage2_method}")

        stage2_time = time.time() - start_time
        logger.info(f"Stage 2 completed in {stage2_time:.2f}s with {evaluations} evaluations")

        return best_pipeline, best_acc, {
            'time': stage2_time,
            'evaluations': evaluations
        }

    def _shapley_refinement(self, category_pipeline: List[int]) -> Tuple[List[int], float, int]:
        """使用Shapley值进行operator选择"""
        evaluations = 0
        position_operators = []

        for cat_idx in category_pipeline:
            category = IDX_TO_CATEGORY[cat_idx]
            operators = OPS_BY_CATEGORY[category]
            position_operators.append(operators)

        # Monte Carlo Shapley值估计
        shapley_values = {i: {op: [] for op in ops} 
                         for i, ops in enumerate(position_operators)}

        for _ in tqdm(range(self.num_samples), desc="Shapley sampling"):
            # 随机排列positions
            positions = list(range(len(category_pipeline)))
            random.shuffle(positions)

            current_pipeline = [-1] * len(category_pipeline)

            for pos in positions:
                operators = position_operators[pos]
                for op in operators:
                    # 测试加入这个operator的贡献
                    test_pipeline = current_pipeline.copy()
                    test_pipeline[pos] = op

                    results = self.evaluator.evaluate_pipelines_op([test_pipeline])
                    key = self.evaluator.get_pipeline_key(test_pipeline)
                    acc = results.get(key, 0.0)
                    evaluations += 1

                    shapley_values[pos][op].append(acc)

                # 选择最佳operator
                best_op = max(operators, 
                            key=lambda op: np.mean(shapley_values[pos][op]))
                current_pipeline[pos] = best_op

        # 构建最终pipeline
        final_pipeline = []
        for pos, ops in enumerate(position_operators):
            best_op = max(ops, key=lambda op: np.mean(shapley_values[pos][op]))
            final_pipeline.append(best_op)

        # 评估最终pipeline
        results = self.evaluator.evaluate_pipelines_op([final_pipeline])
        key = self.evaluator.get_pipeline_key(final_pipeline)
        final_acc = results.get(key, 0.0)

        return final_pipeline, final_acc, evaluations

    def _greedy_refinement(self, category_pipeline: List[int]) -> Tuple[List[int], float, int]:
        """贪心方法进行operator选择"""
        evaluations = 0
        current_pipeline = []

        for cat_idx in category_pipeline:
            category = IDX_TO_CATEGORY[cat_idx]
            operators = OPS_BY_CATEGORY[category]

            best_op = None
            best_acc = 0.0

            for op in operators:
                test_pipeline = current_pipeline + [op]
                results = self.evaluator.evaluate_pipelines_op([test_pipeline])
                key = self.evaluator.get_pipeline_key(test_pipeline)
                acc = results.get(key, 0.0)
                evaluations += 1

                if acc > best_acc:
                    best_acc = acc
                    best_op = op

            if best_op is not None:
                current_pipeline.append(best_op)

        return current_pipeline, best_acc, evaluations

    def run(self) -> Tuple[List[int], float, int, int, Dict]:
        """运行完整的层次化搜索"""
        logger.info(f"Starting Hierarchical Shapley Search for {self.dataset_name}")
        total_start = time.time()

        # 预训练阶段
        mab_init_time = 0.0
        if self.enable_pretrain:
            sample_n = self.mab_pretrain_sample or 100
            mab_init_time = self.initialize_bandits_with_pretraining(sample_n)

        # Stage 1: Category selection
        category_pipeline, stage1_stats = self.stage1_category_selection()

        # Stage 2: Operator refinement
        final_pipeline, final_acc, stage2_stats = self.stage2_operator_refinement(category_pipeline)

        total_time = time.time() - total_start

        logger.info(f"Search completed for {self.dataset_name}")
        logger.info(f"Final pipeline: {final_pipeline}")
        logger.info(f"Final accuracy: {final_acc:.4f}")
        logger.info(f"Total evaluations: {stage1_stats['evaluations'] + stage2_stats['evaluations']}")
        logger.info(f"Total time: {total_time:.2f}s")

        # 保存缓存
        if self.cache_path_prefix:
            self.evaluator.cleanup(os.path.dirname(self.cache_path_prefix))

        time_dict = {
            'mab_init_time': mab_init_time,
            'stage1_time': stage1_stats['time'],
            'stage2_time': stage2_stats['time'],
            'final_time': total_time
        }

        return (final_pipeline, final_acc, 
                stage1_stats['evaluations'], stage2_stats['evaluations'],
                time_dict)


def process_dataset(meta, pipeline_length, enable_pretrain, stage2_method, num_samples, rand_seed, mab_pretrain_sample=None):
    dataset_name = meta['name']
    log_dir = os.path.join(os.getenv('PROJ_DIR'), 'aipipe_reprod', 'shapley', 'saves', 'logs', 'hie_mix_all')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{dataset_name}.log')
    logger_id = logger.add(log_path, rotation='10 MB', retention='7 days', enqueue=True)

    try:
        logger.info(f"=== 开始处理数据集: {dataset_name} (HierarchicalShapley) ===")
        data = path_to_df_dict(meta['path'], meta['label'])

        hierarchical_search = HierarchicalShapley(
            dataset_name=dataset_name,
            data=data,
            pipeline_length=pipeline_length,
            cache_path_prefix=f"aipipe_reprod/shapley/saves/mcts_cache_mix/{dataset_name}",
            other_cache_path='aipipe_reprod/shapley/saves/cache_mix',
            enable_pretrain=enable_pretrain,
            stage2_method=stage2_method,
            num_samples=num_samples,
            mab_pretrain_sample=mab_pretrain_sample,
        )
        pipeline, acc, stage_1_evals, stage_2_evals, time_dict = hierarchical_search.run()

        mab_init_time = time_dict['mab_init_time']
        stage1_time = time_dict['stage1_time']
        stage_2_time = time_dict['stage2_time']
        final_time = time_dict['final_time']

        os.makedirs('aipipe_reprod/shapley/saves/mcts', exist_ok=True)
        with open(f'aipipe_reprod/shapley/saves/mcts/mix_ops_mcts.tsv', 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t{dataset_name}\t{num_samples}\t{rand_seed}"
                    + f"\t{mab_pretrain_sample}\t{pipeline}\t{pipeline_length}\t{acc:.4f}"
                    + f"\t{stage_1_evals}\t{stage_2_evals}"
                    + f"\t{mab_init_time:.4f}\t{stage1_time:.4f}\t{stage_2_time:.4f}\t{final_time:.4f}\n")

        logger.info(f"=== 数据集 {dataset_name} 处理完成 ===")
    except Exception as e:
        logger.error(f"处理数据集 {dataset_name} 时出错: {str(e)}", exc_info=True)
    finally:
        logger.remove(logger_id)


def process_dataset_worker(meta, pipeline_length, enable_pretrain, stage2_method, num_samples, rand_seed, mab_pretrain_sample=None):
    """
    Worker function for processing a single dataset in a separate process.
    This function is designed to be used with multiprocessing.Pool.
    """
    dataset_name = meta['name']
    
    # Set up a new logger for this worker process
    log_dir = os.path.join(os.getenv('PROJ_DIR'), 'aipipe_reprod', 'shapley', 'saves', 'logs', 'hie_mix_all')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{dataset_name}_{os.getpid()}.log')
    
    # Remove all existing handlers and add a new one for this process
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> <level>{level}</level> <cyan>{module}</cyan>:<cyan>{line}</cyan> - {message}", enqueue=True)
    logger.add(log_path, rotation='10 MB', retention='7 days', enqueue=True)
    
    try:
        logger.info(f"=== 开始处理数据集: {dataset_name} (HierarchicalShapley) - Process {os.getpid()} ===")
        data = path_to_df_dict(meta['path'], meta['label'])

        hierarchical_search = HierarchicalShapley(
            dataset_name=dataset_name,
            data=data,
            pipeline_length=pipeline_length,
            cache_path_prefix=f"aipipe_reprod/shapley/saves/mcts_cache_mix/{dataset_name}",
            other_cache_path='aipipe_reprod/shapley/saves/cache_mix',
            enable_pretrain=enable_pretrain,
            stage2_method=stage2_method,
            num_samples=num_samples,
            mab_pretrain_sample=mab_pretrain_sample,
        )
        pipeline, acc, stage_1_evals, stage_2_evals, time_dict = hierarchical_search.run()

        mab_init_time = time_dict['mab_init_time']
        stage1_time = time_dict['stage1_time']
        stage_2_time = time_dict['stage2_time']
        final_time = time_dict['final_time']

        # Create a result file for this process
        result_file = f'aipipe_reprod/shapley/saves/mcts/mix_ops_mcts.tsv'
        os.makedirs('aipipe_reprod/shapley/saves/mcts', exist_ok=True)
        with open(result_file, 'w') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t{dataset_name}\t{num_samples}\t{rand_seed}"
                    + f"\t{mab_pretrain_sample}\t{pipeline}\t{pipeline_length}\t{acc:.4f}"
                    + f"\t{stage_1_evals}\t{stage_2_evals}"
                    + f"\t{mab_init_time:.4f}\t{stage1_time:.4f}\t{stage_2_time:.4f}\t{final_time:.4f}\n")

        logger.info(f"=== 数据集 {dataset_name} 处理完成 ===")
        return {
            'dataset_name': dataset_name,
            'pipeline': pipeline,
            'accuracy': acc,
            'stage_1_evals': stage_1_evals,
            'stage_2_evals': stage_2_evals,
            'time_dict': time_dict,
            'result_file': result_file
        }
    except Exception as e:
        logger.error(f"处理数据集 {dataset_name} 时出错: {str(e)}", exc_info=True)
        return {
            'dataset_name': dataset_name,
            'error': str(e)
        }



def run_hierarchical_search(use = [], enable_pretrain=True, stage2_method='shapley', num_samples=100, rand_seed=42, 
                            mab_pretrain_sample=None, use_multiprocessing=False, n_processes=None,dataset=None):
    logger.info("STARTING HIERARCHICAL SHAPLEY SEARCH")
    if dataset is None or dataset == 'diffprep':
        _metadatas = load_diffprep_meta()
    elif dataset == 'deepline':
        _metadatas = load_deepline_meta()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    metadatas = [meta for meta in _metadatas if meta['name'] in use]
    if len(metadatas) == 0:
        raise ValueError("use must be a non-empty list")
    
    # --- 64核机器配置 ---
    pipeline_length = 6

    # Set random seeds
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    
    # Determine number of processes to use
    if use_multiprocessing:
        if n_processes is None:
            n_processes = min(len(metadatas), mp.cpu_count())
        else:
            n_processes = min(n_processes, len(metadatas), mp.cpu_count())
        
        logger.info(f"Using multiprocessing with {n_processes} processes")
        
        # Create a pool of worker processes
        with ProcessPool(processes=n_processes) as pool:
            # Prepare arguments for each worker
            args = [(meta, pipeline_length, enable_pretrain, stage2_method, num_samples, rand_seed, mab_pretrain_sample) 
                   for meta in metadatas]
            
            # Process datasets in parallel
            results = list(tqdm(pool.starmap(process_dataset_worker, args), total=len(metadatas), desc="Processing datasets"))
            
            # Merge result files
            main_result_file = 'aipipe_reprod/shapley/saves/mcts/mix_ops_mcts.tsv'
            with open(main_result_file, 'a') as main_f:
                for result in results:
                    if 'error' not in result and 'result_file' in result:
                        # Copy content from worker result file to main result file
                        if os.path.exists(result['result_file']):
                            with open(result['result_file'], 'r') as worker_f:
                                main_f.write(worker_f.read())
                            # Remove worker result file
                            os.remove(result['result_file'])
                    elif 'error' in result:
                        logger.error(f"Failed to process dataset {result['dataset_name']}: {result['error']}")
    else:
        logger.info("Using sequential processing")
        # Sequential processing (original behavior)
        for meta in metadatas:
            process_dataset(meta, pipeline_length, enable_pretrain, 
                           stage2_method, num_samples, rand_seed, mab_pretrain_sample=mab_pretrain_sample)

    logger.info("所有数据集处理完成！")

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use', type=str, required=True, help='Comma-separated dataset names')
    parser.add_argument('--enable_pretrain', type=int, default=1, help='Enable bandit pre-training (1 or 0)')
    parser.add_argument('--stage2_method', type=str, default='shapley', choices=['shapley', 'greedy'], help='Method for Stage 2 search')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of Monte Carlo samples (K) for Stage 2 Shapley')
    parser.add_argument('--use_multiprocessing', type=int, default=1, help='Use multiprocessing for parallel processing (1 or 0)')
    parser.add_argument('--n_processes', type=int, default=4, help='Number of processes to use (default: number of CPUs)')
    parser.add_argument('--dataset', type=str, default='diffprep', choices=['diffprep', 'deepline'], help='Dataset to use')
    args = parser.parse_args()
    args.use = list(map(lambda x: x.strip(), args.use.split(',')))
    args.enable_pretrain = bool(args.enable_pretrain)
    args.use_multiprocessing = bool(args.use_multiprocessing)
    return args

if __name__ == '__main__':
    args = parse_args()
    # for rand_seed in [42, 64, 128, 256, 512, 1024, 2048, 2025, 65537]:
    for num_samples in range(100, 101, 10):
        rand_seed=42
        np.random.seed(rand_seed)
        random.seed(rand_seed)
        run_hierarchical_search(
            use=args.use,
            enable_pretrain=args.enable_pretrain,
            stage2_method=args.stage2_method,
            num_samples=num_samples,
            rand_seed=rand_seed,
            mab_pretrain_sample=2000,
            use_multiprocessing=args.use_multiprocessing,
            n_processes=args.n_processes,
            dataset=args.dataset,
        )
