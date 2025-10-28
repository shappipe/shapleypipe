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
worker_bandits = None  # Worker will hold a reference to the shared bandits


def init_worker(data, dataset_name, bandits):
    global worker_data, worker_dataset_name, worker_bandits
    worker_data = data
    worker_dataset_name = dataset_name
    worker_bandits = bandits



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
    def __init__(self, dataset_name: str, data: DatasetType, # n_processes: int,
                 bandits: Optional[Dict[str, UCB1_Bandit]] = None,
                 cache_path: Optional[str] = None, dump_interval: int = 10, 
                 other_cache_dir: Optional[str] = None,
                 n_processors=4):
        self.dataset_name = dataset_name
        # self.n_processes = n_processes
        self.bandits = bandits
        self.is_category_mode = bandits is not None
        self.cache_path = cache_path
        self.dump_interval = dump_interval
        self.batch_counter = 0
        self.data = data
        self.n_processors = n_processors

        # 本地缓存用于提高性能
        self.local_cache: Dict[str, float] = {}
        self.sync_interval = 100
        self.eval_counter = 0

        # self.manager = mp.Manager()
        # self.shared_cache = self.manager.dict()
        # self.cache_lock = self.manager.Lock()
        self.shared_cache: dict[str, float] = {}

        if other_cache_dir:
            self.load_other_cache(other_cache_dir)
        elif self.cache_path and os.path.exists(self.cache_path):
            self.load_cache()

        # 初始化进程池
        init_args = (data, dataset_name, self.bandits)
        self.pool = ProcessPool(
            processes=self.n_processors,
            initializer=init_worker,
            initargs=init_args,
            maxtasksperchild=100
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

            self.shared_cache.update(cached_data2)
            self.local_cache.update(cached_data2)
            logger.info(f"Loaded {len(cached_data2)} entries from cache at {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def save_other_cache(self, cache_dir: str):
        cache_path = os.path.join(cache_dir, f'{self.dataset_name}.json')
        try:
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
            self.shared_cache.update(self.local_cache)

    def evaluate_pipelines_cate(self, pipelines: list[list[int]]) -> dict[str, float]:
        if not pipelines: return {}

        results = {}
        try:
            performances = []
            all_bandit_updates = []
            all_caches = []

            for pipe in tqdm(pipelines, leave=False):
                acc, bandit_updates, caches = self.eval_single_cate_pipe(pipe)
                performances.append(acc)
                all_bandit_updates.append(bandit_updates)
                all_caches.append(caches)
            
            for pipeline, performance in zip(pipelines, performances):
                pipeline_key = self.get_cate_pipeline_key(pipeline)
                # self.local_cache[pipeline_key] = performance
                results[pipeline_key] = performance
            
            # for cache in all_caches:
            #     if cache:
            #         self.local_cache.update(cache)

            for update_list in all_bandit_updates:
                if update_list:
                    for update in update_list:
                        try:
                            self.bandits[update['category']].update(
                                update['arm'], update['reward']
                            )
                        except Exception as e:
                            logger.error(f"Failed to update bandit: {e}")
            
            # self.eval_counter += len(pipelines)
            # if self.eval_counter >= self.sync_interval:
            #     self._sync_caches()
            #     self.eval_counter = 0

            return results

        except Exception as e:
            logger.error(f'Worker category eval error for pipelines: {e}')
            return {}

    def eval_single_cate_pipe(self, pipeline: list[int]) -> float:
        try:
            concrete_pipeline = []
            bandit_updates = []
            for op_or_cat_idx in pipeline:
                if op_or_cat_idx < 0:
                    concrete_pipeline.append(op_or_cat_idx)  # 空操作
                else:
                    category = IDX_TO_CATEGORY[op_or_cat_idx]
                    bandit = self.bandits[category]

                    # Use MAB to select the best arm (operator) for this category
                    selected_op = bandit.select_arm()
                    concrete_pipeline.append(selected_op)

                    # Record which arm was pulled for this category
                    bandit_updates.append({'category': category, 'arm': selected_op})

            logger.info(f"Evaluating pipeline {concrete_pipeline}, {pipe_to_names(concrete_pipeline)}")
            acc, caches = self.single_operator_pipe_eval(concrete_pipeline)
            
            cumu = []
            for update in bandit_updates:
                last_cumu = cumu.copy()
                last_acc = caches.get(f'{self.dataset_name}:{last_cumu}', 0.)
                cumu.append(update['arm'])
                update['reward'] = caches.get(f'{self.dataset_name}:{cumu}', 0.) - last_acc

            return acc, bandit_updates, caches
        except Exception as e:
            logger.error(f'Worker category eval error for pipeline {pipeline}: {e}')
            return 0.0, [], None

    def evaluate_pipelines_op(self, pipelines: list[list[int]]) -> dict[str, float]:
        if not pipelines:
            return {}

        results = {}
        uncached_pipelines = []
        for pipeline in pipelines:
            pipeline_key = self.get_pipeline_key(pipeline)

            if pipeline_key in self.local_cache:
                results[pipeline_key] = self.local_cache[pipeline_key]
            elif pipeline_key in self.shared_cache:
                # 将共享缓存中的结果复制到本地缓存
                self.local_cache[pipeline_key] = self.shared_cache[pipeline_key]
                results[pipeline_key] = self.shared_cache[pipeline_key]
            else:
                uncached_pipelines.append(pipeline)

        if not uncached_pipelines:
            return results
        
        logger.info(f"Evaluating {len(uncached_pipelines)} uncached pipelines...")
        all_results = []
        for pipeline in tqdm(uncached_pipelines, leave=False):
            logger.info(f"Evaluating pipeline {pipeline}, {pipe_to_names(pipeline)}")
            all_results.append(self.single_operator_pipe_eval(pipeline))

        performances = [res[0] for res in all_results]
        all_caches = [res[1] for res in all_results]
        for pipeline, perf, caches in zip(uncached_pipelines, performances, all_caches):
            pipeline_key = self.get_pipeline_key(pipeline)
            results[pipeline_key] = perf
            self.local_cache[pipeline_key] = perf
            self.shared_cache[pipeline_key] = perf
            if caches:
                self.local_cache.update(caches)
                self.shared_cache.update(caches)

        return results

    def single_operator_pipe_eval(self, pipeline: list[int]):
        _k = self.get_pipeline_key(pipeline)
        if _k in self.shared_cache:
            return self.shared_cache[_k], {}
        
        if _k in self.local_cache:
            return self.local_cache[_k], {}
        
        try:
            # 深拷贝数据以避免修改原始数据
            d = {k: v.copy(deep=True) for k, v in self.data.items()}

            caches: dict[str, float] = {}
            key = f'{self.dataset_name}:[]'
            if key not in self.shared_cache and key not in self.local_cache:
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

                    # cumu_pipe.append(op)
                    # key = f'{self.dataset_name}:{str(cumu_pipe)}'
                    # lr = QActionProvider.get(QActionProvider.done_action)
                    # y_pred = lr.transform(d['train'], d['target'], d['test'])
                    # last_acc = float(accuracy_score(d['target_test'], y_pred))
                    # caches[key] = last_acc

                except Exception as e:
                    logger.error(f'Transform failed for operator {op} at position {i}: {e}')
                    return 0.0, caches

            lr = QActionProvider.get(QActionProvider.done_action)
            y_pred = lr.transform(d['train'], d['target'], d['test'])
            last_acc = float(accuracy_score(d['target_test'], y_pred))
            caches[key] = last_acc

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

            # if hasattr(self, 'pool'):
            #     self.pool.close()
            #     self.pool.join()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        # finally:
            # if hasattr(self, 'manager'):
            #     try:
            #         self.manager.shutdown()
            #     except:
            #         pass

# --- PermShapley ---
class PermShapley:
    def __init__(self, evaluator: ParallelPipelineEvaluator, pipeline_length: int,
                 is_category_mode: bool = False, num_mcts_samples = 50):
        self.evaluator = evaluator
        self.M = pipeline_length
        self.is_category_mode = is_category_mode
        self.baseline_idx = -1
        self.num_mcts_samples = num_mcts_samples

        if is_category_mode:
            self.players = CATEGORIES
            self.N = len(CATEGORIES)
        else:
            self.players = list(OPERATOR_INFO.keys())
            self.N = len(OPERATOR_INFO)

        logger.info(f"PermShapley initialized. Mode={'Category' if is_category_mode else 'Operator'}. Players={self.N}")

    def calc_shapley_value_full_cate(self, slot_pos: int, target_player_idx: int,
                                previous_sels: List[int]) -> tuple[float, int]:
        """计算完整的Shapley值"""
        remaining_slots = self.M - slot_pos - 1

        if remaining_slots < 0:
            logger.warning(f"Invalid remaining slots: {remaining_slots}")
            return 0.0, 0

        pipelines_to_eval = []
        # coalitions = list(product(range(self.N), repeat=remaining_slots)) # 3^2=9
        coalitions = [random.choices(range(self.N), k=remaining_slots) for _ in range(self.num_mcts_samples)] # 3^2=9

        for coalition in coalitions:
            coalition_list = list(coalition)
            pipelines_to_eval.append(previous_sels + [target_player_idx] + coalition_list)
            pipelines_to_eval.append(previous_sels + [self.baseline_idx] + coalition_list)

        try:
            results = self.evaluator.evaluate_pipelines_cate(pipelines_to_eval)
            shapley_value = 0.0
            valid_comparisons = 0

            for coalition in coalitions:
                coalition_list = list(coalition)
                target_key = self.evaluator.get_cate_pipeline_key(
                    previous_sels + [target_player_idx] + coalition_list
                )
                baseline_key = self.evaluator.get_cate_pipeline_key(
                    previous_sels + [self.baseline_idx] + coalition_list
                )

                target_perf = results.get(target_key, 0.0)
                baseline_perf = results.get(baseline_key, 0.0)

                marginal = target_perf - baseline_perf
                shapley_value += marginal
                valid_comparisons += 1

            return shapley_value / valid_comparisons if valid_comparisons > 0 else 0.0, len(pipelines_to_eval)

        except Exception as e:
            logger.error(f"Shapley calculation failed: {e}")
            return 0.0, 0

    def run_full_enumerate_algorithm_cate(self) -> List[int]:
        """运行完整枚举算法"""
        logger.info(
            f"\n{'=' * 20} STARTING FULL ENUMERATION "
            f"({'CATEGORY' if self.is_category_mode else 'OPERATOR'} LEVEL) {'=' * 20}"
        )

        selected_pipeline_indices = []
        total_evals = 0
        _shapley_store = []

        for slot_pos in range(self.M):
            best_player_idx = -1
            best_shapley = float('-inf')

            logger.info(f"Processing slot {slot_pos + 1}/{self.M}")

            for player_idx in range(self.N):
                try:
                    shapley_val, num_evals = self.calc_shapley_value_full_cate(
                        slot_pos, player_idx, selected_pipeline_indices
                    )
                    total_evals += num_evals

                    player_name = (IDX_TO_CATEGORY.get(player_idx, player_idx)
                                   if self.is_category_mode else player_idx)
                    logger.info(
                        f"  Slot {slot_pos + 1}, Player {player_idx} ({player_name}): "
                        f"Shapley = {shapley_val:.4f}"
                    )
                    _shapley_store.append({
                        'dataset': self.evaluator.dataset_name,
                        'slot_pos': slot_pos,
                        'player_idx': player_idx,
                        'shapley_val': shapley_val,
                    })

                    if shapley_val > best_shapley:
                        best_shapley = shapley_val
                        best_player_idx = player_idx

                except Exception as e:
                    logger.error(f"Error calculating Shapley for player {player_idx}: {e}")
                    continue

            if best_player_idx == -1:
                logger.warning(f"No valid player found for slot {slot_pos + 1}, using random")
                best_player_idx = np.random.choice(self.N)

            selected_pipeline_indices.append(best_player_idx)
            best_name = (IDX_TO_CATEGORY.get(best_player_idx, best_player_idx)
                         if self.is_category_mode else best_player_idx)
            logger.info(
                f"--> Slot {slot_pos + 1} Selected: {best_player_idx} ({best_name}), "
                f"Shapley = {best_shapley:.4f}"
            )
        os.makedirs('aipipe_reprod/shapley/saves/shapley_values/mixed', exist_ok=True)
        pd.DataFrame(_shapley_store).to_csv(
            f'aipipe_reprod/shapley/saves/shapley_values/mixed/{self.evaluator.dataset_name}.tsv',
            index=False, mode='a', sep='\t'
        )
        return selected_pipeline_indices, total_evals


# --- GreedySearcher ---
class GreedySearcher:
    def __init__(self, evaluator: ParallelPipelineEvaluator, bandits: Dict[str, UCB1_Bandit] = None):
        self.evaluator = evaluator
        self.bandits = bandits

    def find_best_pipeline(self, category_pipeline: List[str]) -> List[int]:
        logger.info(f"\n{'=' * 20} STARTING GREEDY SEARCH WITHIN CATEGORY PIPELINE {'=' * 20}")
        logger.info(f"Category sequence: {[c for c in category_pipeline]}")

        best_op_pipeline = []
        for i, category in enumerate(category_pipeline):
            # --- MAB WARM-START ---
            # If we have bandit info, use the best-known arm as a strong candidate
            if self.bandits:
                best_known_op = self.bandits[category].get_best_arm()
                logger.info(f"  MAB suggests best op for {category} is {best_known_op}")

            # The rest of the logic is a simple greedy evaluation, as before
            best_op_for_slot = -1
            best_performance = -1.0
            candidate_ops = OPS_BY_CATEGORY[category]
            logger.info(f"  Searching in Slot {i + 1} (Category: {category}) among {len(candidate_ops)} ops.")

            pipelines_to_eval = [best_op_pipeline + [op] for op in candidate_ops]
            results = self.evaluator.evaluate_pipelines_op(pipelines_to_eval)

            for op in candidate_ops:
                key = self.evaluator.get_pipeline_key(best_op_pipeline + [op])
                performance = results.get(key, 0.0)
                if performance > best_performance:
                    best_performance = performance
                    best_op_for_slot = op

            best_op_pipeline.append(best_op_for_slot)
            logger.info(f"--> Slot {i + 1} Selected Op: {best_op_for_slot} (Acc: {best_performance:.4f})")

        return best_op_pipeline

class ConstrainedPermShapley:
    """
    A generalized PermShapley searcher that can operate on either categories or
    operators within a constrained search space defined by a category pipeline.
    """
    def __init__(self, evaluator: ParallelPipelineEvaluator, pipeline_length: int,
                 mode: str, category_pipeline: Optional[List[str]] = None,
                 num_samples: int = 100): #
        """
        Args:
            evaluator: The parallel evaluator instance.
            pipeline_length: The length of the pipeline to build.
            mode: Either 'category' or 'operator'.
            category_pipeline: A list of category names. Required if mode is 'operator'.
            num_samples: The number of samples to use for the Monte Carlo approximation.
        """
        self.evaluator = evaluator
        self.M = pipeline_length
        self.mode = mode
        self.category_pipeline = category_pipeline
        self.baseline_idx = -1
        self.num_samples = num_samples # 蒙特卡洛采样次数
        
        if self.mode == 'category':
            self.player_sets = [list(range(len(CATEGORIES))) for _ in range(self.M)]
            logger.info(f"Shapley initialized. Mode=Category. Players per slot: ~{len(CATEGORIES)}")
        elif self.mode == 'operator':
            if not category_pipeline or len(category_pipeline) != self.M:
                raise ValueError("A category pipeline of correct length is required for operator mode.")
            self.player_sets = [OPS_BY_CATEGORY[cat] for cat in category_pipeline]
            logger.info(f"Shapley initialized. Mode=Operator. Players per slot: {[len(s) for s in self.player_sets]}")
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _get_player_name(self, player_idx: int) -> str:
        if self.mode == 'category':
            return IDX_TO_CATEGORY.get(player_idx, str(player_idx))
        return QActionProvider.get(player_idx).get_name()

    def calc_shapley_value(self, slot_pos: int, target_player: int,
                             previous_sels: List[int]) -> tuple[float, int]:
        """
            Calculates the APPROXIMATE Shapley value using Monte Carlo sampling.
        """
        remaining_slots = self.M - slot_pos - 1
        if remaining_slots < 0:
            return 0.0

        # 如果是最后一个slot，没有未来路径，直接评估
        if remaining_slots == 0:
            pipelines_to_eval = [
                previous_sels + [target_player],
                previous_sels + [self.baseline_idx]
            ]
            results = self.evaluator.evaluate_pipelines_op(pipelines_to_eval)
            target_key = self.evaluator.get_pipeline_key(pipelines_to_eval[0])
            baseline_key = self.evaluator.get_pipeline_key(pipelines_to_eval[1])
            target_perf = results.get(target_key, 0.0)
            baseline_perf = results.get(baseline_key, 0.0)
            return target_perf - baseline_perf, 0

        # 否则，使用蒙特卡洛采样
        pipelines_to_eval = []
        # 我们一次性生成所有需要的随机未来路径
        for _ in range(self.num_samples):
            # 1. 随机采样一条未来路径 c
            future_path = []
            for k in range(slot_pos + 1, self.M):
                candidate_set = self.player_sets[k]
                if not candidate_set: # 防御性编程，如果某个类别没算子
                    logger.warning(f"Empty candidate set for slot {k}. Skipping this sample.")
                    continue
                future_path.append(random.choice(candidate_set))
            
            if len(future_path) != remaining_slots:
                continue # 如果采样失败则跳过

            # 2. 为这条路径创建目标流水线和基线流水线
            pipelines_to_eval.append(previous_sels + [target_player] + future_path)
            pipelines_to_eval.append(previous_sels + [self.baseline_idx] + future_path)

        try:
            # 批量评估所有采样生成的流水线
            results = self.evaluator.evaluate_pipelines_op(pipelines_to_eval)
            
            total_marginal_contribution = 0.0
            num_valid_samples = 0
            
            # 遍历我们生成的批次，成对处理
            for i in range(0, len(pipelines_to_eval), 2):
                target_pipe = pipelines_to_eval[i]
                baseline_pipe = pipelines_to_eval[i+1]

                target_key = self.evaluator.get_pipeline_key(target_pipe)
                baseline_key = self.evaluator.get_pipeline_key(baseline_pipe)
                
                target_perf = results.get(target_key)
                baseline_perf = results.get(baseline_key)
                
                # 确保两个评估都成功了
                if target_perf is not None and baseline_perf is not None:
                    total_marginal_contribution += (target_perf - baseline_perf)
                    num_valid_samples += 1

            if num_valid_samples == 0:
                logger.warning("No valid samples were evaluated for Shapley calculation. Returning 0.")
                return 0.0
            
            # 3. 求平均
            return total_marginal_contribution / num_valid_samples, len(pipelines_to_eval)

        except Exception as e:
            logger.error(f"Shapley calculation with sampling failed: {e}", exc_info=True)
            return 0.0, 0

    def run_search(self) -> List[int]:
        """
        Runs the full enumeration algorithm to find the best pipeline.
        """
        logger.info(f"\n{'=' * 20} STARTING CONSTRAINED SHAPLEY SEARCH (Mode: {self.mode.upper()}) {'=' * 20}")

        total_evals = 0

        _shapley_store = []

        selected_pipeline = []
        for slot_pos in range(self.M):
            best_player = -1
            best_shapley = float('-inf')

            # The set of candidate players for the current slot is constrained
            candidate_players = self.player_sets[slot_pos]
            logger.info(f"Processing slot {slot_pos + 1}/{self.M}. Candidates: {len(candidate_players)}")

            for player in candidate_players:
                shapley_val, num_evals = self.calc_shapley_value(
                    slot_pos, player, selected_pipeline
                )
                total_evals += num_evals

                player_name = self._get_player_name(player)
                logger.info(f"  Slot {slot_pos + 1}, Player '{player_name}' ({player}): Shapley = {shapley_val:.4f}")

                _shapley_store.append({
                    'dataset': self.evaluator.dataset_name,
                    'slot_pos': slot_pos,
                    'player_idx': player,
                    'shapley_val': shapley_val,
                })

                if shapley_val > best_shapley:
                    best_shapley = shapley_val
                    best_player = player
            
            if best_player == -1:
                logger.warning(f"No valid player found for slot {slot_pos + 1}, using random choice from candidates.")
                best_player = random.choice(candidate_players)

            selected_pipeline.append(best_player)
            best_name = self._get_player_name(best_player)
            logger.info(
                f"--> Slot {slot_pos + 1} Selected: '{best_name}' ({best_player}), Shapley = {best_shapley:.4f}"
            )

        os.makedirs('aipipe_reprod/shapley/saves/shapley_values/mixed', exist_ok=True)
        pd.DataFrame(_shapley_store).to_csv(
            f'aipipe_reprod/shapley/saves/shapley_values/mixed/{self.evaluator.dataset_name}_stage_2.tsv',
            index=False, mode='a', sep='\t',
        )

        return selected_pipeline, total_evals



# --- 主协调器 HierarchicalShapley  ---
class HierarchicalShapley:
    def __init__(self, dataset_name: str, data: DatasetType, pipeline_length: int, 
                 other_cache_path: Optional[str] = None, enable_pretrain: bool = True, stage2_method='shapley',
                 num_samples: int = 100, mab_pretrain_sample=None, cache_path_prefix=None):
        self.dataset_name = dataset_name
        self.data = data
        self.pipeline_length = pipeline_length

        self.cache_path_prefix = cache_path_prefix
        if self.cache_path_prefix is not None:
            os.makedirs(self.cache_path_prefix, exist_ok=True)
            
        self.other_cache_path = other_cache_path
        if self.other_cache_path is None:
            os.makedirs(self.other_cache_path, exist_ok=True)

        self.enable_pretrain = enable_pretrain
        self.stage2_method = stage2_method
        self.num_samples = num_samples # 蒙特卡洛采样次数
        self.mab_pretrain_sample = mab_pretrain_sample

    def run(self):
        # --- 初始化共享的 MAB 实例 ---
        # manager = mp.Manager()
        bandits = {
            category: UCB1_Bandit(arms=ops)
            for category, ops in OPS_BY_CATEGORY.items()
        }

        # 预先训练多臂老虎机
        if self.enable_pretrain:
            logger.info('启用预训练')
            self._pretrain_bandits(bandits, self.mab_pretrain_sample)
        else:
            logger.info('不启用预训练')

        # --- 阶段一: MAB引导的类别搜索 ---
        logger.info(('#' * 10) + f' num_samples: {self.num_samples} ' + ('#' * 10))
        logger.info(f"\n{'#' * 60}\n### STAGE 1: MAB-GUIDED SEARCH FOR BEST CATEGORY PIPELINE ###\n{'#' * 60}")
        logger.info(('=' * 10) + ' create stage 1 evaluator ' + ('=' * 10))
        category_evaluator = ParallelPipelineEvaluator(
            self.dataset_name, self.data,
            bandits=bandits,  # Pass shared bandits
            cache_path=f"{self.cache_path_prefix}_category_cache_mab.json",   # not used if given `other_cache_dir`
            other_cache_dir=self.other_cache_path
        )
        t0 = time.time()
        logger.info(('=' * 10) + ' create category shapley ' + ('=' * 10))
        category_shapley = PermShapley(category_evaluator, self.pipeline_length, is_category_mode=True, num_mcts_samples=self.num_samples)
        mab_init_time = time.time() - t0
        logger.info(('=' * 10) + ' run category shapley ' + ('=' * 10))
        t1 = time.time()
        best_category_indices, stage_1_evals = category_shapley.run_full_enumerate_algorithm_cate()
        stage1_time = time.time() - t1
        logger.info(('=' * 10) + f' got best category indices ' + ('=' * 10) + str(best_category_indices))
        logger.info(('=' * 10) + f'stage 1: {stage_1_evals} evals ' + ('=' * 10))
        # category_searcher = ConstrainedPermShapley(category_evaluator, self.pipeline_length, mode='category')
        # best_category_indices = category_searcher.run_search()
        best_category_pipeline = [IDX_TO_CATEGORY[i] for i in best_category_indices]

        logger.info(f"STAGE 1 RESULT: Best category pipeline found: {[c for c in best_category_pipeline]}")
        category_evaluator.cleanup(self.other_cache_path)
        # category_evaluator.save_other_cache(self.other_cache_path)

        # --- 阶段二: MAB热启动的贪心搜索 ---
        logger.info(f"\n{'#' * 60}\n### STAGE 2: MAB-WARM-STARTED GREEDY SEARCH ###\n{'#' * 60}")
        logger.info(('=' * 10) + ' create stage 2 evaluator ' + ('=' * 10))
        t2 = time.time()
        operator_evaluator = ParallelPipelineEvaluator(
            self.dataset_name, self.data,
            cache_path=f"{self.cache_path_prefix}_operator_cache.json",
            other_cache_dir=self.other_cache_path
        )
        # greedy_searcher = GreedySearcher(operator_evaluator, bandits=bandits)
        # final_pipeline = greedy_searcher.find_best_pipeline(best_category_pipeline)

        if self.stage2_method == 'shapley':
            # 使用蒙特卡洛采样近似Shapley值
            logger.info(('=' * 10) + ' create operator shapley ' + ('=' * 10))
            operator_searcher = ConstrainedPermShapley(
                operator_evaluator,
                self.pipeline_length,
                mode='operator',
                category_pipeline=best_category_pipeline,
                num_samples=self.num_samples
            )
            logger.info(('=' * 10) + ' run operator shapley ' + ('=' * 10))
            final_pipeline, stage_2_evals = operator_searcher.run_search()
            logger.info(('=' * 10) + ' got best operator pipeline ' + ('=' * 10) + str(final_pipeline))
            logger.info(('=' * 10) + ' total evals in stage 2 ' + ('=' * 10) + str(stage_2_evals))
        
        elif self.stage2_method == 'greedy':
            # This is your original GreedySearcher logic
            stage_2_evals = 0
            greedy_searcher = GreedySearcher(operator_evaluator)
            final_pipeline = greedy_searcher.find_best_pipeline(best_category_pipeline)
        
        else:
            raise ValueError(f"Unknown stage 2 method: {self.stage2_method}")
        
        stage2_time = time.time() - t2

        t3 = time.time()
        final_performance_dict = operator_evaluator.evaluate_pipelines_op([final_pipeline])
        final_accuracy = list(final_performance_dict.values())[0] if final_performance_dict else 0.0
        final_time = time.time() - t3

        logger.info(f"\n{'#' * 60}\n### HIERARCHICAL SEARCH COMPLETED ###\n{'#' * 60}")
        logger.info(f"Final Operator Pipeline: {final_pipeline} {pipe_to_names(final_pipeline)}")
        logger.info(f"Final Accuracy: {final_accuracy:.4f}")
        operator_evaluator.cleanup(self.other_cache_path)
        # manager.shutdown()

        return final_pipeline, final_accuracy, stage_1_evals, stage_2_evals, {
            'mab_init_time': mab_init_time,
            'stage1_time': stage1_time,
            'stage2_time': stage2_time,
            'final_time': final_time,
        }
    
    def _pretrain_bandits(self, bandits: Dict[str, UCB1_Bandit], sample_size=None):
        # 为每个类别初始化一个简单的随机策略
        # with open(os.path.join('aipipe_reprod/shapley/saves/cache', f'{self.dataset_name}.json.bak'), 'r') as f:
        #     cache = json.load(f)
        all_cache_filename = sorted(list(filter(lambda x: x.endswith('.json.bak'), os.listdir('aipipe_reprod/shapley/saves/cache'))))
        cache = {}
        for filename in all_cache_filename:
            with open(os.path.join('aipipe_reprod/shapley/saves/cache', filename), 'r') as f:
                cache.update(json.load(f))
        
        # 该缓存将-1的算子去掉
        cache2 = {}
        for k, v in cache.items():
            dsn, pipe_str = k.split(':')
            pipe: list[int] = json.loads(pipe_str)
            pipe = list(filter(lambda x: x >= 0, pipe))
            pipe = list(map(lambda x: x+7, pipe))  # 加载了旧版 cache，算子 id 偏移了 7
            cache2[tuple(pipe)] = v
        
        cache2_list = list(cache2.items())
        cache2_3len_list = [item for item in cache2_list if len(item[0]) == 3]
        if sample_size is not None:
            cache2_3len_list = random.sample(cache2_3len_list, sample_size)
        del cache

        # 取 [A, B, C] 和 [A, B] 计算 C 的边际贡献
        for i in range(len(cache2_3len_list)):
            pipe, acc = cache2_3len_list[i]
            pipe_pop_last = pipe[:2]
            pop_acc = cache2.get(tuple(pipe_pop_last), 0.)
            target_opid = pipe[-1]
            cate_id = QActionProvider.idx_to_factory_id(target_opid)
            bandits[IDX_TO_CATEGORY[cate_id]].update(target_opid, acc - pop_acc)

        # 取 [A, B, C] 和 [A, C] 计算 B 的边际贡献
        for i in range(len(cache2_3len_list)):
            pipe, acc = cache2_3len_list[i]
            pipe_pop_mid = pipe[:1] + pipe[2:]
            pop_acc = cache2.get(tuple(pipe_pop_mid), 0.)
            target_opid = pipe[1]
            cate_id = QActionProvider.idx_to_factory_id(target_opid)
            bandits[IDX_TO_CATEGORY[cate_id]].update(target_opid, acc - pop_acc)

        # 取 [A, B, C] 和 [B, C] 计算 A 的边际贡献
        for i in range(len(cache2_3len_list)):
            pipe, acc = cache2_3len_list[i]
            pipe_pop_first = pipe[1:]
            pop_acc = cache2.get(tuple(pipe_pop_first), 0.)
            target_opid = pipe[0]
            cate_id = QActionProvider.idx_to_factory_id(target_opid)
            bandits[IDX_TO_CATEGORY[cate_id]].update(target_opid, acc - pop_acc)


def load_diffprep_meta():
    dataset_names = sorted(os.listdir('datasets/diffprep_dataset'))
    meta: list[dict] = []
    for dataset_name in dataset_names:
        dataset_path = os.path.join('datasets/diffprep_dataset', dataset_name, 'data.csv')
        info_path = os.path.join('datasets/diffprep_dataset', dataset_name, 'info.json')
        with open(info_path, 'r') as f: info = json.load(f)
        label = info['label']
        meta.append({'name': dataset_name, 'path': dataset_path, 'label': label})
    return meta


def load_deepline_meta():
    metadatas = get_deepline_metadatas()
    return metadatas


def process_dataset(meta, pipeline_length, enable_pretrain, stage2_method, num_samples, rand_seed, mab_pretrain_sample=None):
    dataset_name = meta['name']
    log_dir = os.path.join(os.getenv('PROJ_DIR'), 'aipipe_reprod', 'shapley', 'saves', 'logs', 'hie_mix')
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
            other_cache_path='aipipe_reprod/shapley/saves/cache_mix_util',
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
    # parser.add_argument('--use', type=str, required=True, help='Comma-separated dataset names')
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

    num_samples = 75
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
