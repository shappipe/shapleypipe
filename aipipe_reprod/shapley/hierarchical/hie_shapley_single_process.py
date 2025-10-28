from __future__ import annotations
import random
from datetime import datetime
import traceback
from loguru import logger
import numpy as np
import time
from itertools import product
import pandas as pd
import psutil
# import multiprocessing as mp
from typing import TypedDict, List, Dict, Tuple, Optional
# from multiprocessing.pool import Pool as ProcessPool

import hashlib
from tqdm import tqdm
import sys
import os
import json
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()
sys.path.append(os.getenv('PROJ_DIR'))

from aipipe_reprod.dqn.dataloader.load_diffprep import path_to_df_dict
from aipipe_reprod.shapley.q_action_provider import QActionProvider
from operators.types import DatasetType
from sklearn.metrics import accuracy_score
from aipipe_reprod.primitives import ImputerCatPrim, ImputerMeanPrim, OneHotEncoderPrim, NumericDataPrim
from aipipe_reprod.shapley.hierarchical.operator_info import (
    OPERATOR_INFO, OPS_BY_CATEGORY,
    CATEGORIES, CATEGORY_TO_IDX, IDX_TO_CATEGORY
)
from aipipe_reprod.shapley.hierarchical.mab import UCB1_Bandit
from aipipe_reprod.shapley.utils.load_diffprep_dataset import load_diffprep_meta

# --- Worker process setup ---
# worker_data = None
# worker_dataset_name = None
# worker_bandits = None  # Worker will hold a reference to the shared bandits

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
                 cache_path: Optional[str] = None, dump_interval: int = 10, other_cache_dir: Optional[str] = None):
        self.dataset_name = dataset_name
        # self.n_processes = n_processes
        self.bandits = bandits
        self.is_category_mode = bandits is not None
        self.cache_path = cache_path
        self.dump_interval = dump_interval
        self.batch_counter = 0
        self.data = data

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
        # init_args = (data, dataset_name, self.bandits)
        # self.pool = ProcessPool(
        #     processes=self.n_processes,
        #     initializer=init_worker,
        #     initargs=init_args,
        #     maxtasksperchild=100
        # )

    def get_pipeline_key(self, pipeline: List[int]) -> str:
        p = list(filter(lambda x: x >= 0, pipeline))
        return f'{self.dataset_name}:{str(p)}'
    
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
        cache_path = os.path.join(cache_dir, f'{self.dataset_name}.json.bak')
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            cached_data = {f'{key}': value for key, value in cached_data.items()}
            self.shared_cache.update(cached_data)
            self.local_cache.update(cached_data)
            logger.info(f"Loaded {len(cached_data)} entries from cache at {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def save_other_cache(self, cache_dir: str):
        cache_path = os.path.join(cache_dir, f'{self.dataset_name}.json.bak')
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
        all_results = list(tqdm(
            self.single_operator_pipe_eval(pipeline) for pipeline in uncached_pipelines
        ))

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

            # 验证必要的数据键存在
            # required_keys = ['train', 'test', 'target', 'target_test']
                
            if d['train'].isna().any().any() or d['test'].isna().any().any():
                im = ImputerCatPrim()
                d['train'], d['test'], d['target'] = im.transform(d['train'], d['test'], d['target'])
                im2 = ImputerMeanPrim()
                d['train'], d['test'], d['target'] = im2.transform(d['train'], d['test'], d['target'])
            if d['train'].select_dtypes(include=['object']).any().any():
                ohe = OneHotEncoderPrim()
                d['train'], d['test'], d['target'] = ohe.transform(d['train'], d['test'], d['target'])
                
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
            self.dump_cache(f'{cache_path}/{self.dataset_name}.json.bak')
            logger.info(f"Cache dumped to {cache_path}/{self.dataset_name}.json.bak")

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
                 is_category_mode: bool = False):
        self.evaluator = evaluator
        self.M = pipeline_length
        self.is_category_mode = is_category_mode
        self.baseline_idx = -1

        if is_category_mode:
            self.players = CATEGORIES
            self.N = len(CATEGORIES)
        else:
            self.players = list(OPERATOR_INFO.keys())
            self.N = len(OPERATOR_INFO)

        logger.info(f"PermShapley initialized. Mode={'Category' if is_category_mode else 'Operator'}. Players={self.N}")

    def calc_shapley_value_full_cate(self, slot_pos: int, target_player_idx: int,
                                previous_sels: List[int]) -> float:
        """计算完整的Shapley值"""
        remaining_slots = self.M - slot_pos - 1

        if remaining_slots < 0:
            logger.warning(f"Invalid remaining slots: {remaining_slots}")
            return 0.0

        pipelines_to_eval = []
        coalitions = list(product(range(self.N), repeat=remaining_slots)) # 3^2=9

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

            return shapley_value / valid_comparisons if valid_comparisons > 0 else 0.0

        except Exception as e:
            logger.error(f"Shapley calculation failed: {e}")
            return 0.0

    def run_full_enumerate_algorithm_cate(self) -> List[int]:
        """运行完整枚举算法"""
        logger.info(
            f"\n{'=' * 20} STARTING FULL ENUMERATION "
            f"({'CATEGORY' if self.is_category_mode else 'OPERATOR'} LEVEL) {'=' * 20}"
        )

        selected_pipeline_indices = []

        _shapley_store = []

        for slot_pos in range(self.M):
            best_player_idx = -1
            best_shapley = float('-inf')

            logger.info(f"Processing slot {slot_pos + 1}/{self.M}")

            for player_idx in range(self.N):
                try:
                    shapley_val = self.calc_shapley_value_full_cate(
                        slot_pos, player_idx, selected_pipeline_indices
                    )
                    _shapley_store.append({
                        'dataset': self.evaluator.dataset_name,
                        'slot_pos': slot_pos,
                        'player_idx': player_idx,
                        'shapley_val': shapley_val,
                    })

                    player_name = (IDX_TO_CATEGORY.get(player_idx, player_idx)
                                   if self.is_category_mode else player_idx)
                    logger.info(
                        f"  Slot {slot_pos + 1}, Player {player_idx} ({player_name}): "
                        f"Shapley = {shapley_val:.4f}"
                    )

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

        os.makedirs('save/shapley_value_pos_vis', exist_ok=True)
        pd.DataFrame(_shapley_store).to_csv('save/shapley_value_pos_vis/stage_1_shapley.tsv', sep='\t', mode='a', header=None, index=False)
        return selected_pipeline_indices


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
                 mode: str, category_pipeline: Optional[List[str]] = None):
        """
        Args:
            evaluator: The parallel evaluator instance.
            pipeline_length: The length of the pipeline to build.
            mode: Either 'category' or 'operator'.
            category_pipeline: A list of category names. Required if mode is 'operator'.
        """
        self.evaluator = evaluator
        self.M = pipeline_length
        self.mode = mode
        self.category_pipeline = category_pipeline
        self.baseline_idx = -1

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
                             previous_sels: List[int]) -> float:
        """
        Calculates the Shapley value for a target player at a specific slot,
        respecting the constraints of the search space.
        """
        remaining_slots = self.M - slot_pos - 1
        if remaining_slots < 0:
            return 0.0

        # Dynamically build the sets of players for the remaining slots
        future_player_sets = [self.player_sets[k] for k in range(slot_pos + 1, self.M)]

        # Generate coalitions (future pipeline segments) using itertools.product
        # This is the core of the constrained permutation logic
        if not future_player_sets:
            coalitions = [[]] # If it's the last slot, there's only one empty coalition
        else:
            coalitions = list(product(*future_player_sets))

        pipelines_to_eval = []
        for coalition in coalitions:
            coalition_list = list(coalition)
            pipelines_to_eval.append(previous_sels + [target_player] + coalition_list)
            pipelines_to_eval.append(previous_sels + [self.baseline_idx] + coalition_list)

        try:
            results = self.evaluator.evaluate_pipelines_op(pipelines_to_eval)
            shapley_value = 0.0
            
            for coalition in coalitions:
                coalition_list = list(coalition)
                target_key = self.evaluator.get_pipeline_key(
                    previous_sels + [target_player] + coalition_list
                )
                baseline_key = self.evaluator.get_pipeline_key(
                    previous_sels + [self.baseline_idx] + coalition_list
                )

                target_perf = results.get(target_key, 0.0)
                baseline_perf = results.get(baseline_key, 0.0)
                
                shapley_value += (target_perf - baseline_perf)

            return shapley_value / len(coalitions) if coalitions else 0.0

        except Exception as e:
            logger.error(f"Shapley calculation failed: {e}", exc_info=True)
            return 0.0

    def run_search(self) -> List[int]:
        """
        Runs the full enumeration algorithm to find the best pipeline.
        """
        logger.info(f"\n{'=' * 20} STARTING CONSTRAINED SHAPLEY SEARCH (Mode: {self.mode.upper()}) {'=' * 20}")

        selected_pipeline = []
        _shapley_store = []
        for slot_pos in range(self.M):
            best_player = -1
            best_shapley = float('-inf')

            # The set of candidate players for the current slot is constrained
            candidate_players = self.player_sets[slot_pos]
            # for storing all the shapley values and draw graph, temporarily make candidate_players the full operators
            # candidate_players = list(range(len(QActionProvider.methods)))
            logger.info(f"Processing slot {slot_pos + 1}/{self.M}. Candidates: {len(candidate_players)}")

            for player in candidate_players:
                shapley_val = self.calc_shapley_value(
                    slot_pos, player, selected_pipeline
                )
                _shapley_store.append({
                    'dataset': self.evaluator.dataset_name,
                    'slot_pos': slot_pos,
                    'player_idx': player,
                    'shapley_val': shapley_val,
                })

                player_name = self._get_player_name(player)
                logger.info(f"  Slot {slot_pos + 1}, Player '{player_name}' ({player}): Shapley = {shapley_val:.4f}")

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

        os.makedirs('save/shapley_value_pos_vis', exist_ok=True)
        pd.DataFrame(_shapley_store).to_csv('save/shapley_value_pos_vis/stage_2_shapley.tsv', sep='\t', mode='a', header=None, index=False)
        return selected_pipeline

# --- 主协调器 HierarchicalShapley  ---
class HierarchicalShapley:
    def __init__(self, dataset_name: str, data: DatasetType, pipeline_length: int, 
                 other_cache_path: Optional[str] = None, enable_pretrain: bool = True, stage2_method='shapley'):
        self.dataset_name = dataset_name
        self.data = data
        self.pipeline_length = pipeline_length
        self.cache_path_prefix = f"aipipe_reprod/shapley/saves/hierarchical_cache/{self.dataset_name}"
        self.other_cache_path = other_cache_path
        self.enable_pretrain = enable_pretrain
        self.stage2_method = stage2_method

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
            self._pretrain_bandits(bandits)
        else:
            logger.info('不启用预训练')

        # --- 阶段一: MAB引导的类别搜索 ---
        logger.info(f"\n{'#' * 60}\n### STAGE 1: MAB-GUIDED SEARCH FOR BEST CATEGORY PIPELINE ###\n{'#' * 60}")
        category_evaluator = ParallelPipelineEvaluator(
            self.dataset_name, self.data,
            bandits=bandits,  # Pass shared bandits
            cache_path=f"{self.cache_path_prefix}_category_cache_mab.json",
            other_cache_dir=self.other_cache_path
        )
        category_shapley = PermShapley(category_evaluator, self.pipeline_length, is_category_mode=True)
        best_category_indices = category_shapley.run_full_enumerate_algorithm_cate()
        # category_searcher = ConstrainedPermShapley(category_evaluator, self.pipeline_length, mode='category')
        # best_category_indices = category_searcher.run_search()
        best_category_pipeline = [IDX_TO_CATEGORY[i] for i in best_category_indices]

        logger.info(f"STAGE 1 RESULT: Best category pipeline found: {[c for c in best_category_pipeline]}")
        category_evaluator.cleanup(self.other_cache_path)
        # category_evaluator.save_other_cache(self.other_cache_path)

        # --- 阶段二: MAB热启动的贪心搜索 ---
        logger.info(f"\n{'#' * 60}\n### STAGE 2: MAB-WARM-STARTED GREEDY SEARCH ###\n{'#' * 60}")
        operator_evaluator = ParallelPipelineEvaluator(
            self.dataset_name, self.data,
            cache_path=f"{self.cache_path_prefix}_operator_cache.json",
            other_cache_dir=self.other_cache_path
        )
        # greedy_searcher = GreedySearcher(operator_evaluator, bandits=bandits)
        # final_pipeline = greedy_searcher.find_best_pipeline(best_category_pipeline)

        if self.stage2_method == 'shapley':
            # Use the generalized searcher in 'operator' mode
            operator_searcher = ConstrainedPermShapley(
                operator_evaluator,
                self.pipeline_length,
                mode='operator',
                category_pipeline=best_category_pipeline
            )
            final_pipeline = operator_searcher.run_search()
        
        elif self.stage2_method == 'greedy':
            # This is your original GreedySearcher logic
            greedy_searcher = GreedySearcher(operator_evaluator)
            final_pipeline = greedy_searcher.find_best_pipeline(best_category_pipeline)
        
        else:
            raise ValueError(f"Unknown stage 2 method: {self.stage2_method}")

        final_performance_dict = operator_evaluator.evaluate_pipelines_op([final_pipeline])
        final_accuracy = list(final_performance_dict.values())[0] if final_performance_dict else 0.0

        logger.info(f"\n{'#' * 60}\n### HIERARCHICAL SEARCH COMPLETED ###\n{'#' * 60}")
        logger.info(f"Final Operator Pipeline: {final_pipeline} {pipe_to_names(final_pipeline)}")
        logger.info(f"Final Accuracy: {final_accuracy:.4f}")
        operator_evaluator.cleanup(self.other_cache_path)
        # manager.shutdown()

        return final_pipeline, final_accuracy
    
    def _pretrain_bandits(self, bandits: Dict[str, UCB1_Bandit]):
        # 为每个类别初始化一个简单的随机策略
        with open(os.path.join(self.other_cache_path, f'{self.dataset_name}.json.bak'), 'r') as f:
            cache = json.load(f)
        
        # 该缓存将-1的算子去掉
        cache2 = {}
        for k, v in cache.items():
            dsn, pipe_str = k.split(':')
            pipe: list[int] = json.loads(pipe_str)
            pipe = list(filter(lambda x: x >= 0, pipe))
            cache2[tuple(pipe)] = v
        
        # 将cache2转换为列表
        cache2_list = list(cache2.items())
        # 取出所有长度为3的pipe和utility
        cache2_3len_list = [item for item in cache2_list if len(item[0]) == 3]
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


def process_dataset(meta, pipeline_length, enable_pretrain, stage2_method='shapley'):
    dataset_name = meta['name']
    log_dir = os.path.join(os.getenv('PROJ_DIR'), 'aipipe_reprod', 'shapley', 'saves', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{dataset_name}_hierarchical.log')

    logger_id = logger.add(log_path, rotation='10 MB', retention='7 days', enqueue=True)
    try:
        logger.info(f"=== 开始处理数据集: {dataset_name} (HierarchicalShapley) ===")
        data = path_to_df_dict(meta['path'], meta['label'])

        hierarchical_search = HierarchicalShapley(
            dataset_name=dataset_name,
            data=data,
            pipeline_length=pipeline_length,
            other_cache_path='aipipe_reprod/shapley/saves/cache',
            enable_pretrain=enable_pretrain,
            stage2_method=stage2_method,
        )
        pipeline, acc = hierarchical_search.run()

        with open(f'aipipe_reprod/shapley/saves/hierarchical_runtime.tsv', 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t{dataset_name}\t{pipeline}\t{acc:.4f}\t{stage2_method}\t{'pretrain' if enable_pretrain else 'no_pretrain'}\n")

        logger.info(f"=== 数据集 {dataset_name} 处理完成 ===")
    except Exception as e:
        logger.error(f"处理数据集 {dataset_name} 时出错: {str(e)}", exc_info=True)
        traceback.print_exc()
    finally:
        logger.remove(logger_id)


def run_hierarchical_search(use = [], enable_pretrain=True, stage2_method='shapley'):
    logger.info("STARTING HIERARCHICAL SHAPLEY SEARCH")
    _metadatas = load_diffprep_meta()
    metadatas = [meta for meta in _metadatas if meta['name'] in use]
    if len(metadatas) == 0:
        raise ValueError("use must be a non-empty list")
    
    # --- 64核机器配置 ---
    pipeline_length = 6

    for meta in metadatas:
        process_dataset(meta, pipeline_length, enable_pretrain, stage2_method)

    logger.info("所有数据集处理完成！")

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use', type=str, required=True, help='input the dataset names, split with comma')
    parser.add_argument('--enable_pretrain', type=int, default=1, help='enable pretrain')
    args = parser.parse_args()
    args.use = list(map(lambda x: x.strip(), args.use.split(',')))
    return args

if __name__ == '__main__':

    # args = parse_args()
    # run_hierarchical_search(use=args.use, 
    #                         max_concurrent_datasets=args.parallel, 
    #                         n_processes_per_dataset=args.nprocessors,
    #                         enable_pretrain=bool(args.enable_pretrain))
    
    # run_hierarchical_search(use=['abalone','ada_prior','avila','connect-4',
    #                              'eeg','google','house_prices','jungle_chess',
    #                              'micro','mozilla4','obesity','page-blocks',
    #                              'pbcseq','pol','run_or_walk','shuttle',
    #                              'uscensus','wall-robot-nav'
    #                              ], 
    #                         enable_pretrain=True)
    run_hierarchical_search(use=['house_prices'], 
                            enable_pretrain=True,
                            stage2_method='shapley')


