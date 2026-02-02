''''
设置下游任务可选，通过传入参数 --downstream lr/rf/gb/mlp 来指定
'''



from collections import Counter
from datetime import datetime
import traceback
from loguru import logger
import numpy as np
import time
from itertools import permutations
import psutil
import multiprocessing as mp
from typing import TypedDict, List
from multiprocessing import Pool, Manager
from tqdm import tqdm
import sys
import os
import json
from dotenv import load_dotenv
from multiprocessing.pool import Pool as ProcessPool

load_dotenv()
sys.path.append(os.getenv('PROJ_DIR'))

from aipipe_reprod.utils.dataloader.load_diffprep import path_to_df_dict, path_and_label_idx_to_dict
from aipipe_reprod.utils.dataloader.load_cc18 import list_cc18, load_cc18
# from aipipe_reprod.shapley.q_action_provider_simple import QActionProviderSimple
# from aipipe_reprod.shapley.q_action_provider import QActionProvider
from aipipe_reprod.actions.q_action_provider import QActionProvider
from operators.types import DatasetType
from sklearn.metrics import accuracy_score


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


OP_CNT_DICT={'op6':  [0, 1, 5, 9, 15, 17],
             'op10': [0, 1, 2, 5, 6, 9, 11, 15, 17, 24],
             'op15': [0, 1, 2, 3, 5, 6, 7, 9, 11, 12, 13, 15, 17, 23, 24],
             'op20': [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 21, 22, 23, 24],
             'op25': list(range(len(QActionProvider.methods)))}

OP_USE = 'op25'
DT = 'lr'
EARLY_STOP = True

dt_dict = {
    'lr': lambda: QActionProvider.lr(),
    'rf': lambda: QActionProvider.rf(),
    'gb': lambda: QActionProvider.xg(),
    'mlp': lambda: QActionProvider.mlp(),
}


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

            # 只做了填充和编码的时候，不计算额外的那么多cache
            # if op <= QActionProvider.spans[2][-1]: 
            #     continue
            key = f'{worker_dataset_name}:{cumu_pipe}'
            if key in worker_cache:
                accuracy = float(worker_cache[key])
                continue
            lr = dt_dict[DT]() # QActionProvider.lr()
            y_pred = lr.transform_1(d['train'], d['target'], d['test'])
            accuracy = float(accuracy_score(d['target_test'], y_pred))
            caches[key] = accuracy
        if len(pipeline) == 0:
            key = f'{worker_dataset_name}:[]'
            if key in worker_cache:
                accuracy = float(worker_cache[key])
                return accuracy, {}
            lr = dt_dict[DT]() # QActionProvider.lr()
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


class ParallelPipelineEvaluator:
    """完整枚举并行评估器 (进程池优化版)"""

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

        # 共享缓存 (Manager is necessary for inter-process sharing)
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

        # 使用进程池，并通过 initializer 传递大数据
        # maxtasksperchild=1 helps with memory leaks in long-running tasks
        self.pool = Pool(
            processes=self.n_processes,
            initializer=init_worker,
            initargs=(data, dataset_name, self.shared_cache),
            maxtasksperchild=100  # Periodically restart workers to free memory
        )

    def load_cache(self):
        """从文件加载缓存数据"""
        try:
            with open(self.cache_path, 'r') as f:
                cached_data = json.load(f)
            with self.cache_lock:
                self.shared_cache.update(cached_data)
            logger.info(f"Loaded {len(cached_data)} entries from cache file")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def dump_cache(self):
        """保存缓存到文件"""
        if not self.cache_path: return
        try:
            with self.cache_lock:
                cache_data = dict(self.shared_cache.copy())
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'w') as f:
                json.dump(cache_data, f)
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
        """批量评估pipelines - 进程池优化版"""
        results = {}
        uncached_pipelines = []
        hits = 0
        misses = 0

        # 1. 检查缓存 (same as before)
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
            logger.info(f"      所有 {len(pipelines)} 个pipeline均命中缓存")
            return results

        logger.info(f"      评估 {len(uncached_pipelines):,} 个未缓存pipeline...")

        # 3. 使用进程池执行任务
        # pool.imap_unordered is often faster as it yields results as they complete
        # We wrap it in tqdm for a progress bar.
        batch_results_iterator = self.pool.imap_unordered(evaluate_pipeline_worker, uncached_pipelines)

        batch_results = list(tqdm(batch_results_iterator,
                                  total=len(uncached_pipelines),
                                  desc="      评估进度",
                                  leave=False))

        # 4. 更新缓存和结果
        with self.cache_lock:
            for pipeline, (performance, cache) in zip(uncached_pipelines, batch_results):
                pipeline_key = f'{self.dataset_name}:{str(pipeline)}'
                # self.shared_cache[pipeline_key] = performance
                self.shared_cache.update(cache)
                results[pipeline_key] = performance

        # 5. 按间隔保存缓存 (same as before, but outside the lock)
        self.batch_counter += 1
        if self.batch_counter % self.dump_interval == 0:
            self.dump_cache()
            # self.batch_counter is not shared, this is fine

        # 6. 更新评估统计
        self.stats['total_evaluations'] += len(uncached_pipelines)
        logger.info(f"      完成 {len(uncached_pipelines)} 个pipeline评估，命中率: {hits / (hits + misses):.2%}")

        return results

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'pool'):
            self.pool.close()  # 禁止向进程池提交新任务
            self.pool.join()  # 等待所有工作进程退出
        if hasattr(self, 'manager'):
            self.manager.shutdown()

    def dump_final(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            cache = dict(self.shared_cache.copy())
            json.dump(cache, f)


class PermShapley:
    def __init__(self,
                 dataset_name: str,
                 dataset: DatasetType,
                 slot_num=6,
                 n_processes=None,
                 force_large_computation=False,
                 cache_manager='dict',
                 operations=list(range(len(QActionProvider.methods))),
                 cache_path: str = None,
                 dump_interval: int = 10,
                 output_shapley_values_path=None):  # 新增缓存路径参数
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.operations = operations
        self.baseline_idx = -1

        self.shapley_values = {}
        self.N = len(operations)
        self.M = slot_num-3

        self.n_processes = n_processes or min(mp.cpu_count(), 16)
        self.force_large_computation = force_large_computation
        self.evaluator = ParallelPipelineEvaluator(
            self.dataset_name,
            self.dataset,
            self.operations,
            self.n_processes,
            cache_path=cache_path,  # 传递缓存路径
            dump_interval=dump_interval  # 传递缓存保存间隔
        )

        self.output_shapley_values_path = output_shapley_values_path
        self.shapley_cache = []

        # 统计信息
        self.stats: StatsDict = {
            'total_evaluations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'slot_times': [],
            'theoretical_evaluations': 0
        }

    def generate_constrainted_coalitions_cell(self, previous_sels: list[int], slot_pos: int, target_op: int, 
                                              has_cate_miss, has_cate_enc, has_num_miss, num_dim, cate_dim, prep_start):
        global OP_USE
        """生成约束条件下的所有 coalition"""
        cat_miss_span = [-1]
        num_miss_span = [-1]
        cat_enc_span = [-1]
        if has_cate_miss:
            cat_miss_span = list(set(QActionProvider.spans[0]) & set(OP_CNT_DICT[OP_USE]))
        if has_cate_enc:
            cat_enc_span = list(set(QActionProvider.spans[2]) & set(OP_CNT_DICT[OP_USE]))
        if has_num_miss:
            num_miss_span = list(set(QActionProvider.spans[1]) & set(OP_CNT_DICT[OP_USE]))

        feature_spans = list(set([*QActionProvider.spans[3], 
                         *QActionProvider.spans[4], 
                         *QActionProvider.spans[5]]) & set(OP_CNT_DICT[OP_USE]))
        
        prev_sel_positions = [i for i, x in enumerate(previous_sels) if x != -1]
        for cate_miss_op in cat_miss_span:
            for num_miss_op in num_miss_span:
                for cat_enc_op in cat_enc_span:
                    base_pipe = list(filter(lambda x: x != -1, [cate_miss_op, cat_enc_op, num_miss_op]))
                    occupied_positions = set(list(range(len(base_pipe))) + [slot_pos] + prev_sel_positions)  # 这些位置已经被选中了，不能再用于选特征算子
                    remaining_slots = self.M + prep_start - len(occupied_positions)  # 剩下的位置可以用来填充特征算子
                    # 如果 prevsel 是在靠后的位置，导致slot_pos前面有空隙，则这个空隙按照约束是必须被填上的。
                    # |B      B      _      S     P      _ |     其中 empty 是需要枚举联盟的。下面循环中的 r 的范围是 [1,2] 而不是 [0, 1, 2]
                    #  ^base1 ^base2 ^empty ^slot ^prev  ^empty
                    prev_occupied_pos_cnt_before_slot = len([pos for pos in prev_sel_positions if pos < slot_pos])
                    if slot_pos >= len(base_pipe) + prev_occupied_pos_cnt_before_slot:
                        start = slot_pos - len(base_pipe) - prev_occupied_pos_cnt_before_slot
                    else:
                        start = 0
                    for r in range(start, remaining_slots + 1):
                        if r < 0:
                            logger.error(f"r={r} is negative, which is not expected.")
                            continue
                        elif r == 0: 
                            final_pipe = previous_sels.copy()
                            for pos, b in enumerate(base_pipe):
                                final_pipe[pos] = b
                            yield final_pipe
                        else:
                            # 改成不可重复的选择
                            f_spans = list(set(feature_spans) - set(previous_sels) - set([target_op]))
                            for combo in permutations(f_spans, r=r):
                                # 返回最终的pipeline
                                final_pipe = [-1 for _ in range(len(previous_sels))]
                                # 先填充 base_pipe（填充编码等必做项）
                                for pos, op in enumerate(base_pipe): final_pipe[pos] = op
                                # 填充已经做过的算子
                                for pos, prev in enumerate(previous_sels):
                                    if prev != -1: final_pipe[pos] = prev
                                # 填充新选的算子
                                empty_positions = [i for i in range(len(previous_sels)) if final_pipe[i] == -1 and i != slot_pos]
                                for pos, op in zip(empty_positions, combo):
                                    final_pipe[pos] = op
                                
                                final_target_pipe = final_pipe.copy()
                                final_target_pipe[slot_pos] = target_op
                                # 特征嵌入的次数过多则不予考虑
                                counter = Counter(final_target_pipe)
                                if counter.get(23, 0) >= 2: continue  # RandomTreesEmbedding
                                if cate_dim + num_dim >= 14 and counter.get(23, 0) + counter.get(15, 0) + counter.get(16, 0) >= 2: continue  # 特征嵌入过多
                                if (counter.get(17, 0) + counter.get(18, 0) + counter.get(19, 0) 
                                    + counter.get(20, 0) + counter.get(21, 0) + counter.get(22, 0) >= 3): continue  # 特征降维过多
                                if counter.get(24, 0) >= 3: continue  # 特征选择过多

                                # 连着两次特征选择/相同的缩放，得到的pipeline accuracy结果应该会是相同的
                                flag = False
                                for i in range(len(final_target_pipe)-1):
                                    if final_target_pipe[i] == final_target_pipe[i+1] and (final_target_pipe[i] in {24, 7, 8, 9, 10, 13}): 
                                        flag = True
                                        break
                                    if final_target_pipe[i] in {7, 8, 9, 10} and final_target_pipe[i+1] in {7, 8, 9, 10}:  # 连续的 scaler，只有最后一个生效
                                        flag = True
                                        break
                                    if final_target_pipe[i] in {23} and final_target_pipe[i+1] in {23, 15, 16}:  # 特征嵌入过多
                                        flag = True
                                        break
                                    if final_target_pipe[i] in {17, 18, 19, 20, 21, 22} and final_target_pipe[i+1] in {17, 18, 19, 20, 21, 22}:
                                        flag = True
                                        break
                                if flag: continue
                                
                                yield final_pipe


    def calc_cell_shap(self, slot_pos: int, target_op: int, previous_sels: list[int], has_cate_miss, has_cate_enc, has_num_miss, num_dim, cate_dim, prep_start):
        """计算网格中一个单元格的Shapley值"""
        # 当数据集本身有空值，且还没做缺失值处理时，has_cate_miss 为 True
        has_cate_miss = has_cate_miss and len(set(previous_sels) & set(QActionProvider.spans[0])) == 0 and target_op not in QActionProvider.spans[0]
        # 当数据集本身有字符串值，且还没做字符串编码时，has_cate_enc 为 True
        has_cate_enc = has_cate_enc and len(set(previous_sels) & set(QActionProvider.spans[2])) == 0 and target_op not in QActionProvider.spans[2]
        # 当数据集本身有空值，且还没做缺失值处理时，has_num_miss 为 True
        has_num_miss = has_num_miss and len(set(previous_sels) & set(QActionProvider.spans[1])) == 0 and target_op not in QActionProvider.spans[1]

        coalitions = list(self.generate_constrainted_coalitions_cell(
            previous_sels, slot_pos, target_op, has_cate_miss, has_cate_enc, has_num_miss, num_dim, cate_dim, prep_start))
        coalition_cnt = len(coalitions)
        pipelines_to_evaluate = []
        for coalition in coalitions:
            pipeline_base = tuple(filter(lambda x: x != -1, coalition.copy()))
            if pipeline_base not in pipelines_to_evaluate:
                pipelines_to_evaluate.append(pipeline_base)

            pipeline_target = coalition.copy()
            pipeline_target[slot_pos] = target_op
            pipeline_target = tuple(filter(lambda x: x != -1, pipeline_target))
            if pipeline_target not in pipelines_to_evaluate:
                pipelines_to_evaluate.append(pipeline_target)

        if coalition_cnt == 0:
            logger.info(f"        No pipelines to evaluate for slot {slot_pos} with target {target_op}")
            eval_time = 0.
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


    def run_full_enumerate_algorithm(self):
        """运行完整枚举算法"""
        logger.info(f"\n{'=' * 60}")
        logger.info("STARTING FULL ENUMERATION ALGORITHM")
        logger.info("WARNING: This will evaluate ALL possible coalitions")
        global OP_USE, EARLY_STOP, DT
        logger.info(f"OP_USE: {OP_USE}, early stop: {EARLY_STOP}, downstream {DT}")
        logger.info(f"{'=' * 60}")

        algorithm_start = time.time()
        selected_pipeline: list[int] = [-1 for _ in range(self.M)]

        string_columns = self.dataset['train'].select_dtypes(exclude=['number']).columns
        has_cate_enc = len(string_columns) > 0
        has_cate_miss = self.dataset['train'][string_columns].isnull().any().any()
        has_num_miss = self.dataset['train'].select_dtypes(include=['number']).isnull().any().any()

        feature_spans = list(set([
            *QActionProvider.spans[3],
            *QActionProvider.spans[4],
            *QActionProvider.spans[5],
        ]) & set(OP_CNT_DICT[OP_USE]))

        # edge case: 因为字符串填充方法只有一个，所以对于有字符串空值的可以直接用这个方法，不需要考虑其他字符串填充方法（只有一个选项，直接选这个不影响枚举结果）
        start = 0
        if has_cate_miss:
            selected_pipeline.append(-1)
            selected_pipeline[start] = QActionProvider.spans[0][0]
            start += 1

        if has_cate_enc:
            shaps: dict[int, float] = {}
            selected_pipeline.append(-1)
            for op_idx in list(set(QActionProvider.spans[2]) & set(OP_CNT_DICT[OP_USE])):
                shaps[op_idx] = self.calc_cell_shap(start, op_idx, selected_pipeline, has_cate_miss, has_cate_enc, has_num_miss, 
                                                    len(feature_spans), len(string_columns), start)
            selected_op, shap = max(shaps.items(), key=lambda x: x[1])
            selected_pipeline[start] = selected_op
            start += 1

        if has_num_miss:
            shaps: dict[int, float] = {}
            selected_pipeline.append(-1)
            for op_idx in list(set(QActionProvider.spans[1]) & set(OP_CNT_DICT[OP_USE])):
                shaps[op_idx] = self.calc_cell_shap(start, op_idx, selected_pipeline, has_cate_miss, has_cate_enc, has_num_miss, 
                                                    len(feature_spans), len(string_columns), start)
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
                for op_idx in list(set(range(self.N)) & set(feature_spans)):
                    # 跳过已经选过的操作
                    if op_idx in selected_pipeline:
                        continue
                    if (
                        (not has_cate_enc and op_idx in QActionProvider.spans[2])   # 没有字符串列
                        or (op_idx in QActionProvider.spans[2] and len(set(selected_pipeline) & set(QActionProvider.spans[2])) > 0) # 已经做过字符串编码
                    ):
                        continue
                    if (
                        (not has_num_miss and op_idx in QActionProvider.spans[1])    # 用不着做数字的缺失值处理
                        or (op_idx in QActionProvider.spans[1] and len(set(selected_pipeline) & set(QActionProvider.spans[1])) > 0) # 已经做过数字缺失值处理
                    ):
                        continue
                    if (
                        (not has_cate_miss and op_idx in QActionProvider.spans[0])   # 用不着做字符串填充操作
                        or (op_idx in QActionProvider.spans[0] and len(set(selected_pipeline) & set(QActionProvider.spans[0])) > 0) # 已经做过字符串填充操作
                    ):
                        continue

                    # TODO 还有其他要排除：已选的里面有特征工程，紧跟着的这个op_idx也是特征工程，需要屏蔽掉
                    selected_positions = [i for i, x in enumerate(selected_pipeline) if x != -1]
                    fl = False
                    for sel_pos in selected_positions:
                        if slot_pos - sel_pos == 1 and selected_pipeline[sel_pos] == 23 and op_idx in {23,15,16}: fl=True; break
                        if abs(slot_pos - sel_pos) == 1 and selected_pipeline[sel_pos] in {7,8,9,10} and op_idx in {7,8,9,10}: fl=True; break
                        if abs(slot_pos - sel_pos) == 1 and selected_pipeline[sel_pos] in {17,18,19,20,21,22} and op_idx in {17,18,19,20,21,22}: fl=True; break
                        if abs(slot_pos - sel_pos) == 1 and selected_pipeline[sel_pos] == op_idx and op_idx in {24,7,8,9,10,13}: fl=True; break

                    if fl: continue

                    shapley_val = self.calc_cell_shap(slot_pos, op_idx, selected_pipeline, has_cate_miss, has_cate_enc, has_num_miss, 
                        self.dataset['train'].shape[1]-len(string_columns), len(string_columns), start)
                    
                    shap_grid[op_idx][slot_pos] = shapley_val
            
            # 同时选择最优的slot和action
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
            if EARLY_STOP and best_shapley <= 0:
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
        logger.info(f"Evaluations: {self.evaluator.stats['total_evaluations']:,}")
        logger.info(f"Theoretical: {self.stats['theoretical_evaluations']:,}")
        # 修正：使用评估器的实际评估次数
        # cache_efficiency = (self.stats['theoretical_evaluations'] - self.evaluator.stats['total_evaluations']) / \
        #                    self.stats['theoretical_evaluations'] * 100
        # logger.info(f"Cache efficiency: {cache_efficiency:.1f}%")

        # 新增：保存Shapley值缓存
        if self.output_shapley_values_path is not None:
            import pandas as pd
            pd.DataFrame(self.shapley_cache).to_csv(self.output_shapley_values_path.replace('.json', '.csv'), index=False)

        return selected_pipeline, final_accuracy


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
    """运行完整枚举演示 (并发版本 - 优化后)"""
    logger.info("FULL ENUMERATION SHAPLEY VALUE DEMONSTRATION (OPTIMIZED CONCURRENT VERSION)")
    if len(use) == 0:
        raise Exception('No input datasets')

    metadatas = []

    for meta in _metadatas:
        if meta['name'] in use:
            metadatas.append(meta)
    
    pipeline_length = 6
    num_datasets = len(metadatas)
    logger.info(f"共发现 {num_datasets} 个数据集，准备并发处理...")

    # 使用进程池来管理数据集处理任务，更简洁高效
    max_concurrent_datasets = min(mp.cpu_count() - 1, num_datasets,
                                  parallel)  # Limit to 4 to not overload system with many sub-sub-processes
    logger.info(f"使用 {max_concurrent_datasets} 个并发进程处理数据集")

    # Prepare arguments for each dataset task
    tasks = [(meta, pipeline_length, n_processes) for meta in metadatas]

    with NonDaemonicPool(processes=max_concurrent_datasets) as pool:
        pool.starmap(process_dataset, tasks)

    logger.info("所有数据集处理完成！")


def process_dataset(meta, pipeline_length, n_processes=4):
    """处理单个数据集的函数，包含独立日志配置"""
    dataset_name = meta['name']
    log_dir = os.path.join('aipipe_reprod/shapley/saves/logs', f'choosekfrom25_{DT}')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{dataset_name}.log')
    logger_id = logger.add(log_path, rotation='10 MB', retention='7 days', enqueue=True)

    shapley = None  # 初始化为 None，以便 finally 块中可以使用
    try:
        logger.info(f"=== 开始处理数据集: {dataset_name} ===")
        if 'label' in meta:
            dataset_path = meta['path']
            label = meta['label']
            data = path_to_df_dict(dataset_path, label)
        elif 'label_idx' in meta:
            dataset_path = meta['path']
            label_idx = meta['label_idx']
            data = path_and_label_idx_to_dict(dataset_path, label_idx)

        time_start = time.time()
        shapley = PermShapley(
            dataset_name,
            data,
            pipeline_length,
            cache_path=f"aipipe_reprod/shapley/saves/cache_25-6/{DT}/{dataset_name}.json",
            dump_interval=20,
            force_large_computation=True,
            n_processes=n_processes,
            output_shapley_values_path=f'save/shapley_value_pos_vis/{dataset_name}_sv_rand_{pipeline_length}.json'
        )

        # 从返回值中同时捕获 pipeline 和 acc
        pipeline, acc = shapley.run_full_enumerate_algorithm()
        time_spent = time.time() - time_start

        with open(f'aipipe_reprod/shapley/saves/full_enum_gridsv25_forecon_{DT}.tsv', 'a') as f:
            f.write(f'{dataset_name}\t{pipeline}\t{acc}\t{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\tsv\t{time_spent}\t{shapley.evaluator.stats["total_evaluations"]}'
                    f'\t{OP_USE}\t{"early_stop" if EARLY_STOP else "full"}\t{DT}\t{pipeline_length}\n')
            f.flush()

        # 使用 dump_final 保存任务结束后的完整缓存
        shapley.evaluator.dump_final(f'aipipe_reprod/shapley/saves/cache_25-6/{DT}/{dataset_name}.json')  # <--- 修正后的调用

        logger.info(f"=== 数据集 {dataset_name} 处理完成 ===")
    except Exception as e:
        traceback.print_exc()
        logger.error(f"处理数据集 {dataset_name} 时出错: {str(e)}", exc_info=True)
    finally:
        if shapley:  # 确保 cleanup 被调用
            shapley.evaluator.cleanup()
        logger.remove(logger_id)

def process_dataset_cc18(dataset_name, pipeline_length, n_processes=4):
    """处理单个数据集的函数，包含独立日志配置"""
    log_dir = os.path.join('aipipe_reprod/shapley/saves/logs', f'choosekfrom25_{DT}')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{dataset_name}.log')
    logger_id = logger.add(log_path, rotation='10 MB', retention='7 days', enqueue=True)

    shapley = None  # 初始化为 None，以便 finally 块中可以使用
    try:
        logger.info(f"=== 开始处理数据集: {dataset_name} ===")
        data = load_cc18(dataset_name)
        time_start = time.time()

        shapley = PermShapley(
            dataset_name,
            data,
            pipeline_length,
            cache_path=f"aipipe_reprod/shapley/saves/cache_25-6/{DT}/{dataset_name}.json",
            dump_interval=20,
            force_large_computation=True,
            n_processes=n_processes,
            output_shapley_values_path=f'save/shapley_value_pos_vis/{dataset_name}_sv_rand_{pipeline_length}.json'
        )

        # 从返回值中同时捕获 pipeline 和 acc
        pipeline, acc = shapley.run_full_enumerate_algorithm()
        time_spent = time.time() - time_start

        with open(f'aipipe_reprod/shapley/saves/full_enum_gridsv25_forecon_{DT}.tsv', 'a') as f:
            f.write(f'{dataset_name}\t{pipeline}\t{acc}\t{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\tsv\t{time_spent}\t{shapley.evaluator.stats["total_evaluations"]}'
                    f'\t{OP_USE}\t{"early_stop" if EARLY_STOP else "full"}\t{DT}\t{pipeline_length}\n')
            f.flush()

        # 使用 dump_final 保存任务结束后的完整缓存
        shapley.evaluator.dump_final(f'aipipe_reprod/shapley/saves/cache_25-6/{DT}/{dataset_name}.json')  # <--- 修正后的调用

        logger.info(f"=== 数据集 {dataset_name} 处理完成 ===")
    except Exception as e:
        traceback.print_exc()
        logger.error(f"处理数据集 {dataset_name} 时出错: {str(e)}", exc_info=True)
    finally:
        if shapley:  # 确保 cleanup 被调用
            shapley.evaluator.cleanup()
        logger.remove(logger_id)


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=str, default='diffprep', help='input the dataset set, diffprep or cc18')
    parser.add_argument('--use', type=str, required=True, help='input the dataset names, split with comma')
    parser.add_argument('--len', type=int, required=True, help='input the pipeline length')
    parser.add_argument('--dt', type=str, default='lr', choices=['lr', 'rf', 'gb', 'mlp'], help='input the downstream task')
    parser.add_argument('--op', type=int, default=25, choices=[6, 10, 15, 20, 25], help='input the operator count')
    parser.add_argument('--n_processes', type=int, default=4, help='input the n_processes number')
    parser.add_argument('--early_stop', required=True, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # _metadatas = get_deepline_metadatas()
    args = parse_args()
    OP_USE = f'op{args.op}'
    DT = args.dt
    EARLY_STOP = args.early_stop
    if args.set == 'diffprep':
        _metadatas = load_diffprep_meta()
        m = None
        for meta in _metadatas:
            if meta['name'] == args.use:
                m = meta
                break
        if m is None:
            logger.error(f"未找到数据集 {args.use}")
            exit(1)
        process_dataset(m, pipeline_length=args.len, n_processes=args.n_processes)

    elif args.set == 'cc18':
        for dataset_name in args.use.split(','):
            process_dataset_cc18(dataset_name, pipeline_length=args.len, n_processes=args.n_processes)

    else:
        logger.error(f"未找到数据集 {args.set} > {args.use}")
        exit(1)
    

