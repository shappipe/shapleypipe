''''
product 25
'''

from collections import Counter
from datetime import datetime
import traceback
from loguru import logger
import numpy as np
import time
from itertools import product
import psutil
import multiprocessing as mp
from typing import TypedDict, List
from multiprocessing import Pool, Manager
from tqdm import tqdm
import sys
import os
import json
from dotenv import load_dotenv
import multiprocessing as mp
from multiprocessing.pool import Pool as ProcessPool

load_dotenv()
sys.path.append(os.getenv('PROJ_DIR'))

from aipipe_reprod.utils.dataloader.load_diffprep import path_to_df_dict, path_and_label_idx_to_dict
# from aipipe_reprod.shapley.q_action_provider_simple import QActionProviderSimple
# from aipipe_reprod.shapley.q_action_provider import QActionProvider
from aipipe_reprod.actions.q_action_provider import QActionProvider
from operators.types import DatasetType
from sklearn.metrics import accuracy_score

np.random.seed(42)


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


from aipipe_reprod.primitives import ImputerCatPrim, ImputerMeanPrim, LabelEncoderPrim

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
    global worker_data, worker_dataset_name, worker_cache
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

            if op <= QActionProvider.spans[2][-1]: 
                continue
            key = f'{worker_dataset_name}:{cumu_pipe}'
            if key in worker_cache:
                accuracy = float(worker_cache[key])
                continue
            lr = QActionProvider.lr()
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

        # self._analyze_full_complexity()

    def _analyze_full_complexity(self):
        """分析完整枚举的复杂度"""
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

        # 时间估算
        est_time_per_eval = 0.05  # 50ms per evaluation
        sequential_time = total_evaluations * est_time_per_eval
        parallel_time = sequential_time / self.n_processes * 1.3  # overhead

        logger.info(f"\nTIME ESTIMATES:")
        logger.info(
            f"Sequential: {sequential_time:,.1f}s = {sequential_time / 3600:.1f} hours = {sequential_time / 86400:.1f} days")
        logger.info(
            f"Parallel ({self.n_processes} cores): {parallel_time:,.1f}s = {parallel_time / 3600:.1f} hours = {parallel_time / 86400:.1f} days")

        # 警告系统
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

    def generate_all_coalitions(self, remaining_slots):
        for i in range(remaining_slots + 1):
            if i == 0: yield []
            for combo in product(range(self.N), repeat=i):
                yield list(combo)

    def generate_constrainted_coalitions(self, remaining_slots, 
                                         has_cat_miss: bool, has_cate: bool, has_num_miss: bool,
                                         num_dim: int, cate_dim: int):
        cat_miss_span = [-1]
        num_miss_span = [-1]
        cat_enc_span = [-1]
        if has_cat_miss:
            cat_miss_span = QActionProvider.spans[0].copy()
            remaining_slots -= 1
        if has_cate:
            cat_enc_span = QActionProvider.spans[2].copy()
            remaining_slots -= 1
        if has_num_miss:
            num_miss_span = QActionProvider.spans[1].copy()
            remaining_slots -= 1

        feature_spans = [*QActionProvider.spans[3], 
                         *QActionProvider.spans[4], 
                         *QActionProvider.spans[5]]
        
        cnt = 0
        max_cnt = 200

        try:
            for cate_miss_op in cat_miss_span:
                for num_miss_op in num_miss_span:
                    for cat_enc_op in cat_enc_span:
                        base_pipe = [int(np.random.choice(cat_miss_span)), int(np.random.choice(num_miss_span)), int(np.random.choice(cat_enc_span))]
                        if remaining_slots <= 0:
                            yield []
                            cnt += 1
                        else:
                            # for r in range(remaining_slots + 1):
                                # if r == 0:
                                #     yield list(filter(lambda x: x != -1, base_pipe))
                                # for combo in product(feature_spans, repeat=r):
                            while True:
                                combo = list(np.random.choice(feature_spans, size=remaining_slots))
                                combo = [int(x) for x in combo]
                                counter = Counter(combo)
                                if counter.get(23, 0) >= 2: continue  # RandomTreesEmbedding
                                if cate_dim + num_dim >= 14 and counter.get(23, 0) + counter.get(15, 0) + counter.get(16, 0) >= 2: continue  # 特征嵌入过多

                                # 连着两次特征选择/相同的缩放，得到的pipeline accuracy结果应该会是相同的
                                flag = False
                                for i in range(len(combo)-1):
                                    if combo[i] == combo[i+1] and (combo[i] in {24, 7, 8, 9, 10, 13}): 
                                        flag = True
                                        break
                                    if combo[i] in {7, 8, 9, 10} and combo[i+1] in {7, 8, 9, 10}:  # 连续的 scaler，只有最后一个生效
                                        flag = True
                                        break
                                    if combo[i] in {23} and combo[i+1] in {23, 15, 16}:  # 特征嵌入过多
                                        flag = True
                                        break
                                    if combo[i] in {17, 18, 19, 20, 21, 22} and combo[i+1] in {17, 18, 19, 20, 21, 22}:
                                        flag = True
                                        break
                                        
                                if flag: continue
                                # 返回最终的pipeline
                                yield list(filter(lambda x: x != -1, base_pipe + combo))
                                cnt += 1
                                if cnt >= max_cnt:
                                    raise StopIteration
        except StopIteration:
            logger.info(f"Generated {cnt} pipelines, iteration done")


    def calc_shapley_value_full(self, slot_pos: int, target_op: int, previous_sels: list[int],
                                has_cate_miss, has_cate_enc, has_num_miss, num_dim, cate_dim):
        logger.info(f"    Operation {target_op} ({QActionProvider.get(target_op).get_name()}) at slot {slot_pos + 1}:")

        remaining_slots = self.M - slot_pos - 1
        # 当数据集本身有空值，且还没做缺失值处理时，has_cate_miss 为 True
        has_cate_miss = has_cate_miss and len(set(previous_sels) & set(QActionProvider.spans[0])) == 0 and target_op not in QActionProvider.spans[0]
        # 当数据集本身有字符串值，且还没做字符串编码时，has_cate_enc 为 True
        has_cate_enc = has_cate_enc and len(set(previous_sels) & set(QActionProvider.spans[2])) == 0 and target_op not in QActionProvider.spans[2]
        # 当数据集本身有空值，且还没做缺失值处理时，has_num_miss 为 True
        has_num_miss = has_num_miss and len(set(previous_sels) & set(QActionProvider.spans[1])) == 0 and target_op not in QActionProvider.spans[1]
        
        coalitions = list(self.generate_constrainted_coalitions(
            remaining_slots, has_cate_miss, has_cate_enc, has_num_miss, num_dim, cate_dim))
        
        logger.info(f"        Processing {self.N ** remaining_slots:,} coalitions")

        # 构建所有pipeline
        pipelines_to_evaluate = []
        coalition_cnt = 0
        for coalition in coalitions:
            coalition_cnt += 1
            # Target pipeline
            pipeline_target = tuple(previous_sels + [target_op] + coalition)
            if pipeline_target not in pipelines_to_evaluate:
                pipelines_to_evaluate.append(pipeline_target)

            # Baseline pipeline
            # pipeline_baseline = previous_sels + [self.baseline_idx] + coalition
            pipeline_baseline = tuple(previous_sels + coalition)    # baseline 这个相当于是直接跳过这个操作，为了减少计算，可以将baseline为-1的位置删掉
            if pipeline_baseline not in pipelines_to_evaluate:
                pipelines_to_evaluate.append(pipeline_baseline)

        pipelines_to_evaluate = list(map(lambda x: list(x), pipelines_to_evaluate))
        
        logger.info(f"        Total pipelines to evaluate: {len(pipelines_to_evaluate):,}")
        # 并行评估
        start_time = time.time()
        results = self.evaluator.evaluate_pipelines_batch(pipelines_to_evaluate)
        eval_time = time.time() - start_time

        # 删除：原统计逻辑
        # self.stats['total_evaluations'] += len(pipelines_to_evaluate)

        # 计算Shapley值
        shapley_value = 0.0
        weight = 1.0 / coalition_cnt

        for coalition in coalitions:

            target_key = f'{self.dataset_name}:{previous_sels + [target_op] + coalition}'
            # baseline_key = f'{self.dataset_name}:{previous_sels + [self.baseline_idx] + coalition}'
            baseline_key = f'{self.dataset_name}:{previous_sels + coalition}'

            marginal = results[target_key] - results[baseline_key]
            shapley_value += weight * marginal

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
        logger.info(f"{'=' * 60}")

        algorithm_start = time.time()
        selected_pipeline: list[int] = []

        string_columns = self.dataset['train'].select_dtypes(exclude=['number']).columns
        has_cate_enc = len(string_columns) > 0
        has_cate_miss = self.dataset['train'][string_columns].isnull().any().any()
        has_num_miss = self.dataset['train'].select_dtypes(include=['number']).isnull().any().any()

        feature_spans = [
            *QActionProvider.spans[3],
            *QActionProvider.spans[4],
            *QActionProvider.spans[5],
        ]

        # edge case: 因为字符串填充方法只有一个，所以对于有字符串空值的可以直接用这个方法，不需要考虑其他字符串填充方法
        start = 0
        if has_cate_miss:
            start += 1
            selected_pipeline.append(QActionProvider.spans[0][0])

        for slot_pos in range(start, self.M):
            logger.info(f"\n=== SLOT {slot_pos + 1}/{self.M} ===")
            logger.info(f"Previous: {selected_pipeline}")

            slot_start = time.time()
            best_op = None
            best_shapley = float('-inf')

            # 为每个操作计算Shapley值
            for op_idx in range(QActionProvider.spans[3][0], self.N):  # 跳过baseline
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
                # 对于需要填充和编码的也要做一些“预先排除” （有缺失值、且还没处理缺失值、且当前操作不是缺失值处理操作，则不允许做这个操作）
                if has_cate_miss and len(set(selected_pipeline) & set(QActionProvider.spans[0])) == 0 and op_idx in feature_spans:
                    continue
                if has_num_miss and len(set(selected_pipeline) & set(QActionProvider.spans[1])) == 0 and op_idx in feature_spans:
                    continue
                if has_cate_enc and len(set(selected_pipeline) & set(QActionProvider.spans[2])) == 0 and op_idx in feature_spans:
                    continue

                shapley_val = self.calc_shapley_value_full(
                    slot_pos, op_idx, selected_pipeline,
                    has_cate_miss, has_cate_enc, has_num_miss, 
                    self.dataset['train'].shape[1]-len(string_columns), len(string_columns)
                )

                if shapley_val > best_shapley:
                    best_shapley = shapley_val
                    best_op = op_idx

            selected_pipeline.append(best_op)
            slot_time = time.time() - slot_start
            self.stats['slot_times'].append(slot_time)

            logger.info(f"Selected: {best_op}, Shapley: {best_shapley:.6f}")
            logger.info(f"Slot time: {slot_time / 60:.1f} minutes")

            # 新增：slot完成后保存缓存
            self.evaluator.dump_cache()

            # 内存监控
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Memory: {memory_mb:.1f}MB")

        total_time = time.time() - algorithm_start

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
            with open(self.output_shapley_values_path, 'w') as f:
                json.dump(self.shapley_cache, f, indent=2)

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
    
    pipeline_length = 5
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
    log_dir = os.path.join('aipipe_reprod/shapley/saves/logs', 'choose6from18')
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

        im = ImputerCatPrim()
        data['train'], data['test'], data['target'] = im.transform_1(data['train'], data['test'], data['target'])
        im2 = ImputerMeanPrim()
        data['train'], data['test'], data['target'] = im2.transform_1(data['train'], data['test'], data['target'])
        le = LabelEncoderPrim()
        data['train'], data['test'], data['target'] = le.transform_1(data['train'], data['test'], data['target'])

        shapley = PermShapley(
            dataset_name,
            data,
            pipeline_length,
            cache_path=f"aipipe_reprod/shapley/saves/cache_25-6/{dataset_name}.18.json",
            dump_interval=20,
            force_large_computation=True,
            n_processes=n_processes,
            output_shapley_values_path=None, #f'save/shapley_value_pos_vis/{dataset_name}_full.json'
        )

        # 从返回值中同时捕获 pipeline 和 acc
        pipeline, acc = shapley.run_full_enumerate_algorithm()

        with open(f'aipipe_reprod/shapley/saves/full_enum_sv18.tsv', 'a') as f:
            f.write(f'{dataset_name}\t{pipeline}\t{acc}\t{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\tmcts200\n')
            f.flush()

        # 使用 dump_final 保存任务结束后的完整缓存
        shapley.evaluator.dump_final(f'aipipe_reprod/shapley/saves/cache_25-6/{dataset_name}.18.json')  # <--- 修正后的调用

        logger.info(f"=== 数据集 {dataset_name} 处理完成 ===")
    except Exception as e:
        logger.error(f"处理数据集 {dataset_name} 时出错: {str(e)}", exc_info=True)
    finally:
        if shapley:  # 确保 cleanup 被调用
            shapley.evaluator.cleanup()
        logger.remove(logger_id)


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use', type=str, required=True, help='input the dataset names, split with comma')
    parser.add_argument('--parallel', type=int, default=7, help='input the parallel number')
    parser.add_argument('--n_processes', type=int, default=4, help='input the n_processes number')
    args = parser.parse_args()
    args.use = list(map(lambda x: x.strip(), args.use.split(',')))
    return args

if __name__ == '__main__':
    _metadatas = load_diffprep_meta()
    # _metadatas = get_deepline_metadatas()
    # args = parse_args()
    # run_full_enum(use=args.use, parallel=args.parallel, n_processes=args.n_processes)
    run_full_enum(_metadatas, use=[
        # 'abalone', 
        # 'ada_prior',
        # 'avila', 'connect-4', 'eeg', 
        # 'google', 
        # 'house_prices',
        # 'jungle_chess',
        # 'micro',
        # 'mozilla4',
        'obesity',
        # 'page-blocks',
        # 'pbcseq',
        # 'pol',
        # 'run_or_walk',
        # 'shuttle',
        # 'uscensus',
        # 'wall-robot-nav',
    ], parallel=3, n_processes=8)
    # manual_test('eeg', [16, 5, 5])
