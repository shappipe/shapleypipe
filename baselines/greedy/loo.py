import argparse
import sys
import os
import json
import time
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ''))

from utils.operators.q_action_provider import QActionProvider

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.dataloader.load_diffprep import path_to_df_dict, load_diffprep_csv_and_labels
from utils.dataloader.load_cc18 import list_cc18, load_cc18
from utils.datatypes import DatasetType
import multiprocessing as mp
from multiprocessing.pool import Pool as ProcessPool
from utils.operators.primitives import ImputerCatPrim

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

def process_dataset(set_name, meta, early_stop: bool = False):
    if set_name == 'diffprep':
        dataset_name = meta['name']
        dataset_path = meta['path']
        label = meta['label']
        data = path_to_df_dict(dataset_path, label)
    elif set_name == 'cc18':
        dataset_name = meta
        data = load_cc18(dataset_name)
    else:
        raise ValueError(f"Unknown dataset set name: {set_name}")

    string_cols = data['train'].select_dtypes(include=['object']).columns.tolist()
    has_cate_miss = any(data['train'][col].isnull().any() for col in string_cols)
    has_cate = len(string_cols) > 0
    has_num_miss = any(data['train'].select_dtypes(include=['number']).isnull().any())

    original_train_x = data['train'].copy()
    original_train_y = data['target'].copy()
    original_test_x = data['test'].copy()
    original_test_y = data['target_test'].copy()

    train_x, val_x, train_y, val_y = train_test_split(data['train'], data['target'], test_size=0.2, random_state=42)

    if len(train_x) > 20000:
        train_x, _, train_y, _ = train_test_split(train_x, train_y, train_size=20000, random_state=42)

    full_pipe = []
    current_op = -1

    start_time = time.time()

    remaining_slots = 3
    if has_cate_miss:
        # remaining_slots -= 1
        im = ImputerCatPrim()
        train_x, val_x, train_y = im.transform_1(train_x, val_x, train_y)

        current_op = 0
    
        lr = QActionProvider.lr()
        try:
            y_pred = lr.transform_1(train_x, train_y, val_x)
            last_acc = accuracy_score(val_y, y_pred)
        except:
            last_acc = 0.0
        full_pipe.append({'op_idx': current_op, 'acc': last_acc})
        print(dataset_name, full_pipe)
    else:
        print('skip cate missing')

    if has_cate:
        # remaining_slots -= 1
        accs = []
        prev_data = {
            'train': train_x.copy(),
            'val': val_x.copy(), 
            'target': train_y.copy(),
        }
        for op_idx in QActionProvider.spans[2]:
            op = QActionProvider.get(op_idx)
            try:
                temp_train_x, temp_val_x, temp_train_y = op.transform_1(prev_data['train'], prev_data['val'], prev_data['target'])
                lr = QActionProvider.lr()
                y_pred = lr.transform_1(temp_train_x, temp_train_y, temp_val_x)
                acc = accuracy_score(val_y, y_pred)
            except:
                acc = 0.0
            accs.append({'op_idx': op_idx, 'acc': acc})
        
        best_acc = max(accs, key=lambda x: x['acc'])
        
        full_pipe.append(best_acc)
        current_op = best_acc['op_idx']
        train_x, val_x, train_y = QActionProvider.get(current_op).transform_1(prev_data['train'], prev_data['val'], prev_data['target'])
        print(dataset_name, full_pipe)
    else:
        print('skip cate encoding')

    if has_num_miss:
        # remaining_slots -= 1
        accs = []
        prev_data = {
            'train': train_x.copy(),
            'val': val_x.copy(),
            'target': train_y.copy(),
        }
        for op_idx in QActionProvider.spans[1]:
            op = QActionProvider.get(op_idx)
            try:
                temp_train_x, temp_val_x, temp_train_y = op.transform_1(prev_data['train'], prev_data['val'], prev_data['target'])
                lr = QActionProvider.lr()
                y_pred = lr.transform_1(temp_train_x, temp_train_y, temp_val_x)
                acc = accuracy_score(val_y, y_pred)
            except:
                acc = 0.0
            accs.append({'op_idx': op_idx, 'acc': acc})
        
        best_acc = max(accs, key=lambda x: x['acc'])
        full_pipe.append(best_acc)
        current_op = best_acc['op_idx']
        train_x, val_x, train_y = QActionProvider.get(current_op).transform_1(prev_data['train'], prev_data['val'], prev_data['target'])
        print('use num imputer', dataset_name, full_pipe)
    else:
        print('skip num impute')

    feature_spans = [
        *QActionProvider.spans[3],
        *QActionProvider.spans[4],
        *QActionProvider.spans[5],
    ]

    loo_values = []
    y_pred = QActionProvider.lr().transform_1(train_x, train_y, val_x)
    last_acc = accuracy_score(val_y, y_pred)

    for r in range(remaining_slots):
        accs = []
        prev_data = {
            'train': train_x.copy(),
            'val': val_x.copy(),
            'target': train_y.copy(),
        }
        for op_idx in feature_spans:
            op = QActionProvider.get(op_idx)
            temp_train_x, temp_val_x, temp_train_y = op.transform_1(prev_data['train'], prev_data['val'], prev_data['target'])
            lr = QActionProvider.lr()
            y_pred = lr.transform_1(temp_train_x, temp_train_y, temp_val_x)
            acc = accuracy_score(val_y, y_pred)
            accs.append({'op_idx': op_idx, 'acc': acc, 'slot': r, 'gain': acc-last_acc})

            # lr = QActionProvider.lr()
            # y_pred = lr.transform_1(original_train_x, original_train_y, original_test_x)
            # facc = accuracy_score(original_test_y, y_pred)
            print(dataset_name, op_idx, 'got validation', acc)
        
        best_acc = max(accs, key=lambda x: x['acc'])
        last_acc = best_acc['acc']

        if early_stop and len(full_pipe) > 0:
            if best_acc['acc'] <= full_pipe[-1]['acc']:
                print('early stop at', best_acc['op_idx'])
                break

        full_pipe.append(best_acc)
        current_op = best_acc['op_idx']
        train_x, val_x, train_y = QActionProvider.get(current_op).transform_1(prev_data['train'], prev_data['val'], prev_data['target'])

        print(dataset_name, full_pipe)

        loo_values.extend(accs)

    final_train_x = original_train_x.copy()
    final_train_y = original_train_y.copy()
    final_test_x = original_test_x.copy()
    final_test_y = original_test_y.copy()
    
    for obj in full_pipe:
        op = QActionProvider.get(obj['op_idx'])
        final_train_x, final_test_x, final_train_y = op.transform_1(final_train_x, final_test_x, final_train_y)

    lr = QActionProvider.lr()
    y_pred = lr.transform_1(final_train_x, final_train_y, final_test_x)
    acc = accuracy_score(final_test_y, y_pred)
    print('final accuracy', acc)

    end_time = time.time()
    total_time = end_time - start_time

    return {'name': dataset_name, 
            'pipe': list(map(lambda x: x['op_idx'], full_pipe)), 
            'final_acc': acc, 
            'val_acc': list(map(lambda x: x['acc'], full_pipe)),
            'total_time': total_time}, loo_values


def process_all_datasets(early_stop: bool = False):
    meta = load_diffprep_meta()
    results = []
    for m in meta:
        r = process_dataset(m, early_stop)
        results.append(r)
    return results

def process_datasets_mp(early_stop: bool = False):
    meta = load_diffprep_meta()
    params = [(m, early_stop) for m in meta]
    with ProcessPool(2) as p:
        results = p.starmap(process_dataset, params)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Process datasets')
    parser.add_argument('--set', type=str, default='diffprep', help='Dataset set to use (diffprep or cc18)')
    parser.add_argument('--use', type=str, default=None, help='Specific dataset to use (optional)')
    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping')
    return parser.parse_args()


def run_diffprep(args):
    _metadatas = load_diffprep_meta()
    m = None
    for meta in _metadatas:
        if meta['name'] == args.use:
            m = meta
            break
    if m is None:
        print(f"Cannot find dataset {args.use}")
        exit(1)
    dataset_name = m['name']
    results, loo_values = process_dataset('diffprep', m, args.early_stop)

    return results, loo_values, dataset_name

def run_cc18(args):
    dataset_name = args.use
    results, loo_values = process_dataset('cc18', dataset_name, args.early_stop)

    return results, loo_values, dataset_name


if __name__ == '__main__':
    # args = parse_args()
    # if args.set == 'diffprep':
    #     results, loo_values, dataset_name = run_diffprep(args)

    # elif args.set == 'cc18':
    #     results, loo_values, dataset_name = run_cc18(args)

    #### Diffprep ####
    # uses = ['abalone', 'ada_prior', 'avila', 
    #         'connect-4', 'jungle_chess', 'eeg', 
    #         'google', 'house_prices', 'micro', 
    #         'mozilla4', 'obesity', 'page-blocks', 
    #         'pbcseq', 'pol', 'run_or_walk', 
    #         'shuttle', 'uscensus', 'wall-robot-nav']
    
    # for use in uses:
    #     args = {'use': use, 'early_stop': True}
        
    #     t = time.time()
    #     results, loo_values, dataset_name = run_diffprep(argparse.Namespace(**args))
    #     t = time.time() - t
        
    #     loo_df = pd.DataFrame(loo_values)
    #     loo_df['name'] = dataset_name
    #     loo_df['downstream'] = 'lr'
    #     loo_df['time'] = t

    #     res = pd.DataFrame(results).sort_values(by='name', inplace=False)
    #     res.to_csv('saves/greedy.csv', index=False, mode='a', header=False)
    #     loo_df.to_csv(f'saves/greedy/greedy_loo_{dataset_name}.csv', index=False, mode='a', header=False)

    #### cc18 ####
    uses = ['cmc','diabetes','jm1','kc1','kc2','pc4','phoneme','qsar-biodeg','vehicle'] 

    for use in uses:
        args = {'use': use, 'early_stop': True}
        
        t = time.time()
        results, loo_values, dataset_name = run_cc18(argparse.Namespace(**args))
        t = time.time() - t
        
        loo_df = pd.DataFrame(loo_values)
        loo_df['name'] = dataset_name
        loo_df['downstream'] = 'lr'
        loo_df['time'] = t

        res = pd.DataFrame(results).sort_values(by='name', inplace=False)
        res.to_csv('saves/greedy.csv', index=False, mode='a', header=False)
        loo_df.to_csv(f'saves/greedy/greedy_loo_{dataset_name}.csv', index=False, mode='a', header=False)


