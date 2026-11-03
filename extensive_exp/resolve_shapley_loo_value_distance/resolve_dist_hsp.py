import pandas as pd
import numpy as np
import sys
import os
from scipy import stats
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR'))  # type:ignore

def read_loo_values(dataset_name: str):
    BASE_PATH = 'save/sv_or_loo/loo'
    full_name = os.path.join(BASE_PATH, f'greedy_loo_small_{dataset_name}.csv')

    df = pd.read_csv(full_name, header=None, nrows=18*3)
    if df.shape[1] == 4:
        df.columns = ['opid', 'loo_value', 'slot', 'dataset_name']
    elif df.shape[1] == 5:
        df.columns = ['opid', 'loo_value', 'slot', 'gain', 'dataset_name']
    elif df.shape[1] == 6:
        df.columns = ['opid', 'loo_value', 'slot', 'gain', 'dataset_name', 'downstream']
    else:
        df.columns = ['opid', 'loo_value', 'slot', 'gain', 'dataset_name', 'downstream', 'time']

    return {
        'slot_0': df[df['slot'] == 0],
        'slot_1': df[df['slot'] == 1],
        'slot_2': df[df['slot'] == 2],
    }


def read_shap_seq_values(dataset_name: str):
    BASE_PATH = 'save/sv_or_loo/pipesv'
    full_name = os.path.join(BASE_PATH, f'{dataset_name}_sv_rand_3.csv')
    df = pd.read_csv(full_name)
    # columns = ['dataset', 'operation', 'slot', 'shapley_value', 'selected_pipe']
    df = df[df['operation'] >= 7]
    first_slot_value = df.iloc[0]['slot']
    return {
        'slot_0': df[df['slot'] == first_slot_value],
        'slot_1': df[df['slot'] == (first_slot_value + 1)],
        'slot_2': df[df['slot'] == (first_slot_value + 2)],
    }

def read_grid_shap_values(dataset_name: str):
    BASE_PATH = 'save/sv_or_loo/pipesv'
    full_name = os.path.join(BASE_PATH, f'{dataset_name}_sv_rand_3.csv')
    df = pd.read_csv(full_name)
    df = df[df['operation'] >= 7]
    grid = np.zeros((25, 3))
    first_slot_value = df.iloc[0]['slot']
    tmp_grid = np.zeros((25,3))
    last_slot = 0
    last_pipe = None
    cnt = 0
    for row_id, row in df.iterrows():
        if last_pipe != row['selected_pipe']: # last_slot > row['slot'] or (last_slot == max(df['slot']) and last_slot == row['slot'] and cnt >= 18):
            # copy the col with largest value to grid, and ensure the grid col is empty
            zero_cols = np.all(tmp_grid == 0, axis=0)
            tmp_grid[:, zero_cols] = -999
            largest_col_id = np.argmax(tmp_grid) % 3
            grid[:, largest_col_id] = tmp_grid[:, largest_col_id]
            tmp_grid = np.empty(tmp_grid.shape)
            cnt = 0

        tmp_grid[row['operation'], row['slot'] - first_slot_value] = row['shapley_value']
        last_slot = row['slot']
        last_pipe = row['selected_pipe']
        cnt += 1

    zero_cols = np.all(tmp_grid == 0, axis=0)
    tmp_grid[:, zero_cols] = -999
    largest_col_id = np.argmax(tmp_grid) % 3
    grid[:, largest_col_id] = tmp_grid[:, largest_col_id]
    grid[grid == -999] = 0.

    return {
        'slot_0': grid[:, 0].ravel(),
        'slot_1': grid[:, 1].ravel(),
        'slot_2': grid[:, 2].ravel(),
    }

def value_tuple(df: dict[str, pd.DataFrame]):
    return list(df.values())

def normalize(value: pd.Series):
    if value.max() == value.min():
        return value
    return (value - value.min()) / (value.max() - value.min())

def get_rank(value: pd.Series):
    return value.rank(method='min', ascending=False).astype(int)

def process_one_dataset(dataset_name: str):
    print(dataset_name)
    loo_df = value_tuple(read_loo_values(dataset_name))
    shap_list = read_grid_shap_values(dataset_name).values()

    dist = {}

    for slot, (loo, shap) in enumerate(zip(loo_df, shap_list)):
        norm_loo = normalize(loo['gain'].reset_index(drop=True))
        norm_shap = normalize(pd.Series(shap))
        # norm_loo = loo['gain']
        # norm_shap = shap['shapley_value']

        rank_loo = get_rank(norm_loo)
        rank_shap = get_rank(norm_shap)

        L1_dist: float = (norm_loo - norm_shap).abs().sum()
        rank_dist = (rank_loo - rank_shap).abs().sum()
        dist[f'norm_l1_slot_{slot}'] = L1_dist
        dist[f'rank_slot_{slot}'] = rank_dist

    return dist
    

if __name__ == '__main__':
    all_dataset_names = [
        'abalone','ada_prior','avila', 'connect-4',
        'eeg',
        'google','house_prices',
        'jungle_chess',
        'micro','mozilla4',
        'obesity',
        'page-blocks',
        'pbcseq','pol',
        'run_or_walk','shuttle',
        'uscensus',
        'wall-robot-nav','cmc','diabetes',
        'jm1','kc1','kc2','pc4',
        'phoneme','qsar-biodeg','vehicle',
    ]

    full_res = []

    for dataset_name in all_dataset_names:
        res = process_one_dataset(dataset_name)
        res['dataset_name'] = dataset_name
        full_res.append(res)

    full_res = pd.DataFrame(full_res)
    full_res.sort_index(inplace=True, axis=1)
    full_res.to_csv('save/sv_or_loo/slot_L1_hsp_norm_dist.csv', index=False)
