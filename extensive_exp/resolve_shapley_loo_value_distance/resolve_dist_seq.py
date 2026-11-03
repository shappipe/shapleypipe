import pandas as pd
import numpy as np
import sys
import os
from scipy import stats
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR'))  # type:ignore

def read_loo_values(dataset_name: str):
    BASE_PATH = 'saves/greedy'
    full_name = os.path.join(BASE_PATH, f'greedy_loo_{dataset_name}.csv')

    df = pd.read_csv(full_name, header=None, nrows=18*3)
    if df.shape[1] == 4:
        df.columns = ['opid', 'loo_value', 'slot', 'dataset_name']
    elif df.shape[1] == 5:
        df.columns = ['opid', 'loo_value', 'slot', 'gain', 'dataset_name']
    else:
        df.columns = ['opid', 'loo_value', 'slot', 'gain', 'dataset_name', 'downstream']

    return {
        'slot_0': df[df['slot'] == 0],
        'slot_1': df[df['slot'] == 1],
        'slot_2': df[df['slot'] == 2],
    }


def read_shap_seq_values(dataset_name: str):
    BASE_PATH = 'save/shapleyvalue_seq'
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
    shap_df = value_tuple(read_shap_seq_values(dataset_name))

    dist = {}

    for slot, (loo, shap) in enumerate(zip(loo_df, shap_df)):
        # norm_loo = normalize(loo['loo_value'])
        # norm_shap = normalize(shap['shapley_value'])
        norm_loo = loo['gain']
        norm_shap = shap['shapley_value']

        rank_loo = get_rank(norm_loo)
        rank_shap = get_rank(norm_shap)

        L1_dist: float = (norm_loo - norm_shap).abs().mean()
        rank_dist = (rank_loo - rank_shap).abs().sum()
        dist[f'norm_l1_slot_{slot}'] = L1_dist
        dist[f'rank_slot_{slot}'] = rank_dist

    return dist
    

if __name__ == '__main__':
    all_dataset_names = [
        'abalone','ada_prior','avila', 
        'connect-4',
        'eeg',
        'google','house_prices','jungle_chess',
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
    full_res.to_csv('save/shap_when_choose_hsp/slot_L1_non_norm_dist.csv', index=False)
