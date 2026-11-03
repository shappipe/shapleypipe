import pandas as pd
import numpy as np
import sys
import os
from scipy import stats
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR'))

def calc_dist_between_shapley_and_loo(dataset_name, loo_path, shap_grid_path):
    print(f'Processing {dataset_name}')

    # loo_path = 'saves/greedy/greedy_loo_pc4.csv'
    # shap_grid_path = 'save/shapley_value_pos_vis/pc4_sv_rand_3.csv'

    loo_df = pd.read_csv(loo_path, header=None)
    loo_df.columns = ['op_idx', 'acc', 'slot', 'dataset']

    shap_grid_df = pd.read_csv(shap_grid_path)
    shap_grid_df['selected_pipe'] = shap_grid_df['selected_pipe'].apply(lambda x: eval(x))
    slot_to_minus = len(shap_grid_df.loc[len(shap_grid_df) - 1, 'selected_pipe']) - 3
    shap_grid_df['selected_pipe'] = shap_grid_df['selected_pipe'].apply(lambda x: x[-3:])
    shap_grid_df['slot'] = shap_grid_df['slot'] - slot_to_minus
    # columns = [dataset,operation,slot,shapley_value,selected_pipe]
    shap_grid_df = shap_grid_df[shap_grid_df['operation'] >= 7]



    def not_eq_slot(arr, brr):
        for i, (a, b) in enumerate(zip(arr, brr)):
            if a != b:
                return i
        return -1

    prev_selected = [-1, -1, -1]
    sel_order = []
    shap_grid_df['slot_to_sel'] = -1
    last_i = 0

    for i, row in shap_grid_df.iterrows():
        selected_pipe = row['selected_pipe']
        idx = not_eq_slot(selected_pipe, prev_selected)
        if idx != -1:
            prev_selected[idx] = selected_pipe[idx]
            sel_order.append(idx)

            shap_grid_df.loc[last_i:i, 'slot_to_sel'] = idx
            last_i = i
    
    shap_grid_df.loc[last_i:, 'slot_to_sel'] = prev_selected.index(-1)

    shap_grid_df = shap_grid_df[shap_grid_df['slot_to_sel'] + 1 == shap_grid_df['slot']]

    sv_shaps = {}

    for slot in shap_grid_df['slot'].unique():
        slot_df = shap_grid_df[shap_grid_df['slot'] == slot]
        op_slot_shapley_list = []
        for i, row in slot_df.iterrows():
            op_shapley = row['shapley_value']
            op_slot_shapley_list.append({'op_idx': row['operation'], 'shapley_value': op_shapley})

        for op_idx in range(7, 25):
            if op_idx not in map(lambda x: x['op_idx'], op_slot_shapley_list):
                op_slot_shapley_list.append({'op_idx': op_idx, 'shapley_value': 0})

        # calc rank
        op_slot_shapley_list = pd.DataFrame(op_slot_shapley_list)
        op_slot_shapley_list['shapley_rank'] = op_slot_shapley_list['shapley_value'].rank(method='average', ascending=False).astype(int)

        # print(op_slot_shapley_list)
        sv_shaps[int(slot)] = op_slot_shapley_list

    for slot in loo_df['slot'].unique():
        slot_df = loo_df[loo_df['slot'] == slot]
        op_slot_loo_value = slot_df[['op_idx', 'acc']].rename(columns={'acc': 'loo_value'})
        # op_slot_loo_value['loo_rank'] = op_slot_loo_value['loo_value'].rank(method='average', ascending=False).astype(int)
        sv_shaps[int(slot)+1] = pd.merge(sv_shaps[int(slot)+1], op_slot_loo_value, on='op_idx', how='left')
        sv_shaps[int(slot)+1]['loo_rank'] = sv_shaps[int(slot)+1]['loo_value'].rank(method='average', ascending=False).astype(int)

    # print(sv_shaps)

    dists = {}
    spearman_dists = {}
    value_dists = {}
    for k, v in sv_shaps.items():
        shap_rank = v['shapley_rank']
        loo_rank = v['loo_rank']
        rank_dist = sum(abs(shap_rank - loo_rank))  # manhattan distance
        dists[k] = rank_dist
        spearman_dists[k] = stats.spearmanr(shap_rank, loo_rank)[0]  # spearman correlation coefficient

        shap_value = v['shapley_value'].values
        shap_value = (shap_value - shap_value.min()) / (shap_value.max() - shap_value.min())
        loo_value = v['loo_value'].values
        loo_value = (loo_value - loo_value.min()) / (loo_value.max() - loo_value.min())
        value_dist = sum(abs(shap_value - loo_value))  # manhattan distance
        value_dists[k] = value_dist

    return {'dataset': dataset_name, 'dist': dists, 'spearman_dist': spearman_dists, 'value_dist': value_dists}

if __name__ == '__main__':
    all_dataset_names = [
        'abalone','ada_prior','avila', 'connect-4',
        'eeg',
        'google','house_prices','jungle_chess',
        'micro','mozilla4','obesity','page-blocks',
        'pbcseq','pol','run_or_walk','shuttle',
        'uscensus','wall-robot-nav','cmc','diabetes',
        'jm1','kc1','kc2','pc4',
        'phoneme','qsar-biodeg','vehicle',
    ]

    full_res = []
    for dataset_name in all_dataset_names:
        loo_path = f'saves/greedy/greedy_loo_{dataset_name}.csv'
        shap_grid_path = f'save/shapley_value_pos_vis/{dataset_name}_sv_rand_3.csv'
        res = calc_dist_between_shapley_and_loo(dataset_name, loo_path, shap_grid_path)
        print(res)
        full_res.append({
            'dataset': dataset_name,
            'rank_dist1': res['dist'][1],
            'rank_dist2': res['dist'][2],
            'rank_dist3': res['dist'][3],
            'spearman_dist1': res['spearman_dist'][1],
            'spearman_dist2': res['spearman_dist'][2],
            'spearman_dist3': res['spearman_dist'][3],
            'value_dist1': res['value_dist'][1],
            'value_dist2': res['value_dist'][2],
            'value_dist3': res['value_dist'][3],
        })

    full_res = pd.DataFrame(full_res).reindex(columns=['dataset', 
                                                       'rank_dist1', 'rank_dist2', 'rank_dist3', 
                                                       'spearman_dist1', 'spearman_dist2', 'spearman_dist3',
                                                       'value_dist1', 'value_dist2', 'value_dist3'])
    full_res.to_csv('saves/resolve_rank.csv', index=False)
