import sys
import os
from scipy import stats
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ""))
import pandas as pd
import numpy as np
import json
from scipy.special import softmax

np.set_printoptions(precision=3, suppress=True)

def read_shapley(data: list[dict]):
    sv_grid = np.zeros((25, 6))
    cnt = 6
    last_slot = -1
    for d in data[::-1]:
        sv = d['shapley_value']
        op = d['operation']
        slot = d['slot']
        if slot != last_slot:
            cnt -= 1
            last_slot = slot
        sv_grid[op, cnt] = sv

    return sv_grid


def p_i_appear_before_j(sv_row: np.ndarray, j: int):
    prob = softmax(sv_row)
    # print('prob', prob)
    res = 1
    for pit in prob[:j]:
        res *= 1 - pit
    return (1 - res) * prob[j]


def read_sv_and_calc():
    BASE = 'save/shapley_value_pos_vis'
    files = sorted(filter(lambda x: x.endswith('.json'), os.listdir(BASE)))

    for file in files[2:]:
        with open(os.path.join(BASE, file), 'r') as f:
            data = json.load(f)
        sv_grid = read_shapley(data)
        print(sv_grid)

        pj_grid = np.zeros((25, 6))
        for j in range(1, 6):
            for i in range(25):
                pj_grid[i, j] = p_i_appear_before_j(sv_grid[i, :], j)
        
        print(pj_grid)
        print(pj_grid.mean(axis=0))

        break

read_sv_and_calc()


# [0., 0.028, 0.052, 0.071, 0.084, 0.1]
