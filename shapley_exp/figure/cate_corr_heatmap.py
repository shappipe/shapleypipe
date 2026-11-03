from itertools import combinations
import os
import sys
from typing import Literal
import json
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv('PROJ_DIR', ''))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)
plt.rcParams['font.family'] = 'Linux Libertine O'
plt.rcParams['axes.unicode_minus'] = False

from extensive_exp.cate_validation.clustering_analysis import perform_clustering, compute_metrics

# BASE = 'save/shapvalue_grid'
BASE = 'shapley_exp/figure/saves/shapvalue_grid'

diffprep_datasets = [
'abalone.npy',
'ada_prior.npy',
'avila.npy',
'connect-4.npy',
'eeg.npy',
'google.npy',
'house_prices.npy',
'jungle_chess.npy',
'micro.npy',
'mozilla4.npy',
'obesity.npy',
'page-blocks.npy',
'pbcseq.npy',
'pol.npy',
'run_or_walk.npy',
'shuttle.npy',
'uscensus.npy',
'wall-robot-nav.npy',
]

cc18_datasets = [
'cmc.npy',
'diabetes.npy',
'jm1.npy',
'kc1.npy',
'kc2.npy',
'pc4.npy',
'phoneme.npy',
'qsar-biodeg.npy',
'vehicle.npy',
]


def load_all():
    files = sorted(filter(lambda x: x.endswith('.npy'), 
                          os.listdir(BASE)))
    all_e = []
    for f in files:
        e = np.load(os.path.join(BASE, f))
        all_e.append(e)
    all_e = np.array(all_e) # shape (file_num, 25, 6)

    # all_e = np.swapaxes(all_e[[0,1,2,3,4,7,9,16,18,23], :, :], 0, 1)
    all_e = np.swapaxes(all_e, 0, 1)
    
    all_e = all_e.reshape((25, -1))
    # insert row 23 into 17
    # all_e = np.insert(all_e, 17, all_e[23], axis=0)
    # all_e = np.delete(all_e, 24, axis=0)

    print(all_e.shape)

    similarity_matrix = np.corrcoef(all_e)
    similarity_matrix = np.nan_to_num(similarity_matrix)
    print(similarity_matrix.shape)
    # print(similarity_matrix)

    linkage_matrix, predicted_labels = perform_clustering(
        all_e,
        n_clusters=5
    )

    print("\n[Step 5/6] Computing evaluation metrics...")
    metrics = compute_metrics(
        similarity_matrix,
        np.array([0,0,0,0,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4]),
        predicted_labels - 1  # fcluster returns 1-indexed labels
    )

    print("=" * 60)
    print(f"Within-category correlation:  {metrics['rho_within_mean']:.3f} ± {metrics['rho_within_std']:.3f}")
    print(f"Between-category correlation: {metrics['rho_between_mean']:.3f} ± {metrics['rho_between_std']:.3f}")
    print(f"Separation ratio:             {metrics['separation_ratio']:.2f}×")
    print(f"Silhouette score:             {metrics['silhouette_score']:.3f}")
    print(f"Adjusted Rand Index:          {metrics['adjusted_rand_index']:.3f}")
    print("=" * 60)


    # np.save('save/shapvalue_grid/similarity_matrix.npy', similarity_matrix)
    # fig, ax = plt.subplots(figsize=(6, 4))
    # sns.heatmap(
    #     similarity_matrix,
    #     cmap='RdBu_r',
    #     center=0,
    #     vmin=-1,
    #     vmax=1,
    #     square=True,
    #     linewidths=0.5,
    #     cbar_kws={'label': 'Shapley Correlation', 'shrink': 0.8},
    #     ax=ax,
    #     xticklabels=[],  # Hide individual operator names (too crowded)
    #     yticklabels=[]
    # )
    # fig.savefig('save/shapvalue_grid/similarity_matrix.pdf')

load_all()

line_indices = [
    0,
    4,
    7, 
    15,
    24,
    25,
]

labels = [
    ('Impute', 1.5),
    ('Encode', 5),
    ('Scale', 10.5),
    ('Feat Engineer', 19),
    ('Sel', 24),
]

def load_all_split_j():
    files = sorted(filter(lambda x: x.endswith('.npy'), 
                          os.listdir(BASE)))
    all_e = []
    for f in files:
        e = np.load(os.path.join(BASE, f))
        f = 0
        if np.all(e[:, 0] == 0):
            for i in range(1,6):
                if np.any(e[:, i] != 0):
                    e[:, 0] = e[:, i]
                    f = 1
                    print('copied', i, 'to', 0)
                    break
        if np.all(e[:, 1] == 0):
            for i in range(2,6-f):
                if np.any(e[:, i+f] != 0):
                    e[:, 1] = e[:, i+f]
                    print('copied', i+f, 'to', 1)
                    f = 2
                    break
        if np.all(e[:, 2] == 0):
            for i in range(3,6-f):
                if np.any(e[:, i+f] != 0):
                    e[:, 2] = e[:, i+f]
                    print('copied', i+f, 'to', 2)
                    f = 3
                    break
        all_e.append(e)
    all_e = np.array(all_e) # shape (file_num, 25, 6)
    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    fig.subplots_adjust(wspace = .1,hspace = 0.)

    sim_by_j = []

    for j in range(6):
        # get the jth column
        e = all_e[:, :, j].transpose((1, 0))
        # insert row 23 into 15
        e = np.insert(e, 15, e[23], axis=0)
        e = np.delete(e, 24, axis=0)

        similarity_matrix = np.corrcoef(e)
        similarity_matrix = np.nan_to_num(similarity_matrix)
        print(similarity_matrix.shape)
        
        sim_by_j.append(similarity_matrix[:, :])

        row = j // 3
        col = j % 3
        # ax[row, col].imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        sns.heatmap(
            similarity_matrix,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar=False,
            # cbar_kws={'label': 'Shapley Correlation', 'shrink': 0.8},
            ax=ax[row, col],
            xticklabels=[],  # Hide individual operator names (too crowded)
            yticklabels=[]
        )
        ax[row, col].set_title(f'Position={j+1}', fontsize=12, y=-0.1)

        for line_index in line_indices:
            ax[row, col].axvline(line_index, color='black', linewidth=0.5)
            ax[row, col].axhline(line_index, color='black', linewidth=0.5)

        for i, (label, pos) in enumerate(labels):
            if i == 0:
                shift = 0.5
            else:
                shift = 0
            # if row == 0:
            ax[row, col].text(pos-shift+0.5, -1.5, label, ha='center', va='top', fontsize=8)
            # if col == 0:
            ax[row, col].text(-0.5, pos-shift+0.5, label, ha='right', va='center', fontsize=8, rotation=90)

        # linkage_matrix, predicted_labels = perform_clustering(
        #     e,
        #     n_clusters=5
        # )

        # print("\n[Step 5/6] Computing evaluation metrics...")
        # metrics = compute_metrics(
        #     similarity_matrix,
        #     np.array([0,0,0,0,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4]),
        #     predicted_labels - 1  # fcluster returns 1-indexed labels
        # )
    fig.subplots_adjust(left=0.03, bottom=0, right=0.92, top=1)
    cax = plt.axes([0.932, 0.015, 0.015, 0.933])
    cbar = fig.colorbar(ax[0, 0].get_children()[0], cax=cax)
    cbar.ax.tick_params(labelsize=12)
    # cbar.set_label('Shapley Correlation', fontsize=12, loc='right')
    sim_by_j = np.array(sim_by_j)
    sim_by_j_mean = sim_by_j.mean(axis=0)
    print(sim_by_j_mean.shape)
    print(sim_by_j_mean)
    # fig.tight_layout()
    fig.savefig('shapley_exp/figure/saves/similarity_matrix_split_j.pdf')
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        sim_by_j_mean,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Shapley Value Correlation', 'shrink': 0.8},
        ax=ax2,
        xticklabels=[],  # Hide individual operator names (too crowded)
        yticklabels=[]
    )
    ax2.spines['top'].set_linewidth(0.5)
    ax2.spines['bottom'].set_linewidth(0.5)
    ax2.spines['left'].set_linewidth(0.5)
    ax2.spines['right'].set_linewidth(0.5)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for line_index in line_indices:
        ax2.axvline(line_index, color='black', linewidth=0.5)
        ax2.axhline(line_index, color='black', linewidth=0.5)

    for i, (label, pos) in enumerate(labels):
        if i == 0:
            shift = 0.5
        else:
            shift = 0
        ax2.text(pos-shift+0.5, -1, label, ha='center', va='top', fontsize=11)
        ax2.text(-0.5, pos-shift+0.5, label, ha='right', va='center', fontsize=11, rotation=90)

    fig2.tight_layout()
    fig2.savefig('shapley_exp/figure/saves/similarity_matrix_split_j_mean.pdf')
    plt.close()

    # upper = np.triu(np.ones_like(sim_by_j_mean, dtype=bool), k=1)

    intra_mask = np.zeros((len(sim_by_j_mean), len(sim_by_j_mean)), dtype=bool)
    intra_mask[0:4,0:4] = True
    intra_mask[4:7,4:7] = True
    intra_mask[7:15,7:15] = True
    intra_mask[15:24,15:24] = True
    intra_mask[24,24] = True

    inter_mask = ~intra_mask

    inter_vals = sim_by_j_mean[inter_mask]
    intra_vals = sim_by_j_mean[intra_mask]

    print("Intra mean std median:", np.mean(intra_vals), np.std(intra_vals), np.median(intra_vals))
    print("Inter mean std median:", np.mean(inter_vals), np.std(inter_vals), np.median(inter_vals))

    print("Intra positive ratio:", np.mean(intra_vals > 0))
    print("Inter positive ratio:", np.mean(inter_vals > 0))

    from scipy.stats import mannwhitneyu
    u_stat, p_value = mannwhitneyu(
        intra_vals,
        inter_vals,
        alternative='greater'
    )

    print("U statistic:", u_stat)
    print("p-value:", p_value)
    
load_all_split_j()
