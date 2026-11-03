import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
plt.rcParams['font.family'] = 'Linux Libertine O'
plt.rcParams['axes.unicode_minus'] = False

BASE_PATH='saves/interaction/max_inter_by_op'
all_files = sorted(os.listdir(BASE_PATH))

grid_value = []
grid_operator = []

for f in all_files:
    name = f.split('.')[0]
    grid = pd.read_csv(os.path.join(BASE_PATH, f), sep='\t')
    grid_value.append((name, *grid['interaction_value'].values.ravel()))
    grid_operator.append((name, *grid['operator_id'].values.ravel()))

grid_value = pd.DataFrame(grid_value, columns=['Dataset', 'pos1', 'pos2', 'pos3', 'pos4', 'pos5', 'pos6'])
grid_value.drop(columns=['pos6'], inplace=True)
grid_value['average'] = grid_value.iloc[:, 1:].mean(axis=1)
grid_operator = pd.DataFrame(grid_operator, columns=['Dataset', 'pos1', 'pos2', 'pos3', 'pos4', 'pos5', 'pos6'])


data = '''
Dataset	TPOT	AUSK	SAGA	L2C	CtxPipe	HAI-AI	DP	CatDB	GPT4o-ZS	GPT4o-FS	LOO	MC-SP	H-SP
abalone	0.289	0.243	0.255	0.273	0.287	0.260	0.271	0.258	0.267	0.250	0.245	0.281	0.278
ada_prior	0.823	0.832	0.833	0.791	0.818	0.801	0.846	0.835	0.839	0.831	0.821	0.845	0.845
avila	0.593	0.539	0.636	0.678	0.759	0.630	0.633	0.739	0.609	0.679	0.754	0.842	0.941
connect-4	0.658	0.660	0.758	0.747	0.763	0.775	0.702	0.759	0.713	0.657	0.793	0.794	0.794
eeg	0.594	0.647	0.658	0.848	0.740	0.556	0.683	0.529	0.623	0.834	0.740	0.857	0.864
google	0.602	0.558	0.596	0.641	0.590	0.550	0.661	0.602	0.548	0.648	0.675	0.685	0.665
house_prices	0.938	0.912	0.913	0.935	0.818	0.928	0.952	0.906	0.904	0.928	0.935	0.938	0.938
jungle_chess	0.682	0.683	0.745	0.814	0.861	0.760	0.687	0.747	0.760	0.807	0.861	0.862	0.847
micro	0.558	0.565	0.556	0.613	0.605	0.633	0.593	0.590	0.567	0.639	0.635	0.633	0.643
mozilla4	0.890	0.849	0.932	0.939	0.940	0.870	0.927	0.820	0.867	0.931	0.940	0.938	0.934
obesity	0.901	0.695	0.751	0.790	0.868	0.768	0.874	0.724	0.851	0.816	0.891	0.905	0.901
page-blocks	0.967	0.954	0.849	0.973	0.965	0.935	0.973	0.953	0.955	0.908	0.968	0.971	0.970
pbcseq	0.715	0.681	0.866	0.712	0.805	0.733	0.725	0.644	0.704	0.694	0.758	0.784	0.784
pol	0.894	0.885	0.888	0.926	0.949	0.916	0.916	0.947	0.921	0.888	0.981	0.966	0.982
run_or_walk	0.826	0.710	0.832	0.920	0.956	0.915	0.912	0.795	0.718	0.859	0.972	0.973	0.984
shuttle	0.933	0.926	0.405	0.986	1.000	0.951	0.999	0.999	0.994	0.984	1.000	0.999	1.000
uscensus	0.828	0.825	0.835	0.826	0.845	0.807	0.852	0.851	0.841	0.851	0.852	0.847	0.846
wall-robot-nav	0.754	0.692	0.841	0.904	0.946	0.896	0.913	0.898	0.812	0.941	0.947	0.951	0.952
cmc	0.534	0.486	0.550	0.595	0.534	0.549	0.510	0.557	0.568	0.568	0.574	0.595	0.615
diabetes	0.831	0.769	0.772	0.857	0.766	0.773	0.765	0.750	0.766	0.831	0.818	0.857	0.870
jm1	0.812	0.809	0.817	0.818	0.809	0.809	0.807	0.797	0.812	0.812	0.811	0.815	0.824
kc1	0.867	0.858	0.861	0.872	0.872	0.865	0.865	0.844	0.834	0.867	0.872	0.872	0.882
kc2	0.811	0.826	0.856	0.830	0.792	0.857	0.846	0.815	0.830	0.736	0.811	0.830	0.868
pc4	0.925	0.898	0.904	0.924	0.904	0.914	0.900	0.897	0.890	0.890	0.904	0.925	0.938
phoneme	0.832	0.760	0.786	0.867	0.824	0.772	0.756	0.750	0.751	0.823	0.867	0.867	0.882
qsar-biodeg	0.849	0.882	0.879	0.887	0.858	0.867	0.853	0.858	0.868	0.821	0.849	0.906	0.906
vehicle	0.847	0.778	0.787	0.903	0.788	0.812	0.740	0.701	0.741	0.871	0.894	0.906	0.906
'''

df = pd.read_csv(StringIO(data), sep='\t')
df.drop(columns=['MC-SP'], inplace=True)

avg_col = pd.DataFrame({
    'Dataset':df['Dataset'], 
    'diff': df['H-SP'] - df.iloc[:, 1:-1].mean(axis=1)})
# print(avg_col)
to_draw = pd.merge(grid_value[['Dataset', 'average']], avg_col[['Dataset', 'diff']], on='Dataset', how='inner')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
sns.heatmap(grid_value.iloc[:, 1:], ax=ax1, yticklabels=grid_value['Dataset'].to_list(),
            annot=True, fmt=".3f", cmap="Reds", cbar_kws={'label': 'Interaction Value'},
            vmin=0., vmax=0.1, linewidths=0.5)
ax1.axvline(x=0, color='black', linewidth=1)
ax1.axhline(y=0, color='black', linewidth=1)
ax1.axvline(x=5, color='black', linewidth=0.5)
ax1.axvline(x=6, color='black', linewidth=0.5)
ax1.axvline(x=7, color='black', linewidth=1)
ax1.axhline(y=len(grid_value), color='black', linewidth=1)
ax1.set_title('Heat Map of Interaction')
ax1.set_ylabel('Dataset')


ax2.scatter(to_draw['average'], to_draw['diff'], alpha=0.7, s=50)
ax2.set_xlabel('Interaction')
ax2.set_ylabel('HSP - Baselines')
ax2.set_title('Interaction vs HSP-Baselines Difference')
ax2.grid(True, alpha=0.3)
z = np.polyfit(to_draw['average'], to_draw['diff'], 1)
p = np.poly1d(z)
x_line = np.linspace(to_draw['average'].min(), to_draw['average'].max(), 100)
ax2.plot(x_line, p(x_line), 'r--', label=f'Linear fit: y={z[0]:.3f}x+{z[1]:.3f}')
ax2.legend(loc='upper right')

from scipy.stats import pearsonr, spearmanr
# print('correlation', pearsonr(to_draw['average'], to_draw['diff']))
# print('spearman', spearmanr(to_draw['average'], to_draw['diff']))
print('corr', np.corrcoef(to_draw['average'], to_draw['diff']))

plt.tight_layout()
plt.savefig('aipipe_reprod/shapley/sp_interaction/scatter_inte_disij.pdf')

