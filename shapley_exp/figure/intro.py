# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import seaborn as sns

"""
SIMPLIFIED VERSION FOR INTRODUCTION
Only includes Panel (a) and Panel (b)
Panel (c) can be moved to Experiments section
"""

# Set academic style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'Linux Libertine O'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# %% fig a
from matplotlib.patheffects import withStroke
fig, ax = plt.subplots(figsize=(3.5, 2.5))
methods = ['Random\nSearch', 'CtxPipe\n(SOTA)', 'Optimal\n(Exhaustive)']
accuracies = [0.598, 0.759, 0.941]
colors = ['#d62728', '#ff7f0e', '#2ca02c']
bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, 
              edgecolor='black', linewidth=1.5, width=0.5)

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add performance gap annotations
ax.annotate('', xy=(1.25, 0.759), xytext=(1.25, 0.941),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(1.3, (0.759 + 0.941)/2, '24% gap\n(0.182)', 
         fontsize=10, color='red', fontweight='bold', va='center',
         path_effects=[withStroke(linewidth=1.5, foreground='white')])

# Add horizontal reference lines
ax.axhline(y=0.952, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Target Perf')
ax.axhline(y=0.769, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='SOTA Perf')

# Formatting
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_ylim(0.5, 1.0)
# ax.set_title('(a) Performance Gap on Avila Dataset', fontweight='bold', pad=10)
ax.grid(axis='y', alpha=0.3, linestyle=':')
ax.legend(loc='upper left', framealpha=0.9, fontsize=8)

# Add dataset info box
textstr = 'Dataset: Avila\n12-class pattern recognition\nLibrary: 18 operators\nPipeline length: 3'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
ax.text(0.02, 0.74, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('shapley_exp/figure/saves/fig1_intro_a.pdf')
plt.close()

# %% fig b
fig, ax2 = plt.subplots(figsize=(3.5, 2.5))
ax2.axis('off')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
y_ctxpipe = 9
# ctxpipe
ax2.text(0.5, y_ctxpipe + 0.8, 'CtxPipe (Avila Acc: 0.759)', fontweight='bold', fontsize=10, color='#ff7f0e')
positions_ctx = [
    ('Pos 1', 'VT', '#ff6b6b', 'Variance\nThreshold'),
    ('Pos 2', 'PT', '#4ecdc4', 'Power\nTransformer'),
    ('Pos 3', 'RTE', '#95b8e1', 'RandomTrees\nEmbedding')
]
x_start = 1.
box_width = 2.2
box_height=0.8

for i, (pos, op, color, full_name) in enumerate(positions_ctx):
    x = x_start + i * (box_width + 0.6)
    # Draw box
    box = FancyBboxPatch((x, y_ctxpipe - 0.3), box_width, box_height,
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor=color, 
                          linewidth=1.5, alpha=0.7)
    ax2.add_patch(box)
    # Add operator name
    ax2.text(x + box_width/2, y_ctxpipe, op, 
             ha='center', va='center', fontweight='bold', fontsize=11)
    # Add position label
    ax2.text(x + box_width/2, y_ctxpipe - 0.6, pos, 
             ha='center', va='top', fontsize=8, style='italic')
    
    # Add arrow between positions
    if i < len(positions_ctx) - 1:
        ax2.arrow(x + box_width + 0.05, y_ctxpipe, 0.38, 0, 
                 head_width=0.15, head_length=0.08, fc='black', ec='black')
ax2.text(x_start + box_width * 2.5+0.7, y_ctxpipe - 2.5, 
        #  '[!] Early filtering removes\nuseful variance', 
         '[!] PT is sensitive to outliers\nmay cause instability', 
         fontsize=9, color='red', style='italic', ha='center', fontweight='bold')


# Optimal Pipeline
y_optimal = 3.5
ax2.text(0.5, y_optimal - 1.5, 'Optimal (Avila Acc: 0.941)', 
         fontweight='bold', fontsize=10, color='#2ca02c')

# Position boxes for Optimal
positions_opt = [
    ('Pos 1', 'VT', '#ff6b6b', 'Variance\nThreshold'),
    ('Pos 2', 'QT', "#ebc96b", 'Quantile\nTransformer'),
    ('Pos 3', 'RTE', "#95b8e1", 'RandomTrees\nEmbedding'),
]

for i, (pos, op, color, full_name) in enumerate(positions_opt):
    x = x_start + i * (box_width + 0.6)
    # Draw box
    box = FancyBboxPatch((x, y_optimal - 0.3), box_width, box_height,
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor=color, 
                          linewidth=1.5, alpha=0.7)
    ax2.add_patch(box)
    # Add operator name
    ax2.text(x + box_width/2, y_optimal, op, 
             ha='center', va='center', fontweight='bold', fontsize=11)
    # Add position label
    ax2.text(x + box_width/2, y_optimal - 0.6, pos, 
             ha='center', va='top', fontsize=8, style='italic')
    
    # Add arrow between positions
    if i < len(positions_opt) - 1:
        ax2.arrow(x + box_width + 0.05, y_optimal, 0.38, 0, 
                 head_width=0.15, head_length=0.08, fc='black', ec='black')

# Add success annotation
ax2.text(x_start + box_width * 2.5 + 0.4, y_optimal + 1., 
         '[+] QT is more robust to outliers', 
         fontsize=9, color='green', style='italic', ha='center', fontweight='bold')

# Add comparison annotation
ax2.annotate('', xy=(1.2, y_optimal + 0.5), xytext=(1.2, y_ctxpipe - 0.5),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax2.text(3, (y_optimal + y_ctxpipe)/2, 
         'Different operator\n↓\n24% gap', 
         fontsize=9, color='purple', fontweight='bold', 
         ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7))

# Add legend for operators
legend_y = 1.2
ax2.text(0.2, legend_y+0.1, 'Operators:', fontweight='bold', fontsize=9)
ax2.add_patch(Rectangle((0.2, legend_y - 0.5), 0.3, 0.3,
                         facecolor='#ff6b6b', alpha=0.7, edgecolor='black'))
ax2.text(0.8, legend_y-0.35, 'VT = VarianceThreshold', fontsize=8, va='center')
ax2.add_patch(Rectangle((5, legend_y - 0.5), 0.3, 0.3, 
                         facecolor='#95b8e1', alpha=0.7, edgecolor='black'))
ax2.text(5.4, legend_y-0.35, 'RTE = RandomTreesEmbedding', fontsize=8, va='center')
ax2.add_patch(Rectangle((0.2, legend_y - 1), 0.3, 0.3,
                         facecolor='#4ecdc4', alpha=0.7, edgecolor='black'))
ax2.text(0.8, legend_y-0.8, 'PT = PowerTransformer', fontsize=8, va='center')
ax2.add_patch(Rectangle((5, legend_y - 1), 0.3, 0.3, 
                         facecolor='#ebc96b', alpha=0.7, edgecolor='black'))
ax2.text(5.4, legend_y-0.8, 'QT = QuantileTransformer', fontsize=8, va='center')


plt.tight_layout()
plt.savefig('shapley_exp/figure/saves/fig1_intro_b.pdf')
plt.close()
# %%
fig, ax3 = plt.subplots(figsize=(3.5, 2.5))
ax3.axis('off')
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
y_ctxpipe = 9
# ctxpipe
ax3.text(0.5, y_ctxpipe + 0.8, 'Pbcseq Acc: 0.717', fontweight='bold', fontsize=10, color='#ff7f0e')
positions_ctx = [
    ('Pos 1', 'OHE', '#ff6b6b', 'One-Hot Encoding'),
    ('Pos 2', 'RBS', '#4ecdc4', 'Robust Scaler'),
    ('Pos 3', 'PF', '#95b8e1', 'Polynomial Features')
]
x_start = 1.
box_width = 2.2
box_height=0.8

for i, (pos, op, color, full_name) in enumerate(positions_ctx):
    x = x_start + i * (box_width + 0.6)
    # Draw box
    box = FancyBboxPatch((x, y_ctxpipe - 0.3), box_width, box_height,
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor=color, 
                          linewidth=1.5, alpha=0.7)
    ax3.add_patch(box)
    # Add operator name
    ax3.text(x + box_width/2, y_ctxpipe, op, 
             ha='center', va='center', fontweight='bold', fontsize=11)
    # Add position label
    ax3.text(x + box_width/2, y_ctxpipe - 0.6, pos, 
             ha='center', va='top', fontsize=8, style='italic')
    
    # Add arrow between positions
    if i < len(positions_ctx) - 1:
        ax3.arrow(x + box_width + 0.05, y_ctxpipe, 0.38, 0, 
                 head_width=0.15, head_length=0.08, fc='black', ec='black')
ax3.text(x_start + box_width * 2.5+0.4, y_ctxpipe - 2.5, 
        #  '[!] Early filtering removes\nuseful variance', 
         '[!] Early scaling loses\nsome information', 
         fontsize=9, color='red', style='italic', ha='center', fontweight='bold')


# Optimal Pipeline
y_optimal = 3.5
ax3.text(0.5, y_optimal - 1.5, 'Pbcseq Acc: 0.743', 
         fontweight='bold', fontsize=10, color='#2ca02c')

# Position boxes for Optimal
positions_opt = [
    ('Pos 1', 'OHE', '#ff6b6b', 'One-Hot Encoding'),
    ('Pos 2', 'PF', '#95b8e1', 'Polynomial Features'),
    ('Pos 3', 'RBS', '#4ecdc4', 'Robust Scaler'),
]

for i, (pos, op, color, full_name) in enumerate(positions_opt):
    x = x_start + i * (box_width + 0.6)
    # Draw box
    box = FancyBboxPatch((x, y_optimal - 0.3), box_width, box_height,
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor=color, 
                          linewidth=1.5, alpha=0.7)
    ax3.add_patch(box)
    # Add operator name
    ax3.text(x + box_width/2, y_optimal, op, 
             ha='center', va='center', fontweight='bold', fontsize=11)
    # Add position label
    ax3.text(x + box_width/2, y_optimal - 0.6, pos, 
             ha='center', va='top', fontsize=8, style='italic')
    
    # Add arrow between positions
    if i < len(positions_opt) - 1:
        ax3.arrow(x + box_width + 0.05, y_optimal, 0.38, 0, 
                 head_width=0.15, head_length=0.08, fc='black', ec='black')

# Add success annotation
ax3.text(x_start + box_width * 2.5 + 0.4, y_optimal + 1., 
         '[+] Later scaling helps\nmodel generalization', 
         fontsize=9, color='green', style='italic', ha='center', fontweight='bold')

# Add comparison annotation
ax3.annotate('', xy=(1.2, y_optimal + 0.5), xytext=(1.2, y_ctxpipe - 0.5),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax3.text(3, (y_optimal + y_ctxpipe)/2, 
         'Different order\n↓\n3.6% gap', 
         fontsize=9, color='purple', fontweight='bold', 
         ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7))

# Add legend for operators
legend_y = 1.2
ax3.text(0.2, legend_y+0.1, 'Operators:', fontweight='bold', fontsize=9)
ax3.add_patch(Rectangle((0.2, legend_y - 0.5), 0.3, 0.3,
                         facecolor='#ff6b6b', alpha=0.7, edgecolor='black'))
ax3.text(0.8, legend_y-0.35, 'OHE = One-Hot Encoding', fontsize=8, va='center')
ax3.add_patch(Rectangle((5, legend_y - 0.5), 0.3, 0.3, 
                         facecolor='#95b8e1', alpha=0.7, edgecolor='black'))
ax3.text(5.4, legend_y-0.35, 'PF = Polynomial Features', fontsize=8, va='center')
ax3.add_patch(Rectangle((0.2, legend_y - 1), 0.3, 0.3,
                         facecolor='#4ecdc4', alpha=0.7, edgecolor='black'))
ax3.text(0.8, legend_y-0.8, 'RBS = Robust Scaler', fontsize=8, va='center')

plt.tight_layout()
plt.savefig('shapley_exp/figure/saves/fig1_intro_c.pdf')
plt.close()