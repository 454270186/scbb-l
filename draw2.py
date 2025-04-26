import matplotlib.pyplot as plt
import numpy as np

# Prompt chain lengths from 3 to 15
labels = list(range(3, 16))
x = np.arange(len(labels))
width = 0.2

# New total latency data (in ms)
scbb_l_total = np.array([0.2, 0.25, 0.33, 0.36, 0.41, 0.467, 0.52, 0.595, 0.63, 0.69, 0.75, 0.80, 0.96]) * 100
pcdf_total   = np.array([0.22, 0.28, 0.414, 0.49, 0.59, 0.65, 0.73, 0.79, 0.82, 0.92, 1.05, 1.10, 1.195]) * 100
lofd_total   = np.array([0.24, 0.32, 0.4, 0.52, 0.61, 0.72, 0.82, 0.97, 1.16, 1.2, 1.35, 1.50, 1.62]) * 100

# Set the ratio for splitting
comp_ratio = 0.12
trans_ratio = 0.52
verif_ratio = 0.36

# Split the total latency into components
def split_total_latency(total_latency):
    comp = total_latency * comp_ratio
    trans = total_latency * trans_ratio
    verif = total_latency * verif_ratio
    return comp, trans, verif

comp_latency = {}
trans_latency = {}
verif_latency = {}

comp_latency['SCBB-L'], trans_latency['SCBB-L'], verif_latency['SCBB-L'] = split_total_latency(scbb_l_total)
comp_latency['PCDF'], trans_latency['PCDF'], verif_latency['PCDF'] = split_total_latency(pcdf_total)
comp_latency['LOFD'], trans_latency['LOFD'], verif_latency['LOFD'] = split_total_latency(lofd_total)

# Style config (keep)
style_config = {
    'SCBB-L': {'color': '#6b8fb4'},  # 浅蓝
    'PCDF': {'color': '#c1855a'},    # 浅橙
    'LOFD': {'color': '#9dac88'}     # 浅绿
}

# Trans latency colors (three slightly different muted taupe tones)
trans_colors = {
    'SCBB-L': '#A99F95',  # taupe灰棕
    'PCDF': '#B7A99A',    # 稍亮灰米色
    'LOFD': '#9C8F86'     # 稍深暖灰
}

bar_positions = {
    'SCBB-L': x - width,
    'PCDF': x,
    'LOFD': x + width,
}

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

for algo in ['SCBB-L', 'PCDF', 'LOFD']:
    comp = comp_latency[algo]
    trans = trans_latency[algo]
    verif = verif_latency[algo]
    color = style_config[algo]['color']
    pos = bar_positions[algo]

    ax.bar(pos, comp, width=width, color=color, label=f'{algo} - Comp Latency')
    ax.bar(pos, trans, bottom=comp, width=width, color=trans_colors[algo], label=f'{algo} - Trans Latency')
    ax.bar(pos, verif, bottom=comp + trans, width=width, color='white', hatch='...', edgecolor='green',
           label=f'{algo} - Verif Latency')

# Style
ax.set_xlabel('Prompt Chain Length', fontsize=12, fontweight='bold')
ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
ax.set_title('Latency Composition by Prompt Chain Length', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(axis='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
ax.legend(ncol=3, fontsize='small', loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

plt.savefig('latency_composition.png', dpi=300)

plt.show()
