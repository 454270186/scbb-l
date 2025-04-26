import numpy as np
import matplotlib.pyplot as plt

# 原始数据放大100倍
prompt_chain_lengths = np.arange(3, 16)
pcdo = np.array([0.18, 0.23, 0.3, 0.34, 0.389,  0.44,  0.49, 0.56,   0.60, 0.65,  0.70, 0.77, 0.805]) * 100
scbb_l = np.array([0.2, 0.25, 0.33, 0.36, 0.41, 0.467,  0.52, 0.595,  0.63, 0.69,  0.75, 0.80,  0.96]) * 100
pcdf = np.array([0.22, 0.28, 0.414, 0.49, 0.59, 0.65,  0.73, 0.79,   0.82, 0.92,  1.05,  1.10, 1.195]) * 100
lofd = np.array([0.24, 0.32, 0.4, 0.52, 0.61,   0.72,  0.82, 0.97,   1.16, 1.2,   1.35, 1.50, 1.62]) * 100

# 设置柱子的宽度
bar_width = 0.2

# 设置每组柱子的x坐标
x = np.arange(len(prompt_chain_lengths))

# 创建画布
fig, ax = plt.subplots(figsize=(10, 5))

# 绘制柱子
bars_pcdo = ax.bar(x - 1.5 * bar_width, pcdo, width=bar_width, label='PCDO (Reference)', color='white', edgecolor='tab:blue', hatch='//')
bars_scbb_l = ax.bar(x - 0.5 * bar_width, scbb_l, width=bar_width, label='SCBB-L', color='mediumseagreen')
bars_pcdf = ax.bar(x + 0.5 * bar_width, pcdf, width=bar_width, label='PCDF', color='indianred')
bars_lofd = ax.bar(x + 1.5 * bar_width, lofd, width=bar_width, label='LOFD', color='mediumpurple')

# 添加标题和标签
ax.set_title('Comparison of Average Latency Across Prompt Chain Lengths')
ax.set_xlabel('Prompt Chain Length (Number of Prompts)')
ax.set_ylabel('Average Latency (ms)')

# 设置x轴刻度
ax.set_xticks(x)
ax.set_xticklabels(prompt_chain_lengths)

# 添加图例
ax.legend()

# 添加网格
ax.grid(axis='y', linestyle='--', linewidth=0.5)

# 设置y轴范围（这里也扩大到大概170）
ax.set_ylim(0, 170)

# 紧凑布局
plt.tight_layout()

plt.savefig('ave_latency.png', dpi=300)

# 显示图表
plt.show()
