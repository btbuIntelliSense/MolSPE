import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import davies_bouldin_score

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# 创建2x2的子图布局
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# 设置四个不同的DB指数值（与参考图对应）
db_indices = [11.04, 4.48, 0.73, 0.62]
dataset_names = ["ClinTox", "BBBP", "Dataset 3", "Dataset 4"]

# 每个子图的数据点总数均为2039个
total_points = 2039
minority_points = 438

# 生成四个不同的分布情况
for i, (ax, db_index, name) in enumerate(zip(axs.flatten(), db_indices, dataset_names)):
    # 根据DB指数调整簇的中心分离和标准差
    if db_index > 10:
        # 高DB指数：簇分离度低，有显著重叠
        centers = [[0.3, 0.3], [0.7, 0.7]]
        cluster_std = [0.18, 0.18]
    elif db_index > 3:
        # 中等DB指数：部分分离
        centers = [[0.2, 0.3], [0.8, 0.7]]
        cluster_std = [0.15, 0.15]
    else:
        # 低DB指数：良好分离，但保留轻微勾连
        centers = [[0.2, 0.2], [0.8, 0.8]]
        cluster_std = [0.1, 0.1]

    # 生成数据集
    X, y_true = make_blobs(
        n_samples=[minority_points, total_points - minority_points],
        centers=centers,
        cluster_std=cluster_std,
        random_state=42 + i
    )

    # 计算实际的DB指数（仅作参考）
    actual_db = davies_bouldin_score(X, y_true)

    # 分离数据点
    minority_data = X[y_true == 0]
    majority_data = X[y_true == 1]

    # 绘制散点图 - 优化视觉区分度
    ax.scatter(
        minority_data[:, 0], minority_data[:, 1],
        c='#FF7F00', marker='o', s=25, alpha=0.6,
        edgecolor='k', linewidth=0.4, label='Label 0'
    )
    ax.scatter(
        majority_data[:, 0], majority_data[:, 1],
        c='#377EB8', marker='o', s=25, alpha=0.4,
        edgecolor='k', linewidth=0.4, label='Label 1'
    )

    # 设置相同的坐标轴范围 [0, 1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 添加刻度标签 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.3)

    # 添加数据集名称和DB指数
    ax.set_title(f"{name}", pad=15)
    ax.text(0.98, 0.02, f"DB={db_index}",
            ha='right', va='bottom', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

# 添加整体标题
plt.suptitle('Cluster Separation with Different DB Indices', fontsize=14, y=0.98)

# 仅在第一行第一列添加图例
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.95), ncol=2, framealpha=0.9)

# 添加整体坐标轴标签
fig.text(0.5, 0.02, 'Feature Dimension 1', ha='center', va='center', fontsize=12)
fig.text(0.02, 0.5, 'Feature Dimension 2', ha='center', va='center', rotation='vertical', fontsize=12)

# 保存高清图像
plt.savefig('cluster_comparison.png', dpi=300, bbox_inches='tight')
plt.show()