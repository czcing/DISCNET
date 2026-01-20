import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

def cluster_accuracy(y_true, y_pred):
    """
    计算聚类准确率（使用Kuhn-Munkres算法匹配真实标签与预测标签）
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_true.size == y_pred.size

    # 构建混淆矩阵
    max_label = max(y_true.max(), y_pred.max()) + 1
    confusion_matrix = np.zeros((max_label, max_label), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1

    # Kuhn-Munkres算法找最优匹配
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    accuracy = 0
    for r, c in zip(row_ind, col_ind):
        accuracy += confusion_matrix[r, c]
    accuracy /= y_true.size
    return accuracy * 100  # 转为百分比

def evaluate_clustering(y_true, y_pred):
    """
    计算ACC、NMI、ARI
    Returns:
        dict: 包含三个指标的百分比值
    """
    acc = cluster_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred) * 100
    ari = adjusted_rand_score(y_true, y_pred) * 100
    return {
        "ACC": round(acc, 2),
        "NMI": round(nmi, 2),
        "ARI": round(ari, 2)
    }