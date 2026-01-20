# utils.py
import numpy as np
import torch
import matplotlib.pyplot as plt

def visualize_affinity_matrix(A, title="Affinity Matrix"):
    """可视化亲和力矩阵"""
    plt.figure(figsize=(8, 6))
    plt.imshow(A.cpu().detach().numpy() if torch.is_tensor(A) else A, 
               cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.show()

def visualize_clusters(features, labels, title="Cluster Visualization"):
    """可视化聚类结果（t-SNE降维）"""
    from sklearn.manifold import TSNE
    
    features_np = features.cpu().detach().numpy() if torch.is_tensor(features) else features
    labels_np = labels.cpu().detach().numpy() if torch.is_tensor(labels) else labels
    
    # t-SNE降维到2D
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_np)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=labels_np, cmap='tab20', s=10, alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()

def check_model_params(model):
    """检查模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"不可训练参数: {total_params - trainable_params:,}")
    
    # 打印各模块参数
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {module_params:,}")

def set_seed(seed=42):
    """设置随机种子"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False