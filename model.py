# model.py - 完整版（严格按论文公式）
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE, DATASET_CONFIG, TRAIN_CONFIG

class ConvEncoder(nn.Module):
    """卷积编码器（适配不同数据集）"""
    def __init__(self, dataset_name):
        super().__init__()
        self.in_channels = DATASET_CONFIG[dataset_name]["in_channels"]
        self.feature_dim = TRAIN_CONFIG["feature_dim"]
        img_size = DATASET_CONFIG[dataset_name]["img_size"]

        # 不同数据集的编码器结构（参考论文3.2节）
        if img_size <= 32:  # MNIST/Fashion-MNIST/CIFAR-10
            self.conv_layers = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            # 计算flatten后的维度
            output_size = img_size
            for _ in range(3):
                output_size = (output_size + 2*1 - 3) // 2 + 1
            flatten_dim = 128 * output_size * output_size
        else:  # STL-10（96x96）
            self.conv_layers = nn.Sequential(
                nn.Conv2d(self.in_channels, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            flatten_dim = 512 * (img_size//16) * (img_size//16)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, self.feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        conv_feat = self.conv_layers(x)
        feat = self.fc(conv_feat)
        return feat

class SelfExpressionLayer(nn.Module):
    """可训练的自表达层：C就是权重矩阵，严格按论文维度"""
    def __init__(self, batch_size=256, alpha=0.1):
        super().__init__()
        self.alpha = alpha  # 正则化系数
        self.batch_size = batch_size
        # C矩阵初始化为可学习参数 - (batch_size, batch_size) = (n×n)
        self.C = nn.Parameter(torch.eye(batch_size, device=DEVICE))
        print(f"自表达层初始化: C矩阵大小 {batch_size}x{batch_size}, alpha={alpha}")
    
    def forward(self, Z_ori, Z_aug):
        """
        严格按论文公式和维度实现：
        L_SE = 1/2 * ||[Z; Z'] - [Z; Z']C||_F^2 + α/2 * ||C||_F^2
        其中：
        Z, Z' ∈ R^(d×n) [论文维度]
        [Z; Z'] ∈ R^(2d×n) [竖直拼接]
        C ∈ R^(n×n)
        Args:
            Z_ori: (batch_size, feature_dim) 即 (n×d) [我们的维度]
            Z_aug: (batch_size, feature_dim) 即 (n×d) [我们的维度]
        Returns:
            L_SE: 自表达损失
            A_SE: 亲和力矩阵 (2*batch_size, 2*batch_size)
            C_matrix: 系数矩阵 (batch_size, batch_size)
        """
        batch_size = Z_ori.shape[0]
        feature_dim = Z_ori.shape[1]
        # 断言确保batch_size一致
        assert batch_size == self.batch_size, \
            f"输入batch_size={batch_size}，但自表达层期望batch_size={self.batch_size}"
        # 1. 获取可学习的C矩阵
        C_matrix = self.C  # (batch_size, batch_size) = (n×n)
        # 3. 转置到论文维度：Z^T ∈ R^(d×n)
        Z_ori_t = Z_ori.T  # (feature_dim, batch_size) = (d×n)
        Z_aug_t = Z_aug.T  # (feature_dim, batch_size) = (d×n)
        # 4. 竖直拼接：[Z; Z'] ∈ R^(2d×n)
        Z_cat_t = torch.cat([Z_ori_t, Z_aug_t], dim=0)  # (2*feature_dim, batch_size) = (2d×n)
        
        # 5. 计算重构：[Z; Z']C ∈ R^(2d×n)
        Z_recon_t = torch.matmul(Z_cat_t, C_matrix)  # (2d×n) @ (n×n) → (2d×n)
        
        # 6. Frobenius范数平方计算损失
        diff = Z_cat_t - Z_recon_t  # (2d×n)
        recon_loss = 0.5 * torch.norm(diff, p='fro')**2
        
        # 7. Frobenius范数正则化
        reg_loss = 0.5 * self.alpha * torch.norm(C_matrix, p='fro')**2
        
        # 8. 总损失
        L_SE = recon_loss + reg_loss
        
        # 9. 构建亲和力矩阵（基于|C| + |C|^T）/ 2
        C_abs = torch.abs(C_matrix)
        A_sym = (C_abs + C_abs.t()) / 2
        
        # 10. 归一化处理
        eye = torch.eye(batch_size, device=C_matrix.device)
        off_diag_mask = 1 - eye
        
        # 每行的最大非对角线值
        row_max = torch.max(A_sym * off_diag_mask, dim=1, keepdim=True)[0]
        row_max = torch.clamp(row_max, min=1e-8)
        
        # 归一化并设置对角线为1
        A = (A_sym * off_diag_mask) / row_max + eye
        
        # 11. 扩展到2n×2n（匹配增强样本）
        A_SE = torch.block_diag(A, A)
        
        return L_SE, A_SE, C_matrix

class InductiveClusteringHead(nn.Module):
    """归纳子空间聚类头：生成H_IN和A_IN"""
    def __init__(self, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters
        self.fcn = nn.Sequential(
            nn.Linear(TRAIN_CONFIG["feature_dim"], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_clusters),
            nn.Softmax(dim=1)
        )
        # 改进最后一层初始化，让输出更有区分度
        # 使用更大的初始化值，使logits差异更大
    def forward(self, Z):
        """Z: (2*batch_size, feature_dim) → H_IN: (2*batch_size, num_clusters)"""
        return self.fcn(Z)
    def compute_A_IN(self, H_IN):
        """计算亲和力矩阵A_IN（论文公式7）"""
        H_norm = F.normalize(H_IN, p=2, dim=1)
        A_IN = torch.matmul(H_norm, H_norm.t()) 
        return A_IN

class ProjectionHead(nn.Module):
    """投影头：输出与输入数据集类别数相同的维度"""
    def __init__(self, in_dim, num_clusters):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_clusters)  # 输出num_clusters维度
        )
    def forward(self, x):
        return self.proj(x)

class DISSC(nn.Module):
    """完整DISSC模型（严格按论文实现）"""
    def __init__(self, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self.num_clusters = DATASET_CONFIG[dataset_name]["num_clusters"]
        
        # 获取batch_size配置
        batch_size = DATASET_CONFIG[dataset_name]["batch_size"]
        
        # 核心模块
        self.encoder = ConvEncoder(dataset_name).to(DEVICE)
        self.self_expression = SelfExpressionLayer(
            batch_size=batch_size,
            alpha=TRAIN_CONFIG["lambda_SE"]  # 注意：这里使用lambda_SE作为alpha
        ).to(DEVICE)
        self.inductive_head = InductiveClusteringHead(self.num_clusters).to(DEVICE)
        self.proj_se = ProjectionHead(TRAIN_CONFIG["feature_dim"], self.num_clusters).to(DEVICE)  # 自表达模块投影头
        print(f"DISSC模型初始化完成:")
        print(f"  - 数据集: {dataset_name}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - 类别数: {self.num_clusters}")
        print(f"  - 特征维度: {TRAIN_CONFIG['feature_dim']}")
    
    def forward(self, x_ori, x_aug):
        batch_size = x_ori.shape[0]
        
        # 1. 深度特征提取
        Z_ori = self.encoder(x_ori)  # (batch_size, feature_dim)
        Z_aug = self.encoder(x_aug)  # (batch_size, feature_dim)
        
        # 2. 自表达子空间学习
        L_SE, A_SE, C = self.self_expression(Z_ori, Z_aug)
        # 3. 计算 ZC 和 Z'C 作为 proj_se 的输入
        # 首先转置到论文维度
        Z_ori_t = Z_ori.T  # (d, n)
        Z_aug_t = Z_aug.T  # (d, n)
        # 计算 ZC 和 Z'C
        ZC_t = torch.matmul(Z_ori_t, C)  # (d, n) @ (n, n) → (d, n)
        Z_prime_C_t = torch.matmul(Z_aug_t, C)  # (d, n) @ (n, n) → (d, n)
        # 转回我们的维度
        ZC = ZC_t.T  # (n, d)
        Z_prime_C = Z_prime_C_t.T  # (n, d)
        # 拼接得到 [ZC; Z'C]
        ZC_cat = torch.cat([ZC, Z_prime_C], dim=0)  # (2*n, d)
        # 4. 计算 H_SE
        H_SE = self.proj_se(ZC_cat)  # (2*batch_size, num_clusters)
        # 5. 归纳子空间聚类（使用原始特征Z，不是ZC）
        Z = torch.cat([Z_ori, Z_aug], dim=0)  # (2*batch_size, feature_dim)
        H_IN = self.inductive_head(Z)  # (2*batch_size, num_clusters)
        A_IN = self.inductive_head.compute_A_IN(H_IN)
        return {
            "L_SE": L_SE,
            "A_SE": A_SE,
            "A_IN": A_IN,
            "H_SE": H_SE,
            "H_IN": H_IN,
            "C": C,
            "Z_ori": Z_ori,
            "Z_aug": Z_aug,
            "ZC": ZC,
            "Z_prime_C": Z_prime_C
        }