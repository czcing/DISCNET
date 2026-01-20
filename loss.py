# loss.py (完整修正版)
import torch
import torch.nn.functional as F
from config import DEVICE
def nonlocal_contrastive_self_distillation_loss(A_SE, A_IN, H_SE, H_IN, xi, tau):
    """
    修正的损失函数实现
    """
    batch_size_2n = A_SE.shape[0]
    device = A_SE.device
    
    # 1. 构建正负样本掩码
    eye_matrix = torch.eye(batch_size_2n, device=device)
    P_mask = (A_SE >= xi).float()   # 排除自身
    N_mask = (A_IN <= 1 - xi).float()  # 排除自身
    
    # 2. 计算相似度矩阵
    def compute_similarity(H):
        H_norm = F.normalize(H, p=2, dim=1)
        return torch.matmul(H_norm, H_norm.t()) / tau
    
    sim_SE = compute_similarity(H_SE)
    sim_IN = compute_similarity(H_IN)
    
    # 3. 计算完整的概率矩阵（包含自身）
    def compute_full_probability_matrix(sim):
        # 使用稳定的softmax计算
        # 减去最大值以提高数值稳定性
        sim_max = sim.max(dim=1, keepdim=True)[0]
        sim_stable = sim - sim_max
        exp_sim = torch.exp(sim_stable)
        # 包括自身（分母中包括i=j的情况）
        denominator = torch.sum(exp_sim, dim=1, keepdim=True)
        return exp_sim / denominator
    
    # 计算完整的概率矩阵
    P_SE_full = compute_full_probability_matrix(sim_SE)  # 包含P_SE(i,i)
    P_IN_full = compute_full_probability_matrix(sim_IN)  # 包含P_IN(i,i)
    
    # 4. 计算联合概率（严格按论文公式）
    def compute_log_joint_probability(P_full, P_mask, N_mask):
        batch_size = P_full.shape[0]
        log_probs = []
        
        for i in range(batch_size):
            # 获取当前样本的概率分布
            p_i = P_full[i]  # (2n,)
            
            # 正样本部分：log Π_{j∈P(i)} P(i,j)
            pos_mask_i = P_mask[i]  # (2n,)
            # 注意：P_mask已经排除了自身（i=j的情况）
            pos_log_sum = torch.sum(torch.log(p_i + 1e-8) * pos_mask_i)
            
            # 负样本部分：log Π_{j∈N(i)} (1 - P(i,j))
            neg_mask_i = N_mask[i]  # (2n,)
            # 注意：N_mask已经排除了自身（i=j的情况）
            neg_log_sum = torch.sum(torch.log(1 - p_i + 1e-8) * neg_mask_i)
            
            log_probs.append(pos_log_sum + neg_log_sum)
        
        return torch.stack(log_probs)
    
    # 5. 计算对数联合概率
    log_prob_SE = compute_log_joint_probability(P_SE_full, P_mask, N_mask)
    log_prob_IN = compute_log_joint_probability(P_IN_full, P_mask, N_mask)
    
    # 6. 计算总损失（论文公式11）
    loss_SE = -torch.mean(log_prob_SE)
    loss_IN = -torch.mean(log_prob_IN)
    total_loss = loss_SE + loss_IN
    
    return total_loss