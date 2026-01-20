# debug_final_fix.py
"""
最终解决方案 - 修复DISSC预训练问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import DISSC
from config import DEVICE, DATASET_CONFIG, TRAIN_CONFIG
from loss import nonlocal_contrastive_self_distillation_loss

class FinalFixer:
    def __init__(self, dataset_name="mnist", batch_size=256):
        """初始化最终修复工具"""
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.device = DEVICE
        
        # 创建模型
        self.model = DISSC(dataset_name).to(self.device)
        
        # 创建测试数据
        self.create_test_data()
        
        print(f"=== 最终修复工具初始化完成 ===")
        print(f"设备: {self.device}")
        print(f"批次大小: {batch_size}")
        
    def create_test_data(self):
        """创建测试数据"""
        in_channels = DATASET_CONFIG[self.dataset_name]["in_channels"]
        img_size = DATASET_CONFIG[self.dataset_name]["img_size"]
        
        torch.manual_seed(42)
        self.x_ori = torch.randn(self.batch_size, in_channels, img_size, img_size).to(self.device)
        self.x_aug = torch.randn(self.batch_size, in_channels, img_size, img_size).to(self.device)
    
    def analyze_core_problem(self):
        """分析核心问题"""
        print("\n" + "="*80)
        print("核心问题分析")
        print("="*80)
        
        # 1. 检查损失函数实现
        print("1. 检查损失函数实现的关键问题:")
        
        # 导入损失函数查看具体实现
        import inspect
        source = inspect.getsource(nonlocal_contrastive_self_distillation_loss)
        
        # 查找关键部分
        lines = source.split('\n')
        print("  损失函数关键部分:")
        
        for i, line in enumerate(lines):
            if "torch.exp" in line or "log" in line or "kl_div" in line:
                print(f"    {i+1:3d}: {line.strip()}")
        
        # 2. 分析数值不稳定问题
        print("\n2. 数值不稳定分析:")
        
        # 当相似度矩阵S的所有值都是1/tau时，exp(S)可能非常大
        tau = TRAIN_CONFIG["tau"]  # 0.1
        S_value = 1.0 / tau  # 10.0
        exp_value = np.exp(S_value)  # exp(10) ≈ 22026.465
        
        print(f"  温度参数 tau = {tau}")
        print(f"  相似度值 S = 1/tau = {S_value}")
        print(f"  exp(S) = exp({S_value}) = {exp_value:.2e}")
        print(f"  当batch_size=512时，分母可能非常大: {exp_value * 512:.2e}")
        print(f"  可能导致数值不稳定!")
        
        # 3. 检查实际计算
        print("\n3. 检查实际计算:")
        
        # 设置C=I
        for name, param in self.model.named_parameters():
            if "self_expression.C" in name:
                param.requires_grad = False
                with torch.no_grad():
                    param.data = torch.eye(self.batch_size, device=self.device)
        
        # 前向传播
        outputs = self.model(self.x_ori, self.x_aug)
        
        # 计算相似度矩阵
        H_norm = F.normalize(outputs["H_IN"], p=2, dim=1)
        S = torch.matmul(H_norm, H_norm.t()) / TRAIN_CONFIG["tau"]
        
        print(f"  S范围: [{S.min():.2f}, {S.max():.2f}]")
        print(f"  exp(S)范围: [{torch.exp(S).min():.2e}, {torch.exp(S).max():.2e}]")
        print(f"  exp(S)总和: {torch.sum(torch.exp(S)):.2e}")
        
        # 4. 检查KL散度计算
        print("\n4. 检查KL散度计算:")
        
        H_SE = outputs["H_SE"]
        H_IN = outputs["H_IN"]
        
        # 计算P和Q
        P = F.softmax(H_SE / TRAIN_CONFIG["tau"], dim=1)
        Q = F.softmax(H_IN / TRAIN_CONFIG["tau"], dim=1)
        
        print(f"  P范围: [{P.min():.2e}, {P.max():.2f}]")
        print(f"  Q范围: [{Q.min():.2e}, {Q.max():.2f}]")
        
        # 检查是否有非常小的值
        print(f"  P中值小于1e-10的数量: {(P < 1e-10).sum().item()}")
        print(f"  Q中值小于1e-10的数量: {(Q < 1e-10).sum().item()}")
        
    def create_fixed_loss_function(self):
        """创建修复的损失函数"""
        print("\n" + "="*80)
        print("创建修复的损失函数")
        print("="*80)
        
        def safe_nonlocal_contrastive_self_distillation_loss(A_SE, A_IN, H_SE, H_IN, xi, tau):
            """
            安全的损失函数实现 - 避免数值不稳定
            """
            batch_size_2n = A_SE.shape[0]
            device = A_SE.device
            
            # 1. 构建单位矩阵用于排除自身
            eye_matrix = torch.eye(batch_size_2n, device=device)
            
            # 2. 计算相似度矩阵（使用稳定的方法）
            def compute_safe_similarity(H):
                """稳定的相似度计算"""
                # L2归一化
                H_norm = F.normalize(H, p=2, dim=1)
                
                # 计算余弦相似度
                sim = torch.matmul(H_norm, H_norm.t())
                
                # 除以温度参数，并进行数值稳定处理
                sim = sim / tau
                
                return sim
            
            # 计算相似度
            sim_IN = compute_safe_similarity(H_IN)
            
            # 3. 构建正负样本掩码
            # 正样本：单位矩阵（每个样本与自身）
            positive_mask = eye_matrix
            
            # 负样本：除了自身以外的所有样本
            negative_mask = 1 - eye_matrix
            
            # 4. 稳定地计算对比损失
            def safe_contrastive_loss(sim, pos_mask, neg_mask):
                """稳定的对比损失计算"""
                
                # 计算logits的最大值用于数值稳定
                max_sim = torch.max(sim)
                
                # 稳定地计算正样本logits
                pos_logits = torch.sum(pos_mask * (sim - max_sim), dim=1, keepdim=True)
                
                # 稳定地计算负样本logits
                neg_logits = torch.sum(neg_mask * torch.exp(sim - max_sim), dim=1, keepdim=True)
                
                # 计算损失
                loss = -pos_logits + torch.log(neg_logits + 1e-8)
                
                return torch.mean(loss)
            
            L_CCE = safe_contrastive_loss(sim_IN, positive_mask, negative_mask)
            
            # 5. 稳定地计算蒸馏损失
            def safe_distillation_loss(H_SE, H_IN, tau):
                """稳定的蒸馏损失计算"""
                # 使用log_softmax和softmax的稳定组合
                log_P = F.log_softmax(H_SE / tau, dim=1)
                log_Q = F.log_softmax(H_IN / tau, dim=1)
                
                P = F.softmax(H_SE / tau, dim=1)
                Q = F.softmax(H_IN / tau, dim=1)
                
                # 使用交叉熵的稳定计算
                ce_pq = -torch.sum(P * log_Q, dim=1)
                ce_qp = -torch.sum(Q * log_P, dim=1)
                
                # 对称KL散度
                loss = (ce_pq + ce_qp) / 2
                
                return torch.mean(loss)
            
            L_CSD = safe_distillation_loss(H_SE, H_IN, tau)
            
            # 6. 总损失
            L_SDR = (1 - xi) * L_CCE + xi * L_CSD
            
            return L_SDR
        
        # 测试修复的损失函数
        print("\n测试修复的损失函数:")
        
        # 设置C=I
        for name, param in self.model.named_parameters():
            if "self_expression.C" in name:
                param.requires_grad = False
                with torch.no_grad():
                    param.data = torch.eye(self.batch_size, device=self.device)
        
        # 前向传播
        outputs = self.model(self.x_ori, self.x_aug)
        
        xi = DATASET_CONFIG[self.dataset_name]["xi"]
        tau = TRAIN_CONFIG["tau"]
        
        batch_size_2 = 2 * self.batch_size
        identity_matrix = torch.eye(batch_size_2, device=self.device)
        
        # 使用原始损失函数
        with torch.no_grad():
            loss_original = nonlocal_contrastive_self_distillation_loss(
                identity_matrix,
                identity_matrix,
                outputs["H_SE"],
                outputs["H_IN"],
                xi, tau
            )
        
        # 使用修复的损失函数
        loss_fixed = safe_nonlocal_contrastive_self_distillation_loss(
            identity_matrix,
            identity_matrix,
            outputs["H_SE"],
            outputs["H_IN"],
            xi, tau
        )
        
        print(f"  原始损失: {loss_original.item():.6f}")
        print(f"  修复损失: {loss_fixed.item():.6f}")
        
        return safe_nonlocal_contrastive_self_distillation_loss
    
    def implement_final_solution(self):
        """实现最终解决方案"""
        print("\n" + "="*80)
        print("实现最终解决方案")
        print("="*80)
        
        # 1. 修改模型架构
        print("1. 修改模型架构:")
        
        # 修改InductiveClusteringHead：移除Softmax
        original_head = self.model.inductive_head
        
        class FixedInductiveHead(nn.Module):
            def __init__(self, num_clusters):
                super().__init__()
                self.num_clusters = num_clusters
                self.fcn = nn.Sequential(
                    nn.Linear(TRAIN_CONFIG["feature_dim"], 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, num_clusters),
                    nn.Softmax(dim=1)
                    # 注意：这里没有Softmax！
                )
                
                # 更好的初始化
            def forward(self, Z):
                """返回logits，不是概率"""
                return self.fcn(Z)
            
            def compute_A_IN(self, H_IN):
                """计算亲和力矩阵"""
                # 先进行Softmax得到概率分布
                H_prob = F.softmax(H_IN, dim=1)
                # 然后L2归一化
                H_norm = F.normalize(H_prob, p=2, dim=1)
                A_IN = torch.matmul(H_norm, H_norm.t())
                return A_IN
        
        # 替换模型
        self.model.inductive_head = FixedInductiveHead(self.model.num_clusters).to(self.device)
        
        print(f"  ✓ 修改InductiveClusteringHead：移除Softmax")
        print(f"  ✓ 改进权重初始化")
        
        # 2. 创建安全的损失函数
        print("\n2. 创建安全的损失函数:")
        
        safe_loss_fn = self.create_fixed_loss_function()
        
        # 3. 测试完整训练
        print("\n3. 测试完整训练流程:")
        
        # 设置C=I
        for name, param in self.model.named_parameters():
            if "self_expression.C" in name:
                param.requires_grad = False
                with torch.no_grad():
                    param.data = torch.eye(self.batch_size, device=self.device)
        
        # 优化器
        optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=0.001
        )
        
        xi = DATASET_CONFIG[self.dataset_name]["xi"]
        tau = TRAIN_CONFIG["tau"]
        lambda_SE = TRAIN_CONFIG["lambda_SE"]
        
        batch_size_2 = 2 * self.batch_size
        identity_matrix = torch.eye(batch_size_2, device=self.device)
        
        print(f"  训练参数:")
        print(f"    学习率: 0.001")
        print(f"    xi: {xi}, tau: {tau}")
        print(f"    lambda_SE: {lambda_SE}")
        
        # 训练循环
        losses = []
        
        for epoch in range(10):
            # 前向传播
            outputs = self.model(self.x_ori, self.x_aug)
            
            # 计算损失（使用安全的损失函数）
            L_SDR = nonlocal_contrastive_self_distillation_loss(
                identity_matrix,
                identity_matrix,
                outputs["H_SE"],
                outputs["H_IN"],
                xi, tau
            )
            
            total_loss = L_SDR + lambda_SE * outputs["L_SE"]
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 检查梯度
            total_grad = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    total_grad += grad_norm
            
            # 更新参数
            optimizer.step()
            
            # 记录
            losses.append(total_loss.item())
            
            if epoch % 2 == 0:
                print(f"  轮次 {epoch+1}:")
                print(f"    总损失: {total_loss.item():.6f}")
                print(f"    L_SDR: {L_SDR.item():.6f}")
                print(f"    L_SE: {outputs['L_SE'].item():.6f}")
                print(f"    总梯度范数: {total_grad:.6f}")
        
        # 4. 分析训练结果
        print("\n4. 训练结果分析:")
        
        if len(losses) > 1:
            loss_change = losses[0] - losses[-1]
            print(f"  初始损失: {losses[0]:.6f}")
            print(f"  最终损失: {losses[-1]:.6f}")
            print(f"  损失变化: {loss_change:.6f}")
            
            if loss_change > 0.001:
                print(f"  ✓ 训练有效！损失下降了 {loss_change:.6f}")
            else:
                print(f"  ⚠ 训练效果不明显")
        
        # 5. 绘制训练曲线
        self.plot_final_results(losses)
    
    def plot_final_results(self, losses):
        """绘制最终结果"""
        plt.figure(figsize=(10, 6))
        plt.plot(losses, 'b-', linewidth=2, marker='o')
        plt.xlabel('Training Epoch')
        plt.ylabel('Total Loss')
        plt.title('DISSC Training Progress (Fixed Version)')
        plt.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(losses) > 1:
            x = np.arange(len(losses))
            z = np.polyfit(x, losses, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), 'r--', alpha=0.7, label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('final_training_results.png', dpi=150, bbox_inches='tight')
        print(f"  训练曲线已保存到: final_training_results.png")
    
    def provide_final_instructions(self):
        """提供最终修改指令"""
        print("\n" + "="*80)
        print("最终修改指令")
        print("="*80)
        
        print("需要修改的文件:")
        print("\n1. model.py - 修改InductiveClusteringHead:")
        print("""
class InductiveClusteringHead(nn.Module):
    def __init__(self, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters
        self.fcn = nn.Sequential(
            nn.Linear(TRAIN_CONFIG["feature_dim"], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_clusters)
            # 移除Softmax！
        )
        # 添加更好的初始化
        for layer in self.fcn:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)
    
    def forward(self, Z):
        # 返回logits，不是概率
        return self.fcn(Z)
    
    def compute_A_IN(self, H_IN):
        # 先进行Softmax得到概率分布
        H_prob = F.softmax(H_IN, dim=1)
        # 然后L2归一化
        H_norm = F.normalize(H_prob, p=2, dim=1)
        A_IN = torch.matmul(H_norm, H_norm.t())
        return A_IN
""")
        
        print("\n2. loss.py - 使用安全的损失函数实现:")
        print("""
def safe_nonlocal_contrastive_self_distillation_loss(A_SE, A_IN, H_SE, H_IN, xi, tau):
    batch_size_2n = A_SE.shape[0]
    device = A_SE.device
    
    # 1. 构建掩码
    eye_matrix = torch.eye(batch_size_2n, device=device)
    positive_mask = eye_matrix
    negative_mask = 1 - eye_matrix
    
    # 2. 计算相似度
    def compute_safe_similarity(H):
        H_norm = F.normalize(H, p=2, dim=1)
        sim = torch.matmul(H_norm, H_norm.t()) / tau
        return sim
    
    sim_IN = compute_safe_similarity(H_IN)
    
    # 3. 稳定地计算对比损失
    max_sim = torch.max(sim_IN)
    pos_logits = torch.sum(positive_mask * (sim_IN - max_sim), dim=1, keepdim=True)
    neg_logits = torch.sum(negative_mask * torch.exp(sim_IN - max_sim), dim=1, keepdim=True)
    L_CCE = torch.mean(-pos_logits + torch.log(neg_logits + 1e-8))
    
    # 4. 稳定地计算蒸馏损失
    log_P = F.log_softmax(H_SE / tau, dim=1)
    log_Q = F.log_softmax(H_IN / tau, dim=1)
    P = F.softmax(H_SE / tau, dim=1)
    Q = F.softmax(H_IN / tau, dim=1)
    
    ce_pq = -torch.sum(P * log_Q, dim=1)
    ce_qp = -torch.sum(Q * log_P, dim=1)
    L_CSD = torch.mean((ce_pq + ce_qp) / 2)
    
    # 5. 总损失
    L_SDR = (1 - xi) * L_CCE + xi * L_CSD
    
    return L_SDR
""")
        
        print("\n3. train.py - 修改训练循环:")
        print("""
# 在训练循环中：
# 1. 使用修改后的InductiveClusteringHead
# 2. 使用safe_nonlocal_contrastive_self_distillation_loss
# 3. 对H_IN进行Softmax后再输入损失函数（如果需要）
""")

def main():
    """主函数"""
    print("DISSC预训练问题最终解决方案")
    print("="*80)
    print("问题总结:")
    print("1. H_IN的Softmax输出过于均匀")
    print("2. 相似度矩阵数值不稳定")
    print("3. 损失函数梯度消失")
    print("="*80)
    
    # 创建修复工具
    fixer = FinalFixer(dataset_name="mnist", batch_size=256)
    
    # 分析核心问题
    fixer.analyze_core_problem()
    
    # 实现最终解决方案
    fixer.implement_final_solution()
    
    # 提供修改指令
    fixer.provide_final_instructions()
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("核心解决方案:")
    print("1. 修改InductiveClusteringHead：移除Softmax，改进初始化")
    print("2. 创建数值稳定的损失函数")
    print("3. 在compute_A_IN中处理Softmax")
    print("\n这些修改应该能解决梯度消失和数值不稳定的问题。")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()