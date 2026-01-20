# train.py（使用data_loader.py的划分）
import torch
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import time
from config import DEVICE, DATASET_CONFIG, TRAIN_CONFIG
from data_loader import get_dataloader  # 直接使用get_dataloader
from model import DISSC
from loss import nonlocal_contrastive_self_distillation_loss
from metrics import evaluate_clustering
def train_dissc(dataset_name):
    """
    完全按照论文Algorithm 1实现，使用data_loader.py的固定划分
    """
    print(f"=== DISSC训练开始（使用data_loader.py的划分）===")
    # 1. 使用data_loader.py的固定划分
    print(f"\n加载数据集: {dataset_name}")
    train_loader, val_loader, test_loader, num_clusters = get_dataloader(dataset_name)
    print(f"数据集划分:")
    print(f"  训练集: {len(train_loader.dataset)} 样本")
    print(f"  验证集: {len(val_loader.dataset)} 样本")
    print(f"  测试集: {len(test_loader.dataset)} 样本")
    print(f"  类别数: {num_clusters}")
    # 获取配置参数
    xi = DATASET_CONFIG[dataset_name]["xi"]
    lambda_SE = TRAIN_CONFIG["lambda_SE"]
    tau = TRAIN_CONFIG["tau"]
    batch_size = DATASET_CONFIG[dataset_name]["batch_size"]
    save_path = TRAIN_CONFIG["save_path"]
    
    # 计算总迭代次数 M
    M = TRAIN_CONFIG["finetune_epochs"] * len(train_loader)
    
    print(f"\n训练参数:")
    print(f"  批次大小: {batch_size}")
    print(f"  训练批次/epoch: {len(train_loader)}")
    print(f"  总迭代次数 M: {M}")
    print(f"  阈值 ξ: {xi}, 温度 τ: {tau}, λ: {lambda_SE}")
    
    # 2. 初始化模型
    model = DISSC(dataset_name)
    
    # 3. 为不同参数创建优化器
    encoder_params = []
    C_param = None
    
    for name, param in model.named_parameters():
        if "self_expression.C" in name:
            C_param = param
        else:
            encoder_params.append(param)
    
    optimizer_encoder = optim.Adam(encoder_params, lr=TRAIN_CONFIG["lr"], weight_decay=0)
    optimizer_C = optim.Adam([C_param], lr=TRAIN_CONFIG["lr"], weight_decay=0)
    
    # ============================ 预训练阶段 ============================
    # Algorithm 1第3行: Pre-train with fixed matrices
    print("\n" + "="*60)
    print("预训练阶段: 固定 C=I, A_SE=I, A_IN=I")
    print(f"训练 {TRAIN_CONFIG['pretrain_epochs']} 个epoch")
    print(f"冻结自表达层C，只训练编码器参数")
    print("="*60)
    # 冻结自表达层参数C
    for name, param in model.named_parameters():
        if "self_expression.C" in name:
            param.requires_grad = False
            print(f"冻结参数: {name}")
    # 只训练非冻结参数
    pretrain_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_pretrain = optim.Adam(pretrain_params, lr=TRAIN_CONFIG["lr"], weight_decay=0)
    # 预训练循环
    for epoch in range(TRAIN_CONFIG["pretrain_epochs"]):
        model.train()
        epoch_loss = 0
        total_batches = len(train_loader)
        progress_bar = tqdm(train_loader, desc=f"预训练 Epoch {epoch+1}/{TRAIN_CONFIG['pretrain_epochs']}")
        for batch_idx, (x_ori, x_aug, _) in enumerate(progress_bar):
            x_ori = x_ori.to(DEVICE)
            x_aug = x_aug.to(DEVICE)
            # 关键：正常前向传播，但C被冻结
            # 由于C被冻结且初始化为0（xavier初始化），这里需要手动设置为I
            with torch.no_grad():
                model.self_expression.C.data = torch.eye(batch_size, device=DEVICE)
            # 前向传播 - 使用正常的model.forward
            outputs = model(x_ori, x_aug)
            # 固定A_SE和A_IN为单位矩阵
            current_batch_size = x_ori.size(0)
            eye_2n = torch.eye(2 * current_batch_size, device=DEVICE)
        
            # 计算ℒ_SDR - 使用固定的单位矩阵
            L_SDR = nonlocal_contrastive_self_distillation_loss(
                eye_2n,           # A_SE = I (固定)
                eye_2n,           # A_IN = I (固定)
                outputs["H_SE"],  # 会随着编码器变化
                outputs["H_IN"],  # 会随着编码器变化
                xi, tau
            )
            # ℒ_SE - 当C=I时，重构损失应为0，只有正则化项
            L_SE = outputs["L_SE"]
            # 总损失
            loss = L_SDR + lambda_SE * L_SE
            epoch_loss += loss.item()
            # 反向传播 - 只更新非冻结参数（编码器参数）
            optimizer_pretrain.zero_grad()
            loss.backward()
            optimizer_pretrain.step()
            # 调试：记录权重更新后的值
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'L_SDR': f'{L_SDR.item():.4f}'
            })
    # 打印epoch统计信息
    avg_epoch_loss = epoch_loss / total_batches
    print(f"预训练Epoch {epoch+1}完成: 平均损失 = {avg_epoch_loss:.4f}")
# 预训练完成后解冻C参数
    print("\n预训练完成！解冻自表达矩阵C，开始微调阶段")
    for name, param in model.named_parameters():
        if "self_expression.C" in name:
            param.requires_grad = True
            print(f"解冻参数: {name}")
    print("="*60)
    # ============================ 微调阶段 ============================
    # Algorithm 1第4-8行
    print(f"\n微调阶段: M={M}次迭代")
    print("="*60)
    
    best_metrics = {"ACC": 0, "NMI": 0, "ARI": 0}
    loss_history = []
    current_epoch = 0  # 初始化current_epoch变量
    val_metrics = {"ACC": 0, "NMI": 0, "ARI": 0}  # 初始化val_metrics变量
    # 微调阶段
    for iteration in range(M):
        # 采样batch
        indices = torch.randperm(len(train_loader.dataset))[:batch_size]
        batch_data = [train_loader.dataset[i] for i in indices]
        x_ori = torch.stack([item[0] for item in batch_data]).to(DEVICE)
        x_aug = torch.stack([item[1] for item in batch_data]).to(DEVICE)
        
        # --- 步骤1: 仅更新C ---
        with torch.no_grad():
            Z_ori = model.encoder(x_ori)
            Z_aug = model.encoder(x_aug)
    
        L_SE_batch, _, _ = model.self_expression(Z_ori, Z_aug)
        optimizer_C.zero_grad()
        L_SE_batch.backward()
        torch.nn.utils.clip_grad_norm_([C_param], TRAIN_CONFIG["grad_clip"])
        optimizer_C.step()
        
        # --- 步骤2: 更新所有参数 ---
        outputs = model(x_ori, x_aug)
        L_SE_total = outputs["L_SE"]
        L_SDR_total = nonlocal_contrastive_self_distillation_loss(
            outputs["A_SE"],
            outputs["A_IN"],
            outputs["H_SE"],
            outputs["H_IN"],
            xi, tau
        )
        total_loss = L_SDR_total + lambda_SE * L_SE_total
        
        # 记录损失
        loss_history.append(total_loss.item())
        
        optimizer_encoder.zero_grad()
        optimizer_C.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
        optimizer_encoder.step()
        optimizer_C.step()
        
        # 每100次迭代显示一次进度
        if iteration % 100 == 0:
            if len(loss_history) > 0:
                avg_loss = np.mean(loss_history[-100:]) if len(loss_history) >= 100 else np.mean(loss_history)
                print(f"迭代 {iteration}/{M}: 损失={avg_loss:.4f}")
        
        # 每个epoch后用验证集评估
        if (iteration + 1) % len(train_loader) == 0:
            current_epoch = (iteration + 1) // len(train_loader)
            
            # 计算epoch平均损失
            if len(loss_history) > 0:
                epoch_losses = loss_history[-len(train_loader):]
                epoch_avg_loss = np.mean(epoch_losses)
                print(f"\n微调 Epoch {current_epoch}/{TRAIN_CONFIG['finetune_epochs']} 完成:")
                print(f"  平均损失: {epoch_avg_loss:.4f}")
                val_metrics = evaluate_model(model, val_loader)
                print(f"  验证集 - ACC: {val_metrics['ACC']:.2f}%, NMI: {val_metrics['NMI']:.2f}%, ARI: {val_metrics['ARI']:.2f}%")
                    
                # 保存最佳模型
                if val_metrics["ACC"] > best_metrics["ACC"]:
                    best_metrics = val_metrics
                    torch.save({
                        'epoch': current_epoch,
                        'iteration': iteration,
                        'model_state_dict': model.state_dict(),
                        'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
                        'optimizer_C_state_dict': optimizer_C.state_dict(),
                        'metrics': best_metrics,
                        'loss_history': loss_history,
                    }, save_path)
                    print(f"  ✓ 保存最佳模型到 {save_path}")
    print("\n微调阶段完成")
    # ============================ 测试阶段 ============================
    # Algorithm 1第9-13行
    print("\n" + "="*60)
    print("测试阶段")
    print("使用从未见过的测试集数据")
    print("="*60)
    # 加载最佳模型
    checkpoint = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    # 测试
    test_metrics = evaluate_model(model, test_loader)
    print(f"\n训练完成!")
    print(f"总迭代次数: {iteration}")
    print(f"\n最佳验证集指标:")
    print(f"  ACC: {best_metrics['ACC']:.2f}%")
    print(f"  NMI: {best_metrics['NMI']:.2f}%")
    print(f"  ARI: {best_metrics['ARI']:.2f}%")
    print(f"\n测试集指标 (未见过的数据):")
    print(f"  ACC: {test_metrics['ACC']:.2f}%")
    print(f"  NMI: {test_metrics['NMI']:.2f}%")
    print(f"  ARI: {test_metrics['ARI']:.2f}%")
    # 保存最终结果
    results = {
        'dataset': dataset_name,
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'test_size': len(test_loader.dataset),
        'best_val_metrics': best_metrics,
        'test_metrics': test_metrics,
        'total_iterations': iteration,
        'total_epochs': current_epoch
    }
    with open(f"results_{dataset_name}.txt", "w") as f:
        f.write("DISSC训练结果\n")
        f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"数据集划分:\n")
        f.write(f"  训练集: {len(train_loader.dataset)} 样本\n")
        f.write(f"  验证集: {len(val_loader.dataset)} 样本\n")
        f.write(f"  测试集: {len(test_loader.dataset)} 样本\n\n")
        f.write(f"最佳验证集指标:\n")
        f.write(f"  ACC: {best_metrics['ACC']:.2f}%\n")
        f.write(f"  NMI: {best_metrics['NMI']:.2f}%\n")
        f.write(f"  ARI: {best_metrics['ARI']:.2f}%\n\n")
        f.write(f"测试集指标:\n")
        f.write(f"  ACC: {test_metrics['ACC']:.2f}%\n")
        f.write(f"  NMI: {test_metrics['NMI']:.2f}%\n")
        f.write(f"  ARI: {test_metrics['ARI']:.2f}%\n")
    print(f"\n结果已保存至: results_{dataset_name}.txt")
    return model, best_metrics, test_metrics
def evaluate_model(model, dataloader):
    """评估模型"""
    model.eval()
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for x_ori, _, y in dataloader:
            x_ori = x_ori.to(DEVICE)
            Z_ori = model.encoder(x_ori)
            H_IN = model.inductive_head(Z_ori)
            y_pred = torch.argmax(H_IN, dim=1).cpu().numpy()
            all_y_true.extend(y.numpy())
            all_y_pred.extend(y_pred)
    
    metrics = evaluate_clustering(np.array(all_y_true), np.array(all_y_pred))
    model.train()
    return metrics

if __name__ == "__main__":
    # 设置全局随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print("DISSC训练 - 使用data_loader.py的划分")
    print("="*60)
    
    # 打印设备信息
    print(f"使用设备: {DEVICE}")
    
    start_time = time.time()
    
    try:
        model, best_metrics, test_metrics = train_dissc(dataset_name="mnist")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n总训练时间: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
        
    except Exception as e:
        print(f"训练出错: {e}")
        import traceback
        traceback.print_exc()