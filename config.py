# config.py
import torch

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 数据集配置
DATASET_CONFIG = {
    "mnist": {
        "num_clusters": 10,
        "in_channels": 1,
        "img_size": 28,
        "xi": 0.8,  # 正负样本阈值
        "batch_size": 256,
        "data_root": "./data"
    },
    "fashion-mnist": {
        "num_clusters": 10,
        "in_channels": 1,
        "img_size": 28,
        "xi": 0.7,
        "batch_size": 256,
        "data_root": "./data"
    },
    "cifar10": {
        "num_clusters": 10,
        "in_channels": 3,
        "img_size": 32,
        "xi": 0.8,
        "batch_size": 128,
        "data_root": "./data"
    },
    "stl10": {
        "num_clusters": 10,
        "in_channels": 3,
        "img_size": 96,
        "xi": 0.8,
        "batch_size": 64,
        "data_root": "./data"
    }
}

# 训练超参数（更新）
TRAIN_CONFIG = {
    "feature_dim": 128,        # 编码器输出特征维度
    "tau": 0.1,                # 温度系数
    "lambda_SE": 0.1,          # 自表达损失权重
    "lr": 1e-3,                # 学习率
    "pretrain_epochs": 10,     # 预训练轮次
    "finetune_epochs": 100,    # 微调轮次
    "save_path": "best_model.pth",  # 最优模型保存路径
    "grad_clip": 1.0,          # 梯度裁剪阈值
    "early_stop_patience": 20, # 早停耐心值
}

# 数据增强配置（论文16种增强策略）
AUGMENT_CONFIG = {
    "mnist": [
        "Identity", "Rotate", "ShearX", "ShearY", "Posterize",
        "Invert", "Brightness", "Equalize", "Solarize", "Contrast",
        "AutoContrast", "Sharpness", "TranslateX", "TranslateY", "GaussianBlur"
    ],
    "default": [
        "Identity", "FlipLR", "FlipUD", "Rotate", "ShearX", "ShearY",
        "Posterize", "Invert", "Brightness", "Equalize", "Solarize",
        "Contrast", "AutoContrast", "Sharpness", "TranslateX", "TranslateY",
        "Cutout", "GaussianBlur"
    ]
}