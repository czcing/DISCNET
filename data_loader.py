from numpy.ma import true_divide
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from config import DATASET_CONFIG, AUGMENT_CONFIG

class TrivialAugment:
    """论文使用的TrivialAugment（参考[57]）"""
    def __init__(self, dataset_name):
        self.aug_list = AUGMENT_CONFIG[dataset_name] if dataset_name in AUGMENT_CONFIG else AUGMENT_CONFIG["default"]
        self.strength_range = 30  # 强度离散化为30级

    def __call__(self, x):
        # 随机选择增强方式和强度
        aug_name = np.random.choice(self.aug_list)
        strength = np.random.randint(0, self.strength_range) / self.strength_range  # 归一化到[0,1]

        if aug_name == "Identity":
            return x
        elif aug_name == "FlipLR":
            return transforms.RandomHorizontalFlip(p=1.0)(x)
        elif aug_name == "FlipUD":
            return transforms.RandomVerticalFlip(p=1.0)(x)
        elif aug_name == "Rotate":
            angle = 30 * strength  # 最大30度
            return transforms.RandomRotation(degrees=(-angle, angle))(x)
        elif aug_name == "ShearX":
            shear = 0.3 * strength  # 最大0.3
            # use symmetric range so shear is sampled from [-shear, shear]
            return transforms.RandomAffine(degrees=0, shear=(-shear, shear))(x)
        elif aug_name == "ShearY":
            shear = 0.3 * strength
            # RandomAffine accepts a 4-tuple (min_x, max_x, min_y, max_y)
            # previous code placed min_y > max_y when shear>0 which caused uniform_ error.
            # Use symmetric per-axis range: (min_x, max_x, min_y, max_y)
            return transforms.RandomAffine(degrees=0, shear=(0, 0, -shear, shear))(x)
        elif aug_name == "TranslateX":
            translate = 0.1 * strength  # 最大10%
            return transforms.RandomAffine(degrees=0, translate=(translate, 0))(x)
        elif aug_name == "TranslateY":
            translate = 0.1 * strength
            return transforms.RandomAffine(degrees=0, translate=(0, translate))(x)
        elif aug_name == "Brightness":
            return transforms.ColorJitter(brightness=0.5*strength + 0.5)(x)
        elif aug_name == "Contrast":
            return transforms.ColorJitter(contrast=0.5*strength + 0.5)(x)
        elif aug_name == "Sharpness":
            return transforms.RandomAdjustSharpness(sharpness_factor=2*strength + 0.5)(x)
        elif aug_name == "GaussianBlur":
            kernel_size = 3 if x.shape[1] <= 32 else 5
            min_sigma = 0.1
            max_sigma = max(min_sigma, 2.0*strength)
            return transforms.GaussianBlur(kernel_size=kernel_size, sigma=(min_sigma, max_sigma))(x)
        else:
            return x

class SubspaceDataset(Dataset):
    """数据集类：返回原始数据、增强数据、真实标签"""
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.x, self.y = self._load_data()
        self.augment = TrivialAugment(dataset_name)
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]

    def _load_data(self):
        """加载数据集"""
        if self.dataset_name == "mnist":
            x, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
            x = x.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0 
        elif self.dataset_name == "fashion-mnist":
            x, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
            x = x.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        elif self.dataset_name == "cifar10":
            from torchvision.datasets import CIFAR10
            # 加载训练集和测试集
            train_dataset = CIFAR10(root="./data", train=True, download=True)
            test_dataset = CIFAR10(root="./data", train=False, download=True)
            x_train = np.transpose(train_dataset.data, (0, 3, 1, 2)).astype(np.float32) / 255.0
            y_train = np.array(train_dataset.targets)
            x_test = np.transpose(test_dataset.data, (0, 3, 1, 2)).astype(np.float32) / 255.0
            y_test = np.array(test_dataset.targets)
            # 合并训练集和测试集
            x = np.concatenate([x_train, x_test], axis=0)
            y = np.concatenate([y_train, y_test], axis=0)
        elif self.dataset_name == "stl10":
            from torchvision.datasets import STL10
            # 加载所有数据
            train_dataset = STL10(root="./data", split="train", download=True)
            test_dataset = STL10(root="./data", split="test", download=True)
            unlabeled_dataset = STL10(root="./data", split="unlabeled", download=True)
            x_train = np.transpose(train_dataset.data, (0, 3, 1, 2)).astype(np.float32) / 255.0
            y_train = np.array(train_dataset.labels)
            x_test = np.transpose(test_dataset.data, (0, 3, 1, 2)).astype(np.float32) / 255.0
            y_test = np.array(test_dataset.labels)
            x_unlabeled = np.transpose(unlabeled_dataset.data, (0, 3, 1, 2)).astype(np.float32) / 255.0
            y_unlabeled = np.array(unlabeled_dataset.labels)
            # 合并所有数据
            x = np.concatenate([x_train, x_test, x_unlabeled], axis=0)
            y = np.concatenate([y_train, y_test, y_unlabeled], axis=0)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        y = y.astype(np.int64)
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_ori = self.normalize(self.x[idx])
        x_aug = self.augment(x_ori)
        y = self.y[idx]
        return x_ori, x_aug, y

def get_dataloader(dataset_name):
    """获取DataLoader（包含训练集、验证集、测试集）"""
    batch_size = DATASET_CONFIG[dataset_name]["batch_size"]
    dataset = SubspaceDataset(dataset_name)
    
    # 划分数据集：60%训练集，20%验证集，20%测试集
    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    # 生成索引并划分
    indices = list(range(total_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # 创建子集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, DATASET_CONFIG[dataset_name]["num_clusters"]