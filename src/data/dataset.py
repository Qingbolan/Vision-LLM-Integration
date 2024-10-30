# src/data/dataset.py

import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random

class ConcreteCrackDataset(Dataset):
    def __init__(self, file_paths, labels=None, transform=None):
        self.file_paths = file_paths
        self.labels = labels  # 对于无监督学习，训练集的 labels 可以是 None
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image

def load_data(raw_data_path, train_split=0.8, method_type='supervised'):
    negative_dir = os.path.join(raw_data_path, 'Negative')  # 无裂纹（正常）
    positive_dir = os.path.join(raw_data_path, 'Positive')  # 有裂纹（异常）

    # 获取所有正常和异常样本的文件路径
    negative_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    positive_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    negative_labels = [0] * len(negative_files)
    positive_labels = [1] * len(positive_files)

    if method_type == 'supervised':
        # 有监督学习：使用所有数据进行训练和验证
        file_paths = negative_files + positive_files
        labels = negative_labels + positive_labels

        # 打乱数据
        file_paths, labels = shuffle(file_paths, labels, random_state=42)

        # 划分训练集和验证集
        train_files, val_files, train_labels, val_labels = train_test_split(
            file_paths, labels, train_size=train_split, random_state=42, stratify=labels
        )

    elif method_type == 'unsupervised':
        # 无监督学习：训练集只包含正常样本
        # 将正常样本划分为训练集和验证集
        neg_train_files, neg_val_files = train_test_split(
            negative_files, train_size=train_split, random_state=42
        )

        # 计算验证集中需要的异常样本数量（正常验证样本的1/10）
        n_pos_val = len(neg_val_files) // 10
        
        # 随机选择指定数量的异常样本用于验证集
        pos_val_files = random.sample(positive_files, min(n_pos_val, len(positive_files)))

        # 验证集文件路径和标签
        val_files = neg_val_files + pos_val_files
        val_labels = [0] * len(neg_val_files) + [1] * len(pos_val_files)

        # 打乱验证集数据
        val_files, val_labels = shuffle(val_files, val_labels, random_state=42)

        # 训练集只有正常样本，无需标签
        train_files = neg_train_files
        train_labels = None

    else:
        raise ValueError(f"Unsupported method_type: {method_type}")

    return train_files, train_labels, val_files, val_labels