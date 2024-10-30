# src/evaluation/autoencoder_evaluator.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import os

def evaluate_autoencoder(model, dataloader, device, checkpoint_path, num_images=10, threshold=None):
    model.eval()
    recon_errors = []
    labels = []
    reconstructions = []
    originals = []

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            error = F.mse_loss(outputs, inputs, reduction='none')
            error = error.view(error.size(0), -1).mean(dim=1)
            recon_errors.extend(error.cpu().numpy())
            labels.extend(label.numpy())
            # 收集部分图像用于可视化
            reconstructions.append(outputs.cpu())
            originals.append(inputs.cpu())

    recon_errors = np.array(recon_errors)
    labels = np.array(labels)

    # 绘制重构误差分布
    plot_reconstruction_error(recon_errors, labels, checkpoint_path)

    # 如果没有给定阈值，使用验证集正常样本的95百分位数
    if threshold is None:
        threshold = np.percentile(recon_errors[labels==0], 95)
        print(f'Dynamic threshold set to {threshold}')

    # 基于阈值确定预测标签
    preds = (recon_errors > threshold).astype(int)

    # 计算评估指标
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)

    print('Confusion Matrix:')
    print(cm)
    print('Classification Report:')
    print(classification_report(labels, preds))

    # 可视化重构图像
    reconstructions = torch.cat(reconstructions, dim=0)[:num_images]
    originals = torch.cat(originals, dim=0)[:num_images]

    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(5, 2 * num_images))
    for i in range(num_images):
        # 原始图像
        axes[i, 0].imshow(np.clip(originals[i].permute(1, 2, 0).numpy(), 0, 1))
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original')

        # 重构图像
        axes[i, 1].imshow(np.clip(reconstructions[i].permute(1, 2, 0).numpy(), 0, 1))
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Reconstructed')

    plt.tight_layout()
    recon_path = os.path.join(checkpoint_path, 'autoencoder_reconstructions.png')
    plt.savefig(recon_path)
    plt.close()
    print(f'Reconstruction images saved at {recon_path}')

    # 绘制混淆矩阵
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['No Crack', 'Crack'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(checkpoint_path, 'autoencoder_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f'Confusion matrix saved at {cm_path}')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }

def plot_reconstruction_error(recon_errors, labels, checkpoint_path):
    plt.figure(figsize=(10,6))
    plt.hist(recon_errors[labels==0], bins=50, alpha=0.5, label='Normal')
    plt.hist(recon_errors[labels==1], bins=50, alpha=0.5, label='Anomaly')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.title('Reconstruction Error Distribution')
    plt.savefig(os.path.join(checkpoint_path, 'reconstruction_error_distribution.png'))
    plt.close()
