import torch
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def evaluate_vit_anomaly(model, dataloader, device, checkpoint_path, num_images=10, threshold=None):
    """
    评估无监督ViT异常检测模型。

    参数:
        model: 已训练的模型。
        dataloader: 测试数据加载器，包含正常和异常图像。
        device: 计算设备（'cuda' 或 'cpu'）。
        checkpoint_path: 结果保存路径。
        num_images: 用于可视化的图像数量。
        threshold: 异常阈值，如果为None，则动态计算。

    返回:
        包含评估指标的字典。
    """
    model.eval()
    recon_errors = []
    labels = []
    reconstructions = []
    originals = []

    with torch.no_grad():
        for data in dataloader:
            if isinstance(data, (tuple, list)):
                inputs, label = data
                inputs = inputs.to(device)
                labels.extend(label.numpy())
            else:
                inputs = data.to(device)
                labels = None  # 如果没有标签

            z, reconstructed = model(inputs)

            # 计算重构误差
            features = model.vit(inputs)
            recon_error = F.mse_loss(reconstructed, features, reduction='none')
            recon_error = recon_error.view(recon_error.size(0), -1).mean(dim=1)
            recon_errors.extend(recon_error.cpu().numpy())

            # 收集部分图像用于可视化
            if len(reconstructions) < num_images:
                reconstructions.append(reconstructed.cpu())
                originals.append(inputs.cpu())

    if labels is None:
        raise ValueError("测试数据集需要包含标签以进行评估。")

    recon_errors = np.array(recon_errors)
    labels = np.array(labels)

    # 绘制重构误差分布
    plot_reconstruction_error(recon_errors, labels, checkpoint_path)

    # 如果没有给定阈值，使用正常样本的95百分位数作为阈值
    if threshold is None:
        threshold = np.percentile(recon_errors[labels == 0], 95)
        print(f'Dynamic threshold set to {threshold}')

    # 基于阈值确定预测标签
    preds = (recon_errors > threshold).astype(int)

    # 计算评估指标
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    roc_auc = roc_auc_score(labels, recon_errors)

    print('Confusion Matrix:')
    print(cm)
    print('Classification Report:')
    print(classification_report(labels, preds))
    print(f'ROC AUC Score: {roc_auc:.4f}')

    # 可视化重构图像
    reconstructions = torch.cat(reconstructions, dim=0)[:num_images]
    originals = torch.cat(originals, dim=0)[:num_images]

    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(10, 4 * num_images))
    for i in range(num_images):
        # 原始图像
        axes[i, 0].imshow(np.transpose(np.clip(originals[i].cpu().numpy(), 0, 1), (1, 2, 0)))
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original')

        # 重构图像
        axes[i, 1].imshow(np.transpose(np.clip(reconstructions[i].cpu().numpy(), 0, 1), (1, 2, 0)))
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(checkpoint_path, 'autoencoder_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f'Confusion matrix saved at {cm_path}')

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(labels, recon_errors)
    roc_auc_val = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    roc_path = os.path.join(checkpoint_path, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f'ROC curve saved at {roc_path}')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }

def plot_reconstruction_error(recon_errors, labels, checkpoint_path):
    """
    绘制重构误差的分布图。

    参数:
        recon_errors: 重构误差数组。
        labels: 标签数组。
        checkpoint_path: 结果保存路径。
    """
    plt.figure(figsize=(10,6))
    plt.hist(recon_errors[labels==0], bins=50, alpha=0.5, label='Normal')
    plt.hist(recon_errors[labels==1], bins=50, alpha=0.5, label='Anomaly')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.title('Reconstruction Error Distribution')
    plt.savefig(os.path.join(checkpoint_path, 'reconstruction_error_distribution.png'))
    plt.close()
