# src/evaluation/variational_autoencoder_evaluator.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import os

from src.data.get_and_store_dataset_info import upsert_model_training_info

def evaluate_variational_autoencoder(model, dataloader, device, checkpoint_path, num_images=10, threshold=None):
    model.eval()
    recon_errors = []
    kl_divergences = []
    labels = []
    reconstructions = []
    originals = []

    with torch.no_grad():
        for inputs, label in dataloader:
            inputs = inputs.to(device)
            outputs, mu, logvar = model(inputs)
            recon_loss = F.mse_loss(outputs, inputs, reduction='none')
            recon_loss = recon_loss.view(recon_loss.size(0), -1).mean(dim=1)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            total_loss = recon_loss + kl_loss
            recon_errors.extend(recon_loss.cpu().numpy())
            kl_divergences.extend(kl_loss.cpu().numpy())
            labels.extend(label.numpy())
            # 收集部分图像用于可视化
            reconstructions.append(outputs.cpu())
            originals.append(inputs.cpu())

    recon_errors = np.array(recon_errors)
    kl_divergences = np.array(kl_divergences)
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

    # 绘制 ROC 曲线
    fpr, tpr, thresholds = roc_curve(labels, recon_errors)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(checkpoint_path, 'roc_curve.png'))
    plt.close()
    print(f'ROC curve saved at {os.path.join(checkpoint_path, "roc_curve.png")}')

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
    recon_path = os.path.join(checkpoint_path, 'variational_autoencoder_reconstructions.png')
    plt.savefig(recon_path)
    plt.close()
    print(f'Reconstruction images saved at {recon_path}')

    # 绘制混淆矩阵
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['No Crack', 'Crack'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(checkpoint_path, 'variational_autoencoder_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f'Confusion matrix saved at {cm_path}')

    # 准备评估结果字典
    evaluation_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }

    # 推断 data_info.json 的存储位置
    # 如果 checkpoint_path 是 './checkpoints/', 则 data_info.json 位于 './data/raw/data_info.json'
    # 否则，假设 checkpoint_path 是 './checkpoints/SomeDataset/', 则 data_info.json 位于 './data/SomeDataset/data_info.json'
    default_checkpoint_dir = './checkpoints/'
    if os.path.abspath(checkpoint_path) == os.path.abspath(default_checkpoint_dir):
        data_info_path = os.path.join('./data/raw/', 'data_info.json')
    else:
        # 获取相对于 checkpoints/ 目录的子目录
        relative_path = os.path.relpath(checkpoint_path, default_checkpoint_dir)
        data_info_path = os.path.join('./data/', relative_path, 'data_info.json')
    
    # 确保 data_info.json 所在目录存在
    data_info_dir = os.path.dirname(data_info_path)
    os.makedirs(data_info_dir, exist_ok=True)

    # 记录模型的训练和评估信息
    # 假设模型名称和方法类型可以从模型对象或其他来源获取
    # 这里需要您根据实际情况传递正确的 model_name 和 method_type
    # 例如，如果模型名称存储在 model.name 中
    model_name = getattr(model, 'name', 'autoencoder')  # 默认名称为 'autoencoder'
    method_type = 'unsupervised'  # 根据您的实际情况设置

    # 调用 upsert_model_training_info 函数
    upsert_model_training_info(
        data_info_path=data_info_path,
        model_name=model_name,
        method_type=method_type,
        evaluation_results=evaluation_results,
    )

    return evaluation_results


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
