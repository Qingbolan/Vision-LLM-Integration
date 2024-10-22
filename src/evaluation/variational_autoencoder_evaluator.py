# src/evaluation/variational_autoencoder_evaluator.py

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

def evaluate_variational_autoencoder(model, dataloader, device, checkpoint_path, num_images=10, threshold=0.5):
    """
    评估 ConvolutionalVariationalAutoencoder 模型，计算重构误差和 KL 散度，并计算分类指标。
    
    Args:
        model (nn.Module): 训练好的变分自动编码器模型。
        dataloader (DataLoader): 验证集数据加载器。
        device (torch.device): 使用的设备（CPU 或 CUDA）。
        checkpoint_path (str): 用于保存评估结果的目录。
        num_images (int): 要可视化的图像数量。
        threshold (float): 用于确定异常的阈值。
    
    Returns:
        dict: 包含平均重构误差、平均 KL 散度和分类指标的字典。
    """
    model.eval()
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0
    reconstructions = []
    originals = []
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, mu, log_sigma = model(inputs)
            
            # 计算重构损失（MSE）
            recon_loss = F.mse_loss(outputs, inputs, reduction='sum')
            total_recon_loss += recon_loss.item()
            
            # 计算 KL 散度
            kl_loss = -0.5 * torch.sum(1 + 2*log_sigma - mu.pow(2) - torch.exp(2*log_sigma))
            total_kl_loss += kl_loss.item()
            
            # 计算总异常分数（重构误差 + KL 散度）
            recon_error = F.mse_loss(outputs, inputs, reduction='none').view(inputs.size(0), -1).mean(dim=1).cpu().numpy()
            kl_score = (-0.5 * torch.sum(1 + 2*log_sigma - mu.pow(2) - torch.exp(2*log_sigma), dim=1)).cpu().numpy()
            total_error = recon_error + kl_score
            all_scores.extend(total_error)
            all_labels.extend(labels.cpu().numpy())
            
            total_samples += inputs.size(0)
            
            # 收集部分重构图像用于可视化
            reconstructions.append(outputs.cpu())
            originals.append(inputs.cpu())
            if len(reconstructions) * dataloader.batch_size >= num_images:
                break
    
    avg_recon_loss = total_recon_loss / (total_samples * np.prod(inputs.shape[1:]))
    avg_kl_loss = total_kl_loss / total_samples
    
    # 基于阈值确定预测标签
    all_preds = [1 if score > threshold else 0 for score in all_scores]
    
    # 计算分类指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f'Average Reconstruction Loss (MSE): {avg_recon_loss:.6f}')
    print(f'Average KL Divergence: {avg_kl_loss:.6f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    # 可视化并保存重构图像
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(checkpoint_path, 'variational_autoencoder_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f'Confusion matrix saved at {cm_path}')
    
    return {
        'average_reconstruction_loss': avg_recon_loss,
        'average_kl_divergence': avg_kl_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }