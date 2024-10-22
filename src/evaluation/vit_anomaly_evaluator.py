import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate_vit_anomaly(model, dataloader, device, metrics=["accuracy", "precision", "recall", "f1", "confusion_matrix"], checkpoint_path='./checkpoints/'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    results = {}
    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(all_labels, all_preds)
    if "precision" in metrics:
        results["precision"] = precision_score(all_labels, all_preds, average='binary')
    if "recall" in metrics:
        results["recall"] = recall_score(all_labels, all_preds, average='binary')
    if "f1" in metrics:
        results["f1"] = f1_score(all_labels, all_preds, average='binary')
    if "confusion_matrix" in metrics:
        cm = confusion_matrix(all_labels, all_preds)
        results["confusion_matrix"] = cm
        # 绘制混淆矩阵
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        confusion_matrix_path = os.path.join(checkpoint_path, 'vit_anomaly_confusion_matrix.png')
        plt.savefig(confusion_matrix_path)
        plt.close()
        print(f'Confusion matrix saved at {confusion_matrix_path}')

    return results