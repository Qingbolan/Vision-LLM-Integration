import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate_model(model, dataloader, device, metrics=None, save_confusion_matrix=False, checkpoint_path='./checkpoints/'):
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1"]

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
        if save_confusion_matrix:
            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            cm_path = os.path.join(checkpoint_path, 'confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()
            print(f'Confusion matrix saved at {cm_path}')

    return results