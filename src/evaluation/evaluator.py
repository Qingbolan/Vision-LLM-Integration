import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, dataloader, device, metrics=None):
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1"]

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    results = {}
    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(all_labels, all_preds)
    if "precision" in metrics:
        results["precision"] = precision_score(all_labels, all_preds, average='binary')
    if "recall" in metrics:
        results["recall"] = recall_score(all_labels, all_preds, average='binary')
    if "f1" in metrics:
        results["f1"] = f1_score(all_labels, all_preds, average='binary')

    return results