import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def evaluate_vit_anomaly(model, dataloader, device, threshold=0.5, metrics=["accuracy", "precision", "recall", "f1", "confusion_matrix"], checkpoint_path='./checkpoints/'):
    """
    Evaluate ViT model for anomaly detection with threshold control
    
    Args:
        model: The ViT model
        dataloader: Test data loader
        device: Computing device (cuda/cpu)
        threshold: Classification threshold for anomaly detection (default: 0.5)
        metrics: List of metrics to compute
        checkpoint_path: Path to save confusion matrix plot
        
    Returns:
        Dictionary containing computed metrics
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Get model outputs
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            # Store probabilities for positive class
            pos_probs = probs[:, 1].cpu().numpy()
            all_probs.extend(pos_probs)
            
            # Apply threshold to probabilities
            preds = (pos_probs >= threshold).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    results = {
        'threshold': threshold,
        'positive_samples': sum(all_preds == 1),
        'negative_samples': sum(all_preds == 0)
    }
    
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
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (threshold={threshold:.2f})')
        
        # Save confusion matrix plot
        os.makedirs(checkpoint_path, exist_ok=True)
        confusion_matrix_path = os.path.join(
            checkpoint_path, 
            f'vit_anomaly_confusion_matrix_thresh_{threshold:.2f}.png'
        )
        plt.savefig(confusion_matrix_path)
        plt.close()
        print(f'Confusion matrix saved at {confusion_matrix_path}')
        
        # Calculate additional metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        results.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        })
    
    return results