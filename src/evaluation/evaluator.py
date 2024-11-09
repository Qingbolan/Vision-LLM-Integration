import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.data.get_and_store_dataset_info import upsert_model_training_info

def evaluate_model(model, dataloader, device, metrics=None, save_confusion_matrix=False, checkpoint_path='./checkpoints/', model_name='model', method_type='supervised'):
    """
    评估模型并记录评估结果到 data_info.json 文件中。

    Args:
        model (torch.nn.Module): 已训练的模型。
        dataloader (torch.utils.data.DataLoader): 数据加载器。
        device (torch.device): 计算设备。
        metrics (list, optional): 需要计算的评估指标。
        save_confusion_matrix (bool, optional): 是否保存混淆矩阵图像。
        checkpoint_path (str, optional): 模型检查点路径。
        model_name (str, optional): 模型名称。
        method_type (str, optional): 方法类型，例如 'supervised' 或 'unsupervised'。

    Returns:
        dict: 评估结果。
    """
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
        results["confusion_matrix"] = cm.tolist()  # 转换为列表以便JSON序列化
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
    
    # 准备评估结果字典
    evaluation_results = {}
    for metric in metrics:
        if metric in results:
            evaluation_results[metric] = results[metric]
    
    # 推断 data_info.json 的存储位置
    # 如果 checkpoint_path 是 './checkpoints/', 则 data_info.json 位于 './data/raw/data_info.json'
    # 否则，假设 checkpoint_path 是 './checkpoints/SomeDataset/', 则 data_info.json 位于 './data/SomeDataset/data_info.json'
    default_checkpoint_dir = './checkpoints/'
    abs_checkpoint_path = os.path.abspath(checkpoint_path)
    abs_default_checkpoint_dir = os.path.abspath(default_checkpoint_dir)
    
    if abs_checkpoint_path == abs_default_checkpoint_dir:
        data_info_path = os.path.join('./data/raw/', 'data_info.json')
    else:
        # 获取相对于 checkpoints/ 目录的子目录
        relative_path = os.path.relpath(checkpoint_path, default_checkpoint_dir)
        data_info_path = os.path.join('./data/', relative_path, 'data_info.json')
    
    # 确保 data_info.json 所在目录存在
    data_info_dir = os.path.dirname(data_info_path)
    os.makedirs(data_info_dir, exist_ok=True)
    
    # 记录模型的训练和评估信息
    # 调用 upsert_model_training_info 函数
    upsert_model_training_info(
        data_info_path=data_info_path,
        model_name=model_name,
        method_type=method_type,
        evaluation_results=evaluation_results,
    )
    
    return evaluation_results