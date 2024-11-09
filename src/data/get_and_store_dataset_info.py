# src/data/get_and_store_dataset_info.py

import os
import json
from datetime import datetime
import logging

def setup_logging(log_file='model_utils.log'):
    """
    配置日志记录。
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def load_data_info(data_info_path):
    """
    加载现有的 data_info.json 文件。如果文件不存在，则返回一个空的结构。
    """
    if not os.path.isfile(data_info_path):
        logging.info(f"{data_info_path} 不存在，将创建一个新的文件。")
        return {
            "dataset_info": {},
            "model_training_info": []
        }
    with open(data_info_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data_info(data_info, data_info_path):
    """
    将 data_info 结构保存到 data_info.json 文件中。
    """
    with open(data_info_path, 'w', encoding='utf-8') as f:
        json.dump(data_info, f, indent=4, ensure_ascii=False)
    logging.info(f"数据已保存到 {data_info_path}")

# 数据集信息相关函数
def create_dataset_info(data_info_path, description, labels, source, version, sample_count, additional_info=None):
    """
    创建并记录数据集的信息到 data_info.json 文件中。
    
    Args:
        data_info_path (str): data_info.json 文件的路径。
        description (str): 数据集描述。
        labels (dict): 标签映射，例如 {'0': 'No Crack', '1': 'Crack'}。
        source (str): 数据集来源。
        version (str): 数据集版本。
        sample_count (dict): 每个类别的样本数量，例如 {'0': 20000, '1': 20000}。
        additional_info (dict, optional): 其他相关信息。
    """
    data_info = load_data_info(data_info_path)
    data_info['dataset_info'] = {
        "description": description,
        "labels": labels,
        "source": source,
        "version": version,
        "sample_count": sample_count,
        "additional_info": additional_info or {}
    }
    save_data_info(data_info, data_info_path)
    logging.info("数据集信息已创建。")

def update_dataset_info(data_info_path, key, value):
    """
    更新数据集的信息。
    
    Args:
        data_info_path (str): data_info.json 文件的路径。
        key (str): 要更新的键，例如 'description'。
        value: 更新后的值。
    """
    data_info = load_data_info(data_info_path)
    if 'dataset_info' not in data_info:
        data_info['dataset_info'] = {}
    data_info['dataset_info'][key] = value
    save_data_info(data_info, data_info_path)
    logging.info(f"数据集信息已更新：{key} = {value}")

# 模型训练信息相关函数
def create_model_training_info(data_info_path, model_name, method_type, evaluation_results, additional_metadata=None):
    """
    创建并记录模型的训练和评估信息到 data_info.json 文件中。
    
    Args:
        data_info_path (str): data_info.json 文件的路径。
        model_name (str): 模型名称，例如 'resnet50'。
        method_type (str): 方法类型，例如 'supervised' 或 'unsupervised'。
        evaluation_results (dict): 评估结果，例如 {
            "accuracy": 0.8273,
            "precision": 0.8081,
            "recall": 1.0000,
            "f1": 0.8939,
            "confusion_matrix": [[11, 19], [0, 80]]
        }
        additional_metadata (dict, optional): 其他元数据，例如学习率、权重衰减等。
    """
    data_info = load_data_info(data_info_path)
    training_entry = {
        "model_name": model_name,
        "method_type": method_type,
        "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_results": evaluation_results,
        "additional_metadata": additional_metadata or {}
    }
    data_info['model_training_info'].append(training_entry)
    save_data_info(data_info, data_info_path)
    logging.info("模型训练信息已创建。")

def upsert_model_training_info(data_info_path, model_name, method_type, evaluation_results, additional_metadata=None):
    """
    更新或插入模型的训练和评估信息到 data_info.json 文件中。
    
    如果模型已经存在，则更新其评估结果；否则，添加一个新的记录。
    
    Args:
        data_info_path (str): data_info.json 文件的路径。
        model_name (str): 模型名称，例如 'resnet50'。
        method_type (str): 方法类型，例如 'supervised' 或 'unsupervised'。
        evaluation_results (dict): 评估结果。
        additional_metadata (dict, optional): 其他元数据，例如学习率、权重衰减等。
    """
    data_info = load_data_info(data_info_path)
    
    # 查找是否已有该模型的训练信息
    existing_entry = None
    for entry in data_info.get('model_training_info', []):
        if entry['model_name'] == model_name and entry['method_type'] == method_type:
            existing_entry = entry
            break
    
    if existing_entry:
        # 更新现有记录
        existing_entry['training_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        existing_entry['evaluation_results'] = evaluation_results
        existing_entry['additional_metadata'] = additional_metadata or {}
        logging.info(f"更新模型训练信息：{model_name} - {method_type}")
    else:
        # 添加新记录
        training_entry = {
            "model_name": model_name,
            "method_type": method_type,
            "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_results": evaluation_results,
            "additional_metadata": additional_metadata or {}
        }
        data_info['model_training_info'].append(training_entry)
        logging.info(f"添加新模型训练信息：{model_name} - {method_type}")
    
    save_data_info(data_info, data_info_path)

def update_model_training_info(data_info_path, model_index, key, value):
    """
    更新模型训练的信息。
    
    Args:
        data_info_path (str): data_info.json 文件的路径。
        model_index (int): 模型训练信息的索引。
        key (str): 要更新的键，例如 'evaluation_results.accuracy'。
        value: 更新后的值。
    """
    data_info = load_data_info(data_info_path)
    if 'model_training_info' not in data_info or model_index >= len(data_info['model_training_info']):
        logging.error("指定的模型训练信息不存在。")
        return
    keys = key.split('.')
    entry = data_info['model_training_info'][model_index]
    for k in keys[:-1]:
        entry = entry.setdefault(k, {})
    entry[keys[-1]] = value
    save_data_info(data_info, data_info_path)
    logging.info(f"模型训练信息已更新：{key} = {value}")

def read_data_info(data_info_path):
    """
    读取 data_info.json 文件并返回其内容。
    
    Args:
        data_info_path (str): data_info.json 文件的路径。
    
    Returns:
        dict: data_info.json 的内容。
    """
    if not os.path.isfile(data_info_path):
        logging.warning(f"{data_info_path} 不存在。")
        return {}
    with open(data_info_path, 'r', encoding='utf-8') as f:
        return json.load(f)

from config.config_operator import get_model_dataset_info

def get_recent_model_dataset_info():
    """
    Args:
        data_info_path (str): data_info.json 文件的路径。
    
    Returns:
        dict: 最近一次训练的模型信息。
    """
    setting = get_model_dataset_info()
    print(setting)
    
    data_info = read_data_info(get_model_dataset_info()['data_info_path'])
    if not data_info:
        return {}
    return data_info
