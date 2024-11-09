import yaml
import os
import logging

def setup_logging(log_file='config_update.log'):
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

def load_yaml(yaml_path):
    """
    加载YAML文件。
    
    Args:
        yaml_path (str): YAML文件的路径。
        
    Returns:
        dict: YAML文件内容。
    """
    if not os.path.isfile(yaml_path):
        logging.error(f"YAML文件不存在: {yaml_path}")
        return {}
    
    with open(yaml_path, 'r', encoding='utf-8') as file:
        try:
            data = yaml.safe_load(file)
            logging.info(f"成功加载YAML文件: {yaml_path}")
            return data
        except yaml.YAMLError as exc:
            logging.error(f"加载YAML文件时出错: {exc}")
            return {}

def save_yaml(data, yaml_path):
    """
    保存数据到YAML文件。
    
    Args:
        data (dict): 要保存的数据。
        yaml_path (str): YAML文件的路径。
    """
    with open(yaml_path, 'w', encoding='utf-8') as file:
        try:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
            logging.info(f"成功保存YAML文件: {yaml_path}")
        except yaml.YAMLError as exc:
            logging.error(f"保存YAML文件时出错: {exc}")

def update_method_type_and_selection(yaml_path, method_type, selection):
    """
    更新方法类型（监督学习或非监督学习）以及对应的模型或方法选择。
    
    Args:
        yaml_path (str): YAML配置文件的路径。
        method_type (str): 方法类型，'supervised' 或 'unsupervised'。
        selection (str): 
            - 如果method_type为'supervised'，则为模型名称（如'resnet50', 'alexnet', 'vgg16', 'vit'）。
            - 如果method_type为'unsupervised'，则为方法名称（如'dcae', 'dcvae', 'vit_anomaly'）。
    """
    data = load_yaml(yaml_path)
    if not data:
        logging.error("无法加载YAML数据，终止更新。")
        return
    
    if method_type not in ['supervised', 'unsupervised']:
        logging.error("方法类型无效。请选择'supervised'或'unsupervised'。")
        return
    
    data['method']['type'] = method_type
    logging.info(f"设置方法类型为: {method_type}")
    
    if method_type == 'supervised':
        supervised_models = ['resnet50', 'alexnet', 'vgg16', 'vit']
        if selection not in supervised_models:
            logging.error(f"无效的监督学习模型选择。可选项: {supervised_models}")
            return
        data['method']['supervised']['model']['name'] = selection
        logging.info(f"设置监督学习模型为: {selection}")
    else:
        unsupervised_methods = ['dcae', 'dcvae', 'vit_anomaly']
        if selection not in unsupervised_methods:
            logging.error(f"无效的非监督学习方法选择。可选项: {unsupervised_methods}")
            return
        data['method']['unsupervised']['method'] = selection
        logging.info(f"设置非监督学习方法为: {selection}")
    
    save_yaml(data, yaml_path)

def update_data_paths(yaml_path, raw_data_path, checkpoint_path=None):
    """
    更新数据集路径（原始数据路径）和检查点路径。
    
    Args:
        yaml_path (str): YAML配置文件的路径。
        raw_data_path (str): 原始数据集的路径（例如'./data/BreastMNIST/'）。
        checkpoint_path (str, optional): 检查点路径。如果未提供，将自动推断。
    """
    data = load_yaml(yaml_path)
    if not data:
        logging.error("无法加载YAML数据，终止更新。")
        return
    
    data['data']['raw_data_path'] = raw_data_path
    logging.info(f"设置原始数据路径为: {raw_data_path}")
    
    if checkpoint_path:
        data['training']['checkpoint_path'] = checkpoint_path
        logging.info(f"设置检查点路径为: {checkpoint_path}")
    else:
        # 根据raw_data_path推断checkpoint_path
        dataset_name = os.path.basename(os.path.normpath(raw_data_path))
        inferred_checkpoint_path = f"./checkpoints/{dataset_name}/"
        data['training']['checkpoint_path'] = inferred_checkpoint_path
        logging.info(f"根据数据集推断检查点路径为: {inferred_checkpoint_path}")
    
    save_yaml(data, yaml_path)

def change_dataset(yaml_path, dataset_name):
    """
    根据数据集名称自动更新原始数据路径和检查点路径。
    
    Args:
        yaml_path (str): YAML配置文件的路径。
        dataset_name (str): 数据集名称（例如'BreastMNIST'）。
    """
    raw_data_path = f"./data/{dataset_name}/"
    checkpoint_path = f"./checkpoints/{dataset_name}/"
    
    update_data_paths(yaml_path, raw_data_path, checkpoint_path)

if __name__ == "__main__":
    setup_logging()
    
    # 示例用法：
    config_yaml_path = '.config.yaml'  # 替换为您的YAML文件路径
    
    # 1. 更新方法类型和选择
    # 例如，将方法类型设置为'supervised'，并选择'resnet50'作为模型
    update_method_type_and_selection(config_yaml_path, 'supervised', 'resnet50')
    
    # 或者，将方法类型设置为'unsupervised'，并选择'dcae'作为方法
    # update_method_type_and_selection(config_yaml_path, 'unsupervised', 'dcae')
    
    # 2. 更新数据路径
    # 例如，手动设置数据路径和检查点路径
    update_data_paths(config_yaml_path, './data/BreastMNIST/', './checkpoints/BreastMNIST/')
    
    # 3. 根据数据集名称自动更新数据路径和检查点路径
    # 例如，切换到'ConcreteCrack'
    # change_dataset(config_yaml_path, 'ConcreteCrack')
