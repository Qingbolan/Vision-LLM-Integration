from src.DeepLearning_pipeline_with_cofig import main as main_pipeline
from config.config_operator import update_method_type_and_selection, update_data_paths, change_dataset

def train(is_supervised=True, model_name='resnet50', data_path='./data/raw/', checkpoint_path='./checkpoints/'):
    config_yaml_path = 'config/config.yaml'  # 替换为您的YAML文件路径
    
    # # 1. 更新方法类型和选择
    # # 例如，将方法类型设置为'supervised'，并选择'resnet50'作为模型
    # update_method_type_and_selection(config_yaml_path, 'supervised', 'resnet50')
    
    # # 或者，将方法类型设置为'unsupervised'，并选择'dcae'作为方法
    # # update_method_type_and_selection(config_yaml_path, 'unsupervised', 'dcae')
    
    # # 2. 更新数据路径
    # # 例如，手动设置数据路径和检查点路径
    # update_data_paths(config_yaml_path, './data/BreastMNIST/', './checkpoints/BreastMNIST/')
    
    # # 3. 根据数据集名称自动更新数据路径和检查点路径
    # # 例如，切换到'ConcreteCrack'
    # # change_dataset(config_yaml_path, 'ConcreteCrack')

    update_method_type_and_selection(config_yaml_path, 'supervised' if is_supervised else 'unsupervised', model_name)
    update_data_paths(config_yaml_path, data_path, checkpoint_path)
    change_dataset(config_yaml_path, data_path.split('/')[-2])
    main_pipeline()