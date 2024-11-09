import yaml
import os
import logging

def setup_logging(log_file='config_update.log'):
    """
    Configure logging.
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
    Load a YAML file.
    
    Args:
        yaml_path (str): Path to the YAML file.
        
    Returns:
        dict: Contents of the YAML file.
    """
    if not os.path.isfile(yaml_path):
        logging.error(f"YAML file does not exist: {yaml_path}")
        return {}
    
    with open(yaml_path, 'r', encoding='utf-8') as file:
        try:
            data = yaml.safe_load(file)
            logging.info(f"Successfully loaded YAML file: {yaml_path}")
            return data
        except yaml.YAMLError as exc:
            logging.error(f"Error loading YAML file: {exc}")
            return {}

def save_yaml(data, yaml_path):
    """
    Save data to a YAML file.
    
    Args:
        data (dict): Data to be saved.
        yaml_path (str): Path to the YAML file.
    """
    with open(yaml_path, 'w', encoding='utf-8') as file:
        try:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
            logging.info(f"Successfully saved YAML file: {yaml_path}")
        except yaml.YAMLError as exc:
            logging.error(f"Error saving YAML file: {exc}")

def update_method_type_and_selection(yaml_path, method_type, selection):
    """
    Update the method type (supervised or unsupervised) and the corresponding model or method selection.
    
    Args:
        yaml_path (str): Path to the YAML configuration file.
        method_type (str): Method type, 'supervised' or 'unsupervised'.
        selection (str): 
            - If method_type is 'supervised', the model name (e.g., 'resnet50', 'alexnet', 'vgg16', 'vit').
            - If method_type is 'unsupervised', the method name (e.g., 'dcae', 'dcvae', 'vit_anomaly').
    """
    data = load_yaml(yaml_path)
    if not data:
        logging.error("Unable to load YAML data, aborting update.")
        return
    
    if method_type not in ['supervised', 'unsupervised']:
        logging.error("Invalid method type. Please choose 'supervised' or 'unsupervised'.")
        return
    
    data['method']['type'] = method_type
    logging.info(f"Set method type to: {method_type}")
    
    if method_type == 'supervised':
        supervised_models = ['resnet50', 'alexnet', 'vgg16', 'vit']
        if selection not in supervised_models:
            logging.error(f"Invalid supervised learning model selection. Options: {supervised_models}")
            return
        data['method']['supervised']['model']['name'] = selection
        logging.info(f"Set supervised learning model to: {selection}")
    else:
        unsupervised_methods = ['dcae', 'dcvae', 'vit_anomaly']
        if selection not in unsupervised_methods:
            logging.error(f"Invalid unsupervised learning method selection. Options: {unsupervised_methods}")
            return
        data['method']['unsupervised']['method'] = selection
        logging.info(f"Set unsupervised learning method to: {selection}")
    
    save_yaml(data, yaml_path)

def update_data_paths(yaml_path, raw_data_path, checkpoint_path=None):
    """
    Update the dataset path (raw data path) and checkpoint path.
    
    Args:
        yaml_path (str): Path to the YAML configuration file.
        raw_data_path (str): Path to the raw dataset (e.g., './data/BreastMNIST/').
        checkpoint_path (str, optional): Path to the checkpoint. If not provided, it will be inferred automatically.
    """
    data = load_yaml(yaml_path)
    if not data:
        logging.error("Unable to load YAML data, aborting update.")
        return
    
    data['data']['raw_data_path'] = raw_data_path
    logging.info(f"Set raw data path to: {raw_data_path}")
    
    if checkpoint_path:
        data['training']['checkpoint_path'] = checkpoint_path
        logging.info(f"Set checkpoint path to: {checkpoint_path}")
    else:
        # Infer checkpoint_path based on raw_data_path
        dataset_name = os.path.basename(os.path.normpath(raw_data_path))
        inferred_checkpoint_path = f"./checkpoints/{dataset_name}/"
        data['training']['checkpoint_path'] = inferred_checkpoint_path
        logging.info(f"Inferred checkpoint path based on dataset: {inferred_checkpoint_path}")
    
    save_yaml(data, yaml_path)

def change_dataset(yaml_path, dataset_name):
    """
    Automatically update the raw data path and checkpoint path based on the dataset name.
    
    Args:
        yaml_path (str): Path to the YAML configuration file.
        dataset_name (str): Name of the dataset (e.g., 'BreastMNIST').
    """
    raw_data_path = f"./data/{dataset_name}/"
    checkpoint_path = f"./checkpoints/{dataset_name}/"
    
    update_data_paths(yaml_path, raw_data_path, checkpoint_path)

def get_model_dataset_info(yaml_path='config/config.yaml'):
    """
    Output whether the currently used model is supervised or unsupervised,
    the model name, and the raw data path based on the YAML file.
    
    Args:
        yaml_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Information containing method type, model name, and raw data path.
    """
    data = load_yaml(yaml_path)
    if not data:
        logging.error("Unable to load YAML data, cannot retrieve model information.")
        return {}
    
    method_type = data.get('method', {}).get('type', 'Undefined')
    if method_type == 'supervised':
        model_name = data.get('method', {}).get('supervised', {}).get('model', {}).get('name', 'Undefined')
    elif method_type == 'unsupervised':
        model_name = data.get('method', {}).get('unsupervised', {}).get('method', 'Undefined')
    else:
        model_name = 'Undefined'
    
    raw_data_path = data.get('data', {}).get('raw_data_path', 'Undefined')
    
    model_info = {
        'Method Type': method_type,
        'Model Name': model_name,
        'Raw Data Path': raw_data_path,
        'data_info_path': f"{raw_data_path}/data_info.json"
    }
    
    logging.info(f"Model Information: {model_info}")
    return model_info

if __name__ == "__main__":
    setup_logging()
    
    # Example usage:
    config_yaml_path = 'config/config.yaml'  # Replace with your YAML file path
    
    # 1. Update method type and selection
    # For example, set method type to 'supervised' and select 'resnet50' as the model
    # update_method_type_and_selection(config_yaml_path, 'supervised', 'resnet50')
    
    # Or, set method type to 'unsupervised' and select 'dcae' as the method
    # update_method_type_and_selection(config_yaml_path, 'unsupervised', 'dcae')
    
    # 2. Update data paths
    # For example, manually set the data path and checkpoint path
    # update_data_paths(config_yaml_path, './data/BreastMNIST/', './checkpoints/BreastMNIST/')
    
    # 3. Automatically update data paths and checkpoint paths based on dataset name
    # For example, switch to 'ConcreteCrack'
    # change_dataset(config_yaml_path, 'ConcreteCrack')
    
    # 4. Get and print model information
    model_info = get_model_dataset_info(config_yaml_path)
    if model_info:
        print("Current Model Information:")
        print(f"Method Type: {model_info['Method Type']}")
        print(f"Model Name: {model_info['Model Name']}")
        print(f"Raw Data Path: {model_info['Raw Data Path']}")
