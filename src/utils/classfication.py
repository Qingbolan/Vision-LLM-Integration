# load_model.py
import os
import yaml
import torch
from torchvision import transforms
from PIL import Image
import random
import logging
from src.llm.prompt import LLM_prompt_enhance

from src.models import (
    get_resnet_model, 
    get_alexnet_model, 
    get_vgg_model, 
    get_vit_model, 
    UnsupervisedViTAnomalyDetector,
    get_autoencoder_model,
    get_variational_autoencoder_model,
)
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

def check_pth_files(model_name, checkpoint_dir):
    """
    检查指定模型的 .pth 文件是否存在于指定目录。

    Args:
        model_name (str): 模型名称。
        checkpoint_dir (str): 检查点文件所在目录。

    Returns:
        bool: 如果模型文件存在，返回 True，否则返回 False。
    """
    model_file = os.path.join(checkpoint_dir, f'{model_name}_final.pth')
    exists = os.path.isfile(model_file)
    if exists:
        logging.info(f"找到模型文件: {model_file}")
    else:
        logging.warning(f"未找到模型文件: {model_file}")
    return exists

def get_model_instance(model_config, device):
    """
    根据模型配置获取模型实例。

    Args:
        model_config (dict): 模型的配置字典。
        device (torch.device): 设备。

    Returns:
        dict: 包含模型实例和模型类型的字典。例如 {'model': model, 'type': 'supervised'}
    """
    model = get_model(model_config, device)
    method_type = model_config.get('method_type', 'supervised').lower()
    return {'model': model, 'type': method_type}

def get_model(model_config, device):
    """
    根据模型配置获取具体的模型实例。

    Args:
        model_config (dict): 模型的配置字典。
        device (torch.device): 设备。

    Returns:
        torch.nn.Module: 模型实例。
    """
    model_name = model_config.get('name').lower()
    pretrained = model_config.get('pretrained', False)
    num_classes = model_config.get('num_classes', 2)
    print(f"Model: {model_name}")
    print(f"Pretrained: {pretrained}")

    if model_name == 'resnet50':
        return get_resnet_model(pretrained=pretrained, num_classes=num_classes).to(device)
    elif model_name == 'alexnet':
        return get_alexnet_model(pretrained=pretrained, num_classes=num_classes).to(device)
    elif model_name == 'vgg16':
        return get_vgg_model(pretrained=pretrained, num_classes=num_classes).to(device)
    elif model_name == 'vit':
        return get_vit_model(pretrained=pretrained, num_classes=num_classes).to(device)
    elif model_name in ['dcae', 'cae']:
        encoded_space_dim = model_config.get('encoded_space_dim', 128)
        return get_autoencoder_model(encoded_space_dim=encoded_space_dim).to(device)
    elif model_name in ['dcvae', 'cvae']:
        encoded_space_dim = model_config.get('encoded_space_dim', 128)
        return get_variational_autoencoder_model(encoded_space_dim=encoded_space_dim).to(device)
    elif model_name == 'vit_anomaly':
        img_size = model_config.get('img_size', 224)
        patch_size = model_config.get('patch_size', 16)
        embed_dim = model_config.get('embed_dim', 768)
        num_heads = model_config.get('num_heads', 12)
        mlp_dim = model_config.get('mlp_dim', 3072)
        num_layers = model_config.get('num_layers', 12)
        
        # 新增的参数
        latent_dim = model_config.get('latent_dim', 240)  
        pretrained = model_config.get('pretrained', False)  
        
        # 初始化无监督异常检测模型
        model = UnsupervisedViTAnomalyDetector(
            pretrained=pretrained,
            latent_dim=latent_dim,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers
        ).to(device)
        
        return model
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def classification_by_model(model_info, image_path, transform, labels=None):
    """
    使用加载好的模型对单张图片进行分类或异常检测。

    Args:
        model_info (dict): 包含模型实例和类型的字典，例如 {'model': model, 'type': 'supervised'}。
        image_path (str): 图片文件路径。
        transform (torchvision.transforms.Compose): 图片预处理转换。
        labels (dict, optional): 类别标签映射字典，例如 {'0': '正常', '1': '裂缝'}。

    Returns:
        dict: 包含预测结果的字典，格式取决于模型类型。
    """
    model = model_info['model']
    method_type = model_info['type']

    device = model.device if hasattr(model, 'device') else next(model.parameters()).device

    try:
        # 打开并预处理图片
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)  # 增加 batch 维度
        logging.info(f"已加载并预处理图片: {image_path}")

        with torch.no_grad():
            if method_type == 'supervised':
                outputs = model(input_tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # 处理如 VAE 等模型的输出
                probabilities = torch.softmax(outputs, dim=1)
                top_prob, top_class = probabilities.topk(1, dim=1)

                if labels is None:
                    logging.error("缺少标签映射字典。")
                    raise ValueError("缺少标签映射字典。")

                predicted_label = labels.get(str(top_class.item()), "Unknown")
                result = {
                    'predicted_label': predicted_label,
                    'probability': top_prob.item()
                }
                logging.info(f"图片分类结果: {result}")
                return result

            elif method_type in ['unsupervised', 'vit_anomaly']:
                # 对于无监督模型，计算重构误差或异常得分
                if method_type == 'vit_anomaly':
                    # 假设 vit_anomaly 模型返回 logits 或其他输出
                    outputs = model(input_tensor)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # 取第一个输出
                    probabilities = torch.softmax(outputs, dim=1)
                    anomaly_score = probabilities[:, 1].item()  # 假设类别1为异常
                    result = {
                        'anomaly_score': anomaly_score
                    }
                    logging.info(f"异常检测结果: {result}")
                    return result
                else:
                    # 对于 autoencoder 或 variational autoencoder
                    outputs = model(input_tensor)
                    if isinstance(outputs, tuple):
                        if len(outputs) == 3:
                            # VAE 模型返回 (reconstructed, mu, logvar)
                            reconstructed = outputs[0]
                        else:
                            # 其他返回形式
                            reconstructed = outputs[0]
                    else:
                        reconstructed = outputs

                    # 计算重构误差
                    recon_error = torch.mean((reconstructed - input_tensor) ** 2).item()
                    result = {
                        'reconstruction_error': recon_error
                    }
                    logging.info(f"重构误差: {result}")
                    return result

            else:
                logging.error(f"不支持的模型类型: {method_type}")
                raise ValueError(f"Unsupported method type: {method_type}")

    except Exception as e:
        logging.error(f"图片分类或异常检测失败: {str(e)}")
        raise



def get_image_files(dataset_path):
    """
    获取指定目录下所有支持格式的图像文件路径。

    Args:
        dataset_path (str): 数据集根目录路径。

    Returns:
        list: 图像文件路径列表。
    """
    supported_formats = ('.png', '.jpg', '.jpeg')
    image_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_files.append(os.path.join(root, file))
    return image_files

def main(config_path='config/config.yaml', ORIGINAL_PROMPT = "请根据以下信息回答问题。"):
    """
    主函数，用于测试深度模型的调用，包括数据集选择、模型加载、图像选择和预测生成提示词。
    """
    # 配置日志记录
    setup_logging()

    # Load configuration
    if not os.path.isfile(config_path):
        logging.error(f"配置文件不存在: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 数据相关配置
    data_config = config.get('data', {})
    raw_data_path = data_config.get('raw_data_path', './data/raw/')
    processed_data_path = data_config.get('processed_data_path', './data/processed/')
    image_size = data_config.get('image_size', 224)

    # 方法相关配置
    method_config = config.get('method', {})
    method_type = method_config.get('type', 'supervised').lower()

    # 选择模型简称和配置
    model_abbreviation = None
    model_configs = {}
    if method_type == 'supervised':
        supervised_config = method_config.get('supervised', {}).get('model', {})
        model_abbreviation = supervised_config.get('name', 'resnet50')
        model_configs = {
            'method_type': 'supervised',
            'name': supervised_config.get('name', 'resnet50'),
            'pretrained': supervised_config.get('pretrained', False),
            'num_classes': supervised_config.get('num_classes', 2),
            'learning_rate': supervised_config.get('learning_rate', 0.0001),
            'weight_decay': supervised_config.get('weight_decay', 0.00001),
            'num_epochs': supervised_config.get('num_epochs', 10),
            'device': supervised_config.get('device', 'cuda')
        }
    elif method_type == 'unsupervised':
        unsupervised_method = method_config.get('unsupervised', {}).get('method', 'dcae').lower()
        model_abbreviation = unsupervised_method
        unsupervised_model_config = method_config.get('unsupervised', {}).get(unsupervised_method, {}).get('model', {})
        if unsupervised_method == 'vit_anomaly':
            model_configs = {
                'method_type': 'unsupervised',
                'method': unsupervised_method,
                'name': unsupervised_model_config.get('name'),
                'pretrained': unsupervised_model_config.get('pretrained', False),
                'encoded_space_dim': unsupervised_model_config.get('encoded_space_dim', 256),
                'learning_rate': unsupervised_model_config.get('learning_rate', 0.0001),
                'weight_decay': unsupervised_model_config.get('weight_decay', 0.00001),
                'num_epochs': unsupervised_model_config.get('num_epochs', 1),
                'device': unsupervised_model_config.get('device', 'cuda'),
                'img_size': unsupervised_model_config.get('img_size', 224),
                'patch_size': unsupervised_model_config.get('patch_size', 16),
                'embed_dim': unsupervised_model_config.get('embed_dim', 768),
                'num_heads': unsupervised_model_config.get('num_heads', 12),
                'mlp_dim': unsupervised_model_config.get('mlp_dim', 3072),
                'num_layers': unsupervised_model_config.get('num_layers', 12),
                'latent_dim': unsupervised_model_config.get('latent_dim', 240)
            }
        else:
            model_configs = {
                'method_type': 'unsupervised',
                'method': unsupervised_method,
                'name': unsupervised_model_config.get('name'),
                'pretrained': unsupervised_model_config.get('pretrained', False),
                'encoded_space_dim': unsupervised_model_config.get('encoded_space_dim', 256),
                'learning_rate': unsupervised_model_config.get('learning_rate', 0.0001),
                'weight_decay': unsupervised_model_config.get('weight_decay', 0.00001),
                'num_epochs': unsupervised_model_config.get('num_epochs', 10),
                'device': unsupervised_model_config.get('device', 'cuda')
                # 添加其他参数如有需要
            }
    else:
        logging.error(f"不支持的 method type: {method_type}")
        return

    # 设置数据集路径
    if method_type == 'supervised':
        dataset_name = 'raw'  # 假设 supervised 使用 raw 数据集
        dataset_path = raw_data_path
    elif method_type == 'unsupervised':
        # 假设 unsupervised 使用 processed_data_path 下的具体方法名称子目录
        dataset_name = method_config.get('unsupervised', {}).get('method', 'dcae').lower()
        dataset_path = os.path.join(processed_data_path, dataset_name)
    else:
        logging.error(f"不支持的 method type: {method_type}")
        return

    if not os.path.isdir(dataset_path):
        logging.error(f"数据集路径不存在或不是目录: {dataset_path}")
        return

    # 设置检查点路径
    training_config = config.get('training', {})
    checkpoint_base_path = training_config.get('checkpoint_path', './checkpoints/')
    if dataset_name == 'raw':
        checkpoint_dir = checkpoint_base_path
    else:
        checkpoint_dir = os.path.join(checkpoint_base_path, dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 检查模型文件是否存在
    if not check_pth_files(model_abbreviation, checkpoint_dir):
        logging.error(f"模型文件不存在: {os.path.join(checkpoint_dir, f'{model_abbreviation}_final.pth')}")
        return

    # 加载模型实例
    try:
        device = torch.device(model_configs['device'] if torch.cuda.is_available() else 'cpu')
        model_info = get_model_instance(model_configs, device=device)
    except Exception as e:
        logging.error(f"模型初始化失败: {str(e)}")
        return

    # 加载模型权重
    model_file = os.path.join(checkpoint_dir, f'{model_abbreviation}_final.pth')
    try:
        # 修正 map_location 使用 device 变量
        state_dict = torch.load(model_file, map_location=device)
        model_info['model'].load_state_dict(state_dict)
        model_info['model'].to(device)
        model_info['model'].eval()
        logging.info(f"成功加载模型权重: {model_file}")
    except Exception as e:
        logging.error(f"加载模型权重失败: {str(e)}")
        return

    # 获取所有图像文件
    image_files = get_image_files(dataset_path)
    if not image_files:
        logging.error(f"在数据集路径中未找到任何图像文件: {dataset_path}")
        return

    # 随机选择一张图像作为测试
    test_image_path = random.choice(image_files)
    logging.info(f"随机选择的测试图像: {test_image_path}")

    # 定义图像预处理转换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 定义标签映射（仅适用于有监督模型）
    if model_info['type'] == 'supervised':
        # 根据配置文件中的标签数量动态生成标签映射
        # 假设类别从 0 开始
        num_classes = model_configs.get('num_classes', 2)
        # 示例标签映射，可以根据实际情况调整
        labels_mapping = { '0': '正常', '1': '裂缝' }
    else:
        labels_mapping = None  # 无监督模型不需要标签映射

    # 进行分类或异常检测
    try:
        classification_result = classification_by_model(
            model_info=model_info,
            image_path=test_image_path,
            transform=transform,
            labels=labels_mapping
        )
    except Exception as e:
        logging.error(f"classification failed: {str(e)}")
        return

    # 生成增强后的提示词
    try:
        enhanced_prompt = LLM_prompt_enhance(model_info, classification_result, ORIGINAL_PROMPT, logging)
    except Exception as e:
        logging.error(f"生成提示词失败: {str(e)}")
        return

    # 输出增强后的提示词
    logging.info("prompt: ")
    logging.info(enhanced_prompt)
    print("增强后的 LLM 提示词:")
    print(enhanced_prompt)

# if __name__ == '__main__':
# please run the __classification_test.py to test the function, the __main__ function is not working in this file
#     main()