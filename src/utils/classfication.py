# load_model.py
import os
import yaml
import torch
from torchvision import transforms
from PIL import Image
import random
import logging
from src.llm.prompt import LLM_prompt_enhance, generate_reference_prompt
from src.data.get_and_store_dataset_info import get_recent_model_dataset_info

from src.models import (
    get_resnet_model, 
    get_alexnet_model, 
    get_vgg_model, 
    get_vit_model, 
    UnsupervisedViTAnomalyDetector,
    get_autoencoder_model,
    get_variational_autoencoder_model,
)
import numpy as np
import matplotlib.pyplot as plt

def setup_logging(log_file='model_utils.log'):
    """
    配置日志记录。
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - Line: %(lineno)d - %(message)s',
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
        labels (dict, optional): 类别标签映射字典，例如 {'0': 'Negative', '1': 'Positive'}。

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

class GradCAM:
    def __init__(self, model, target_layer):
        """
        初始化 GradCAM 实例。

        Args:
            model (torch.nn.Module): 预训练模型。
            target_layer (torch.nn.Module): 用于生成 Grad-CAM 的目标卷积层。
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        logging.info(f"初始化 GradCAM，目标层: {target_layer}")
        self._register_hooks()

    def _register_hooks(self):
        """
        注册前向和反向钩子以捕获激活和梯度。
        """
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        生成 Grad-CAM 热力图。

        Args:
            input_tensor (torch.Tensor): 预处理后的输入张量。
            class_idx (int, optional): 目标类别索引。如果为 None，则使用预测的最高概率类别。

        Returns:
            np.ndarray: 生成的热力图。
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        target = output[0, class_idx]
        target.backward()

        gradients = self.gradients  # [batch_size, channels, height, width]
        activations = self.activations  # [batch_size, channels, height, width]

        # 全局平均池化梯度
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [batch_size, channels, 1, 1]

        # 计算加权和
        weighted_activations = weights * activations  # [batch_size, channels, height, width]
        cam = torch.sum(weighted_activations, dim=1, keepdim=True)  # [batch_size, 1, height, width]

        # ReLU 激活
        cam = torch.relu(cam)

        # 归一化到 [0, 1]
        cam = cam - cam.min()
        cam = cam / cam.max()

        # 转换为 NumPy 数组
        cam = cam.cpu().numpy()[0, 0, :, :]

        return cam

def overlay_heatmap_on_image(img_path, heatmap, output_path, alpha=0.4, colormap=plt.cm.jet, min_size=227):
    """
    将热力图叠加到原始图像上并保存，确保输出图像至少为指定的最小尺寸。

    Args:
        img_path (str): 原始图像路径。
        heatmap (np.ndarray): 生成的热力图。
        output_path (str): 保存叠加图像的路径。
        alpha (float, optional): 热力图的透明度。默认值为 0.4。
        colormap: 颜色映射。默认为 jet。
        min_size (int): 输出图像的最小边长。默认为227。
    """
    # 打开原始图像
    img = Image.open(img_path).convert('RGB')
    
    # 计算调整后的尺寸，保持纵横比
    width, height = img.size
    aspect_ratio = width / height
    
    if width < min_size or height < min_size:
        if aspect_ratio > 1:  # 宽图
            new_width = max(min_size, int(min_size * aspect_ratio))
            new_height = min_size
        else:  # 高图
            new_width = min_size
            new_height = max(min_size, int(min_size / aspect_ratio))
            
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 调整热力图大小以匹配新的图像尺寸
        heatmap = Image.fromarray(heatmap)
        heatmap = heatmap.resize((new_width, new_height), Image.Resampling.LANCZOS)
        heatmap = np.array(heatmap)
    else:
        # 如果原始图像已经够大，则保持热力图与原始图像相同大小
        heatmap = Image.fromarray(heatmap)
        heatmap = heatmap.resize((width, height), Image.Resampling.LANCZOS)
        heatmap = np.array(heatmap)

    # 创建热力图
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap)
    heatmap = heatmap.resize(img.size, Image.Resampling.LANCZOS)
    heatmap = heatmap.convert("RGB")
    heatmap = np.array(heatmap)
    heatmap = colormap(heatmap[:, :, 0])[:, :, :3] * 255
    heatmap = Image.fromarray(np.uint8(heatmap))

    # 叠加图像
    overlayed_img = Image.blend(img, heatmap, alpha=alpha)

    # 保存叠加图像
    overlayed_img.save(output_path, quality=95)  # 使用较高的质量设置
    
    # 记录最终图像尺寸
    final_size = overlayed_img.size
    logging.info(f"已保存 Grad-CAM 可视化图像: {output_path}, 尺寸: {final_size[0]}x{final_size[1]}")
    
    return final_size
    
def generate_grad_cam_visualization(model_info, image_path, transform, labels=None, output_dir='data/output'):
    """
    使用 Grad-CAM 生成并保存可视化图像。

    Args:
        model_info (dict): 包含模型实例和类型的字典，例如 {'model': model, 'type': 'supervised'}。
        image_path (str): 图片文件路径。
        transform (torchvision.transforms.Compose): 图片预处理转换。
        labels (dict, optional): 类别标签映射字典，例如 {'0': 'Negative', '1': 'Positive'}。
        output_dir (str, optional): 保存可视化图像的目录。默认值为 'data/output'。

    Returns:
        str: 生成的可视化图像的 URL，例如 'localhost:5100/DL-api/output/{照片}'。
    """
    if model_info['type'] != 'supervised':
        logging.warning("Grad-CAM 仅适用于有监督模型。")
        return

    model = model_info['model']
    device = next(model.parameters()).device

    logging.info(f"使用的模型: {model.__class__.__name__}，设备: {device}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开并预处理图片
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # 增加 batch 维度
    logging.info(f"已加载并预处理图片: {image_path}")

    # 确定目标层（以 ResNet50 为例）
    if isinstance(model, torch.nn.Module):
        if 'resnet' in model.__class__.__name__.lower():
            target_layer = model.layer4[-1].conv3
        elif 'alexnet' in model.__class__.__name__.lower():
            target_layer = model.features[-1]
        elif 'vgg' in model.__class__.__name__.lower():
            target_layer = model.features[-1]
        else:
            logging.error("不支持的模型类型用于 Grad-CAM。")
            return
    else:
        logging.error("无效的模型实例。")
        return

    logging.info(f"选择的目标层: {target_layer}")

    # 初始化 Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    # 生成热力图
    heatmap = grad_cam.generate_heatmap(input_tensor)

    # 获取预测的类别索引
    with torch.no_grad():
        outputs = model(input_tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # 处理如 VAE 等模型的输出
        probabilities = torch.softmax(outputs, dim=1)
        top_prob, top_class = probabilities.topk(1, dim=1)
        class_idx = top_class.item()
        class_label = labels.get(str(class_idx), "Unknown") if labels else str(class_idx)
        logging.info(f"预测类别: {class_label} (索引: {class_idx}), 概率: {top_prob.item():.4f}")

    # 定义输出路径
    image_name = os.path.basename(image_path)
    output_filename = f'grad_cam_{image_name}'
    output_path = os.path.join(output_dir, output_filename)

    # 叠加热力图并保存
    overlay_heatmap_on_image(image_path, heatmap, output_path)

    # 构建返回的 URL
    # 假设 data/output 对应的 URL 路径为 localhost:5100/DL-api/output/
    url = f'http://localhost:5100/DL-api/output/{output_filename}'
    logging.info(f"Grad-CAM 可视化图像的 URL: {url}")

    # 返回 URL
    return url

def main(image_path="random", config_path='config/config.yaml'):
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
    if image_path == "random":
        test_image_path = random.choice(image_files)
        logging.info(f"the img path is: {test_image_path}")
    else:
        test_image_path = image_path
        logging.info(f"the img path is: {test_image_path}")

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
        labels_mapping = { '0': 'Negative', '1': 'Positive' }
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

    DeepLearningModelAndDataSetMessage = f"模型名称: {model_abbreviation}，数据集: {dataset_name}"
    
    DeepLearningModelAndDataSetMessage = get_recent_model_dataset_info()
    
    # 生成增强后的提示词
    try:
        enhanced_prompt = LLM_prompt_enhance(model_info, classification_result, DeepLearningModelAndDataSetMessage, logging)
    except Exception as e:
        logging.error(f"生成提示词失败: {str(e)}")
        return

    # 生成 Grad-CAM 可视化图像
    try:
        grad_cam_output_url = generate_grad_cam_visualization(
            model_info=model_info,
            image_path=test_image_path,
            transform=transform,
            labels=labels_mapping,
            output_dir='data/output'  # 设置默认输出目录为 data/output
        )
        logging.info(f"Grad-CAM visualize URL: {grad_cam_output_url}")
        enhanced_prompt += generate_reference_prompt(grad_cam_output_url)
        
    except Exception as e:
        logging.error(f"生成 Grad-CAM 可视化失败: {str(e)}")

    # 输出增强后的提示词
    # logging.info("prompt: ")
    # logging.info(enhanced_prompt)
    return enhanced_prompt

# if __name__ == '__main__':
# please run the __classification_test.py to test the function, the __main__ function is not working in this file
#     main()
