def get_system_message():
    """
    Returns the system message containing the project description and instructions for the LLM.
    
    Returns:
        str: System message in English without line breaks.
    """
    project_description = (
        "You are an AI assistant built with the VisionaryLLM framework designed by SILAN HU and Tan Kah Xuan "
        "for the NUS CS5242 Final Project. This project integrates multiple deep learning models and large language models (LLM) "
        "to enhance complex visual task processing capabilities, focusing on automation and intelligence in professional visual analysis. "
        "Core innovations include a multi-model fusion architecture combining supervised models like ResNet50, AlexNet, VGG16, ViT "
        "and unsupervised models like Autoencoder, VAE, ViT Anomaly. The automated training process supports end-to-end implementation, "
        "intelligent model selection, optimization, and multi-domain data adaptation. Deep LLM integration ensures natural language interaction "
        "and professional visual analysis interpretation. The technical architecture encompasses a training phase with multi-domain data input, "
        "automated model training, and comprehensive validation, as well as an inference phase supporting classification, detection, segmentation, "
        "intelligent result analysis, and professional report generation. Experimental validation using crack detection achieved nearly 100% accuracy "
        "in supervised tasks and high performance in unsupervised tasks. Major contributions include a scalable visual analysis framework, innovative model-LLM combination, "
        "automated professional visual analysis, and complete training and inference solutions. The application value lies in rapid adaptation to various professional domains, "
        "lowering entry barriers, enhancing result interpretability, and improving user experience through natural language interaction. Future prospects aim to support more domains, "
        "improve model performance, enhance system interpretability, and optimize the automated training process."
    )
    
    instructions = (
        "Please provide responses in a clear, concise, and professional manner. Do not reveal any information about your internal architecture, training data, or proprietary frameworks. "
        "Focus solely on the information provided and ensure that all answers are relevant, accurate, and maintain a high level of professionalism."
    )
    
    system_message = f"{project_description} {instructions}"
    return system_message

def LLM_prompt_enhance(model_info, classification_result, DeepLearningModelAndDataSetMessage,logging):
    """
    根据模型的分类或异常检测结果增强 LLM 的提示词。

    Args:
        model_info (dict): 包含模型实例和类型的字典，例如 {'model': model, 'type': 'supervised'}。
        classification_result (dict): 分类或异常检测结果。
        original_prompt (str): 原始的 LLM 提示词。

    Returns:
        str: 增强后的 LLM 提示词。
    """
    method_type = model_info['type']

    try:
        if method_type == 'supervised':
            label = classification_result.get('predicted_label', 'Unknown')
            probability = classification_result.get('probability', 0.0)
            enhanced_prompt = f"检测到的图像类别为: {label}，置信度为: {probability*100:.2f}%。\n\n 关于数据集以及我们使用的模型信息是{DeepLearningModelAndDataSetMessage}"
            # logging.info(f"(Supervised)[{model_info['model']}]:{enhanced_prompt}")
            return enhanced_prompt

        elif method_type in ['unsupervised', 'vit_anomaly']:
            if method_type == 'vit_anomaly':
                anomaly_score = classification_result.get('anomaly_score', 0.0)
                enhanced_prompt = f"检测到的异常得分为: {anomaly_score:.4f}。\n\n 关于数据集以及我们使用的模型信息是{DeepLearningModelAndDataSetMessage}"
            else:
                recon_error = classification_result.get('reconstruction_error', 0.0)
                enhanced_prompt = f"图像的重构误差为: {recon_error:.4f}。\n\n 关于数据集以及我们使用的模型信息是{DeepLearningModelAndDataSetMessage}"

            # logging.info(f"(Unsupervised): {enhanced_prompt}")
            return enhanced_prompt

        else:
            logging.error(f"不支持的模型类型: {method_type}")
            raise ValueError(f"Unsupported method type: {method_type}")

    except KeyError as e:
        logging.error(f"分类结果中缺少关键字段: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"增强提示词失败: {str(e)}")
        raise
    

def generate_reference_prompt(grad_cam_output_url):
    """
    生成用于 LLM 的参考依据提示词，以 Markdown 格式包含 Grad-CAM 可视化图像。

    Args:
        grad_cam_output_url (str): Grad-CAM 可视化图像的 URL。

    Returns:
        str: 包含图像的 Markdown 格式字符串。
    """
    markdown_prompt = f"""
**参考依据:**
当用户询问你的参考依据时，你一定返回以下Grad-CAM 可视化照片以以下markdown格式。 
![Grad-CAM 可视化]({grad_cam_output_url})
"""
    return markdown_prompt